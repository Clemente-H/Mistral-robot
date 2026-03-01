"""
PyBullet simulator wrapper — KUKA arm + scene objects.
Runs headless (no GUI) for screenshot capture to send to Cosmos Reason2.
"""
import os
import math
import time
import numpy as np
import pybullet as p
import pybullet_data


class RobotSimulator:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.physics_client = None
        self.kuka_id = None
        self.objects = {}  # name -> body_id
        self.num_joints = 0
        self._recording = False
        self._recorded_frames: list = []
        self._recording_joints = False
        self._joint_frames: list = []
        # Camera state — spherical coords around scene center
        self.camera_azimuth = 0.0     # degrees, 0 = look from +X side
        self.camera_elevation = 42.0  # degrees above horizontal
        self.camera_distance = 1.5    # meters from target
        self.camera_target = [0.4, 0, 0.2]
        self._held_constraint = None  # PyBullet constraint ID when gripping
        self._held_object = None      # name of the gripped object

    def start(self):
        mode = p.DIRECT if self.headless else p.GUI
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.kuka_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )
        self.num_joints = p.getNumJoints(self.kuka_id)
        self._reset_arm()
        self._spawn_scene_objects()

    def _reset_arm(self):
        home = [0, -math.pi / 4, 0, math.pi / 2, 0, -math.pi / 4, 0]
        for i, angle in enumerate(home[:self.num_joints]):
            p.resetJointState(self.kuka_id, i, angle)

    def _spawn_scene_objects(self):
        # Blue box
        box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                                      rgbaColor=[0.2, 0.4, 1.0, 1.0])
        self.objects["blue_box"] = p.createMultiBody(
            baseMass=0.5, baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=box_vis, basePosition=[0.5, 0.1, 0.05])

        # Red sphere
        sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                         rgbaColor=[1.0, 0.2, 0.2, 1.0])
        self.objects["red_sphere"] = p.createMultiBody(
            baseMass=0.3, baseCollisionShapeIndex=sphere_col,
            baseVisualShapeIndex=sphere_vis, basePosition=[0.6, -0.2, 0.04])

        # Yellow cylinder
        cyl_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.1)
        cyl_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.1,
                                      rgbaColor=[0.98, 0.82, 0.05, 1.0])
        self.objects["yellow_cylinder"] = p.createMultiBody(
            baseMass=0.3, baseCollisionShapeIndex=cyl_col,
            baseVisualShapeIndex=cyl_vis, basePosition=[0.3, 0.4, 0.05])

        # Green cube
        cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04])
        cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04],
                                       rgbaColor=[0.1, 0.8, 0.2, 1.0])
        self.objects["green_cube"] = p.createMultiBody(
            baseMass=0.3, baseCollisionShapeIndex=cube_col,
            baseVisualShapeIndex=cube_vis, basePosition=[0.4, -0.38, 0.04])

        # Purple box
        purp_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04])
        purp_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04],
                                       rgbaColor=[0.6, 0.15, 0.9, 1.0])
        self.objects["purple_box"] = p.createMultiBody(
            baseMass=0.3, baseCollisionShapeIndex=purp_col,
            baseVisualShapeIndex=purp_vis, basePosition=[0.7, 0.25, 0.04])

        # Cyan sphere
        cyan_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        cyan_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                       rgbaColor=[0.0, 0.85, 0.85, 1.0])
        self.objects["cyan_sphere"] = p.createMultiBody(
            baseMass=0.3, baseCollisionShapeIndex=cyan_col,
            baseVisualShapeIndex=cyan_vis, basePosition=[0.22, 0.05, 0.04])

        # Pink cylinder
        pink_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.035, height=0.08)
        pink_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.08,
                                       rgbaColor=[1.0, 0.35, 0.65, 1.0])
        self.objects["pink_cylinder"] = p.createMultiBody(
            baseMass=0.3, baseCollisionShapeIndex=pink_col,
            baseVisualShapeIndex=pink_vis, basePosition=[0.72, -0.35, 0.04])

    # ------------------------------------------------------------------
    # Actions exposed as agent tools
    # ------------------------------------------------------------------

    # KUKA iiwa joint limits (radians) — used to bias IK toward natural posture
    _IK_LL = [-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054]
    _IK_UL = [ 2.967,  2.094,  2.967,  2.094,  2.967,  2.094,  3.054]
    _IK_JR = [ 5.934,  4.188,  5.934,  4.188,  5.934,  4.188,  6.108]
    _IK_RP = [0, -math.pi / 4, 0, math.pi / 2, 0, -math.pi / 4, 0]  # home pose

    def move_to(self, x: float, y: float, z: float):
        """Move end-effector to (x, y, z) in world coordinates.
        Uses restPoses + joint limits so IK always prefers the natural elbow-up posture.
        """
        end_effector_idx = self.num_joints - 1
        # EE pointing straight down — natural pick-and-place orientation
        target_orn = p.getQuaternionFromEuler([0, math.pi, 0])
        joint_poses = p.calculateInverseKinematics(
            self.kuka_id, end_effector_idx,
            targetPosition=[x, y, z],
            targetOrientation=target_orn,
            lowerLimits=self._IK_LL,
            upperLimits=self._IK_UL,
            jointRanges=self._IK_JR,
            restPoses=self._IK_RP,
        )
        for i, pos in enumerate(joint_poses[:self.num_joints]):
            p.setJointMotorControl2(
                self.kuka_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos,
                force=500,
            )
        self._step(120)

    def grab(self):
        """Attach nearest object to end-effector via a fixed constraint so it lifts with the arm."""
        ee_idx = self.num_joints - 1
        ee_state = p.getLinkState(self.kuka_id, ee_idx)
        ee_pos = ee_state[0]

        # Find closest scene object within 0.18 m of the end-effector
        closest_name, closest_id, closest_dist = None, None, 0.18
        for name, body_id in self.objects.items():
            obj_pos, _ = p.getBasePositionAndOrientation(body_id)
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(ee_pos, obj_pos)))
            if d < closest_dist:
                closest_dist = d
                closest_name, closest_id = name, body_id

        if closest_id is not None and self._held_constraint is None:
            # parentFramePosition: offset from EE link origin to grip point (slightly below tip)
            self._held_constraint = p.createConstraint(
                self.kuka_id, ee_idx,
                closest_id, -1,
                p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.03],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(self._held_constraint, maxForce=500)
            self._held_object = closest_name
            print(f"  [grip] attached {closest_name} (dist={closest_dist:.3f}m)")

        self._step(30)

    def release(self):
        """Remove grip constraint so the object falls under gravity."""
        if self._held_constraint is not None:
            p.removeConstraint(self._held_constraint)
            self._held_constraint = None
            self._held_object = None
        self._step(30)

    def _smooth_reset(self):
        """Return to home using motor control so the motion is animated."""
        home = [0, -math.pi / 4, 0, math.pi / 2, 0, -math.pi / 4, 0]
        for i, angle in enumerate(home[:self.num_joints]):
            p.setJointMotorControl2(
                self.kuka_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=300,
            )
        self._step(100)

    def reset(self):
        """Return arm to home position smoothly."""
        self._smooth_reset()

    def wave(self):
        """Wave hello — arm fully vertical, only wrist (j5) swings side-to-side like a metronome."""
        # Upper arm horizontal forward (j1=-π/2), elbow bent up (j3=-π/2) → forearm vertical
        # j4=π/2 orients the wrist so j5 oscillates left-right like a metronome
        wave_pose = [0, -math.pi / 2, 0, -math.pi / 2, math.pi / 2, 0, math.pi / 2]
        for i, angle in enumerate(wave_pose[:self.num_joints]):
            p.setJointMotorControl2(
                self.kuka_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=500,
            )
        self._step(80)

        # j5 oscillates side-to-side like a metronome; hold all other joints firm
        for t in range(300):
            osc = 0.5 * math.sin(t * 0.12)
            p.setJointMotorControl2(self.kuka_id, 0, p.POSITION_CONTROL,
                                    targetPosition=0, force=500)
            p.setJointMotorControl2(self.kuka_id, 1, p.POSITION_CONTROL,
                                    targetPosition=-math.pi / 2, force=500)
            p.setJointMotorControl2(self.kuka_id, 2, p.POSITION_CONTROL,
                                    targetPosition=0, force=500)
            p.setJointMotorControl2(self.kuka_id, 3, p.POSITION_CONTROL,
                                    targetPosition=-math.pi / 2, force=500)
            p.setJointMotorControl2(self.kuka_id, 4, p.POSITION_CONTROL,
                                    targetPosition=math.pi / 2, force=500)
            p.setJointMotorControl2(self.kuka_id, 5, p.POSITION_CONTROL,
                                    targetPosition=osc, force=400)
            p.setJointMotorControl2(self.kuka_id, 6, p.POSITION_CONTROL,
                                    targetPosition=math.pi / 2, force=500)
            self._step(1)

        self._smooth_reset()

    def dance(self):
        """Choreographed polyrhythm dance — multiple joints at different frequencies."""
        # Start pose
        pose1 = [0, -math.pi / 3, 0, math.pi / 4, 0, -math.pi / 6, 0]
        for i, angle in enumerate(pose1[:self.num_joints]):
            p.setJointMotorControl2(self.kuka_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=500)
        self._step(60)

        # Polyrhythm: 4 joints moving at different frequencies simultaneously
        for t in range(450):
            # Base sways slowly
            p.setJointMotorControl2(self.kuka_id, 0, p.POSITION_CONTROL,
                                    targetPosition=0.5 * math.sin(t * 0.02), force=350)
            # Shoulder bobs at a different rhythm
            p.setJointMotorControl2(self.kuka_id, 1, p.POSITION_CONTROL,
                                    targetPosition=-math.pi / 3 + 0.18 * math.sin(t * 0.03 + 1.0), force=400)
            # Elbow pumps faster
            p.setJointMotorControl2(self.kuka_id, 3, p.POSITION_CONTROL,
                                    targetPosition=math.pi / 3 + 0.3 * math.cos(t * 0.05), force=400)
            # Wrist spins fastest (sinusoidal — never hits limits)
            p.setJointMotorControl2(self.kuka_id, 6, p.POSITION_CONTROL,
                                    targetPosition=0.9 * math.sin(t * 0.08), force=300)
            self._step(1)

        # Bow finale
        bow_pose = [0, math.pi / 8, 0, math.pi / 2, 0, math.pi / 4, 0]
        for i, angle in enumerate(bow_pose[:self.num_joints]):
            p.setJointMotorControl2(self.kuka_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=500)
        self._step(90)

    def sweep(self):
        """Scan the workspace with a slow arc — arm extended, rotating base left to right."""
        # Extend arm into scanning pose
        scan_pose = [0, -math.pi / 4, 0, math.pi / 3, 0, -math.pi / 4, 0]
        for i, angle in enumerate(scan_pose[:self.num_joints]):
            p.setJointMotorControl2(self.kuka_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=500)
        self._step(80)

        # Slow arc: left → right → center
        for target in [-math.pi / 2.5, math.pi / 2.5, 0]:
            p.setJointMotorControl2(self.kuka_id, 0, p.POSITION_CONTROL,
                                    targetPosition=target, force=150)  # slow
            self._step(110)

        self._reset_arm()
        self._step(60)

    def helicopter(self):
        """Helicopter — arm extended horizontally, j0 (base) spins via VELOCITY_CONTROL."""
        # Extend arm outward (near-horizontal) so the whole arm sweeps like a rotor blade
        extend_pose = [0, -math.pi / 5, 0, math.pi / 4, 0, math.pi / 5, 0]
        for i, angle in enumerate(extend_pose[:self.num_joints]):
            p.setJointMotorControl2(self.kuka_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=500)
        self._step(70)

        # Spin base (j0) with velocity control — hits ±170° limits and bounces,
        # but visually reads as continuous rotation
        p.setJointMotorControl2(self.kuka_id, 0, p.VELOCITY_CONTROL,
                                targetVelocity=5.0, force=800)
        for t in range(480):
            # Hold arm extension while base spins
            p.setJointMotorControl2(self.kuka_id, 1, p.POSITION_CONTROL,
                                    targetPosition=-math.pi / 5, force=450)
            p.setJointMotorControl2(self.kuka_id, 3, p.POSITION_CONTROL,
                                    targetPosition=math.pi / 4, force=400)
            self._step(1)

        # Stop spin
        p.setJointMotorControl2(self.kuka_id, 0, p.VELOCITY_CONTROL,
                                targetVelocity=0, force=800)
        self._step(20)
        self._smooth_reset()

    def salute(self):
        """Military salute — arm raises to the side, brief hold, then lower."""
        salute_pose = [0.45, -math.pi / 2.5, 0, math.pi / 2.2, 0, -math.pi / 5, 0]
        for i, angle in enumerate(salute_pose[:self.num_joints]):
            p.setJointMotorControl2(self.kuka_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=500)
        self._step(90)

        # Crisp hold
        self._step(60)

        # Slight forward lean (bow-like finish)
        bow_pose = list(salute_pose)
        bow_pose[1] = -math.pi / 4
        for i, angle in enumerate(bow_pose[:self.num_joints]):
            p.setJointMotorControl2(self.kuka_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=500)
        self._step(50)
        self._smooth_reset()

    def push(self, x: float, y: float, z: float):
        """Push the object at (x, y, z) away from the robot along the +X axis."""
        approach_x = max(0.2, x - 0.14)
        self.move_to(approach_x, y, z + 0.06)   # approach above and behind
        self.move_to(approach_x, y, z + 0.01)   # lower to object height
        self.move_to(x + 0.20, y, z + 0.01)     # push through — physics handles the rest
        self._step(60)

    def start_recording(self):
        self._recording = True
        self._recorded_frames = []

    def stop_recording(self) -> list:
        self._recording = False
        frames = self._recorded_frames
        self._recorded_frames = []
        return frames

    # --- 3D keyframe recording (for Three.js playback) ---
    def start_recording_joints(self):
        self._recording_joints = True
        self._joint_frames = []

    def stop_recording_joints(self) -> list:
        self._recording_joints = False
        frames = self._joint_frames
        self._joint_frames = []
        return frames

    def get_link_positions(self) -> list:
        """Return world-space [x,y,z] for base + each link (8 points total)."""
        positions = [[0.0, 0.0, 0.0]]  # fixed base
        for i in range(self.num_joints):
            state = p.getLinkState(self.kuka_id, i)
            positions.append(list(state[0]))
        return positions

    def get_objects_state(self) -> dict:
        """Return world positions of all scene objects."""
        result = {}
        for name, body_id in self.objects.items():
            pos, _ = p.getBasePositionAndOrientation(body_id)
            result[name] = list(pos)
        return result

    def _get_3d_frame(self) -> dict:
        return {"links": self.get_link_positions(), "objects": self.get_objects_state()}

    def _step(self, steps: int = 60):
        for i in range(steps):
            p.stepSimulation()
            if not self.headless:
                time.sleep(1.0 / 240.0)
            if self._recording and i % 6 == 0:  # ~1 frame every 6 steps
                self._recorded_frames.append(self.get_screenshot())
            if self._recording_joints and i % 4 == 0:  # ~60 fps of 3D data
                self._joint_frames.append(self._get_3d_frame())

    # ------------------------------------------------------------------
    # Scene state
    # ------------------------------------------------------------------

    def get_scene_state(self) -> dict:
        """Return positions of all tracked objects + end-effector."""
        state = {}
        end_effector_idx = self.num_joints - 1
        ee_state = p.getLinkState(self.kuka_id, end_effector_idx)
        state["end_effector"] = list(ee_state[0])
        for name, body_id in self.objects.items():
            pos, _ = p.getBasePositionAndOrientation(body_id)
            state[name] = list(pos)
        return state

    def set_camera(self, azimuth: float = None, elevation: float = None, distance: float = None):
        """Update camera spherical coordinates."""
        if azimuth is not None:
            self.camera_azimuth = azimuth % 360
        if elevation is not None:
            self.camera_elevation = max(-10.0, min(85.0, elevation))
        if distance is not None:
            self.camera_distance = max(0.5, min(5.0, distance))

    def get_screenshot(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render scene from camera defined by spherical coords and return RGB array."""
        az = math.radians(self.camera_azimuth)
        el = math.radians(self.camera_elevation)
        d = self.camera_distance
        tx, ty, tz = self.camera_target
        eye = [
            tx + d * math.cos(el) * math.cos(az),
            ty + d * math.cos(el) * math.sin(az),
            tz + d * math.sin(el),
        ]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 0, 1],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=0.1, farVal=10.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_array = np.array(rgb, dtype=np.uint8)
        if rgb_array.ndim == 1:  # Linux pip build returns flat buffer
            rgb_array = rgb_array.reshape(height, width, 4)
        return rgb_array[:, :, :3]  # drop alpha

    def stop(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
