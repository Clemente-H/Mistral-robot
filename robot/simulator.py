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
        # Blue box on the table area
        box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                                      rgbaColor=[0.2, 0.4, 1.0, 1.0])
        box_id = p.createMultiBody(baseMass=0.5,
                                   baseCollisionShapeIndex=box_col,
                                   baseVisualShapeIndex=box_vis,
                                   basePosition=[0.5, 0.1, 0.05])
        self.objects["blue_box"] = box_id

        # Red sphere (target / table marker)
        sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                         rgbaColor=[1.0, 0.2, 0.2, 1.0])
        sphere_id = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=sphere_col,
                                      baseVisualShapeIndex=sphere_vis,
                                      basePosition=[0.6, -0.2, 0.04])
        self.objects["target"] = sphere_id

    # ------------------------------------------------------------------
    # Actions exposed as agent tools
    # ------------------------------------------------------------------

    def move_to(self, x: float, y: float, z: float):
        """Move end-effector to (x, y, z) in world coordinates."""
        end_effector_idx = self.num_joints - 1
        joint_poses = p.calculateInverseKinematics(
            self.kuka_id, end_effector_idx, [x, y, z]
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
        """Simulate gripper close (visual only — KUKA has no gripper in this URDF)."""
        self._step(30)

    def release(self):
        """Simulate gripper open."""
        self._step(30)

    def reset(self):
        """Return arm to home position."""
        self._reset_arm()
        self._step(60)

    def wave(self):
        """Wave hello — raise arm up and oscillate the wrist joint."""
        # Raise arm to wave position
        wave_pose = [0, -math.pi / 6, 0, math.pi / 4, 0, -math.pi / 3, 0]
        for i, angle in enumerate(wave_pose[:self.num_joints]):
            p.setJointMotorControl2(
                self.kuka_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=500,
            )
        self._step(80)

        # Oscillate last joint (wrist) back and forth 3 times
        wrist_joint = self.num_joints - 1
        for _ in range(3):
            for target in [math.pi / 4, -math.pi / 4]:
                p.setJointMotorControl2(
                    self.kuka_id, wrist_joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=300,
                )
                self._step(40)

        self._reset_arm()
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
        return np.array(rgb, dtype=np.uint8)[:, :, :3]  # drop alpha

    def stop(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
