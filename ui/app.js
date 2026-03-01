// ============================================================
//  Three.js imports (CDN — resolved via importmap in index.html)
// ============================================================
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================
//  Renderer + Scene setup
// ============================================================
const viewer = document.getElementById('viewer');

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
viewer.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b10);
scene.fog = new THREE.FogExp2(0x070b10, 0.04);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 30);
camera.position.set(1.8, 1.4, 1.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.4, 0.2, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 0.4;
controls.maxDistance = 7;
controls.maxPolarAngle = Math.PI * 0.9;
camera.lookAt(controls.target);

// ── Lighting ──────────────────────────────────────────────
const ambient = new THREE.AmbientLight(0xffffff, 0.25);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xfff5e0, 2.5);
keyLight.position.set(3, 4, 2);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 15;
keyLight.shadow.camera.left = -3;
keyLight.shadow.camera.right = 3;
keyLight.shadow.camera.top = 3;
keyLight.shadow.camera.bottom = -3;
keyLight.shadow.bias = -0.0005;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x4488ff, 0.6);
fillLight.position.set(-3, 1, -2);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xff8844, 0.4);
rimLight.position.set(0, 0.5, -4);
scene.add(rimLight);

// ── Floor ─────────────────────────────────────────────────
const gridHelper = new THREE.GridHelper(5, 30, 0x1a2030, 0x141c28);
scene.add(gridHelper);

const floorGeo = new THREE.PlaneGeometry(5, 5);
const floorMat = new THREE.MeshStandardMaterial({
  color: 0x080c14, roughness: 0.9, metalness: 0.05,
});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

// ── Materials ──────────────────────────────────────────────
const MAT = {
  link:    new THREE.MeshStandardMaterial({ color: 0x6b7280, metalness: 0.75, roughness: 0.25 }),
  joint:   new THREE.MeshStandardMaterial({ color: 0xf97316, metalness: 0.65, roughness: 0.25,
            emissive: 0x1a0800, emissiveIntensity: 0.3 }),
  ee:      new THREE.MeshStandardMaterial({ color: 0x22c55e, metalness: 0.7, roughness: 0.2,
            emissive: 0x003300, emissiveIntensity: 0.4 }),
  box:     new THREE.MeshStandardMaterial({ color: 0x2563eb, metalness: 0.25, roughness: 0.55,
            emissive: 0x000a1a, emissiveIntensity: 0.2 }),
  sphere:  new THREE.MeshStandardMaterial({ color: 0xdc2626, metalness: 0.3, roughness: 0.45,
            emissive: 0x1a0000, emissiveIntensity: 0.25 }),
  yellow:  new THREE.MeshStandardMaterial({ color: 0xeab308, metalness: 0.2, roughness: 0.5,
            emissive: 0x1a1000, emissiveIntensity: 0.2 }),
  green2:  new THREE.MeshStandardMaterial({ color: 0x16a34a, metalness: 0.2, roughness: 0.5,
            emissive: 0x001a00, emissiveIntensity: 0.2 }),
};

// ── Scene objects ──────────────────────────────────────────
const blueBox = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.1), MAT.box);
blueBox.castShadow = true; blueBox.receiveShadow = true;
scene.add(blueBox);

const redSphere = new THREE.Mesh(new THREE.SphereGeometry(0.04, 24, 24), MAT.sphere);
redSphere.castShadow = true;
scene.add(redSphere);

const yellowCylinder = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.1, 16), MAT.yellow);
yellowCylinder.castShadow = true; yellowCylinder.receiveShadow = true;
scene.add(yellowCylinder);

const greenCube = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.08), MAT.green2);
greenCube.castShadow = true; greenCube.receiveShadow = true;
scene.add(greenCube);

// ── Robot arm group (rebuilt each frame) ──────────────────
const robotGroup = new THREE.Group();
scene.add(robotGroup);

const RADII = [0.065, 0.058, 0.052, 0.046, 0.040, 0.034, 0.026, 0.020];

// PyBullet Z-up → Three.js Y-up
function pb(pos) {
  return new THREE.Vector3(pos[0], pos[2], -pos[1]);
}

function makeCylinder(A, B, r0, r1, mat) {
  const dir = new THREE.Vector3().subVectors(B, A);
  const len = dir.length();
  if (len < 1e-4) return null;
  const geo = new THREE.CylinderGeometry(r1, r0, len, 14, 1, false);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.position.copy(A).addScaledVector(dir.normalize(), len / 2);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
  return mesh;
}

function makeJoint(pos, r, mat) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 14), mat);
  mesh.castShadow = true;
  mesh.position.copy(pos);
  return mesh;
}

function updateRobot(frameData) {
  while (robotGroup.children.length) robotGroup.remove(robotGroup.children[0]);

  const pts = frameData.links.map(pb);

  const baseMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.11, 0.13, 0.06, 18),
    MAT.link
  );
  baseMesh.position.copy(pts[0]).add(new THREE.Vector3(0, 0.03, 0));
  baseMesh.castShadow = true;
  robotGroup.add(baseMesh);

  for (let i = 0; i < pts.length - 1; i++) {
    const cyl = makeCylinder(pts[i], pts[i + 1], RADII[i], RADII[i + 1], MAT.link);
    if (cyl) robotGroup.add(cyl);
    const jMat = i === 0 ? MAT.link : MAT.joint;
    robotGroup.add(makeJoint(pts[i], RADII[i] * 1.25, jMat));
  }

  const eePos = pts[pts.length - 1];
  const eeCone = new THREE.Mesh(new THREE.ConeGeometry(0.022, 0.07, 10), MAT.ee);
  eeCone.position.copy(eePos);
  eeCone.castShadow = true;
  robotGroup.add(eeCone);

  if (frameData.objects) {
    if (frameData.objects.blue_box)
      blueBox.position.copy(pb(frameData.objects.blue_box));
    if (frameData.objects.red_sphere)
      redSphere.position.copy(pb(frameData.objects.red_sphere));
    if (frameData.objects.yellow_cylinder)
      yellowCylinder.position.copy(pb(frameData.objects.yellow_cylinder));
    if (frameData.objects.green_cube)
      greenCube.position.copy(pb(frameData.objects.green_cube));
  }
}

// ============================================================
//  Animation playback
// ============================================================
let _animFrames = [];
let _animIdx    = 0;
let _animTimer  = null;

function playAnimation(frames, fps = 40) {
  if (_animTimer) clearInterval(_animTimer);
  _animFrames = frames;
  _animIdx    = 0;
  _animTimer  = setInterval(() => {
    if (_animIdx >= _animFrames.length) {
      clearInterval(_animTimer);
      setStatus('Ready');
      return;
    }
    updateRobot(_animFrames[_animIdx++]);
  }, 1000 / fps);
}

// ============================================================
//  Resize + render loop
// ============================================================
function resize() {
  const w = viewer.clientWidth;
  const h = viewer.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

(function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
})();

// ============================================================
//  Load initial pose
// ============================================================
fetch('/api/scene')
  .then(r => r.json())
  .then(data => updateRobot(data))
  .catch(console.error);

// ============================================================
//  Status helpers
// ============================================================
const dot      = document.getElementById('status-dot');
const label    = document.getElementById('status-label');
const statusbar = document.getElementById('statusbar');

function setStatus(text, busy = false) {
  label.textContent = text;
  statusbar.textContent = text;
  dot.className = busy ? 'busy' : '';
}

// ============================================================
//  Chat
// ============================================================
const chatEl   = document.getElementById('chat-messages');
const msgInput = document.getElementById('msg-input');
const sendBtn  = document.getElementById('send-btn');

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

async function sendCommand() {
  const text = msgInput.value.trim();
  if (!text || sendBtn.disabled) return;

  msgInput.value = '';
  sendBtn.disabled = true;
  setStatus('Thinking…', true);

  addMsg('user', text);
  const thinking = addMsg('thinking', '● Mistral is planning…');

  try {
    const res = await fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    chatEl.removeChild(thinking);
    addMsg('bot', data.response || '(no text response)');

    if (data.frames && data.frames.length > 0) {
      setStatus(`Animating ${data.frames.length} frames…`, true);
      playAnimation(data.frames);
    } else {
      setStatus('Ready');
    }
  } catch (err) {
    chatEl.removeChild(thinking);
    addMsg('sys', `Error: ${err.message}`);
    setStatus('Error');
  }

  sendBtn.disabled = false;
  msgInput.focus();
}

sendBtn.addEventListener('click', sendCommand);
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendCommand(); }
});

// ============================================================
//  Macros panel
// ============================================================
const macrosToggle = document.getElementById('macros-toggle');
const macrosPanel  = document.getElementById('macros-panel');
const macrosArrow  = document.getElementById('macros-arrow');

macrosToggle.addEventListener('click', () => {
  const open = macrosPanel.style.display === 'block';
  macrosPanel.style.display = open ? 'none' : 'block';
  macrosArrow.textContent = open ? '▼' : '▲';
  if (!open) refreshMacros();
});

async function refreshMacros() {
  const res   = await fetch('/api/macros');
  const data  = await res.json();
  const list  = document.getElementById('macros-list');
  list.innerHTML = '';
  const entries = Object.entries(data);
  if (entries.length === 0) {
    list.innerHTML = '<div style="color:var(--muted);font-size:0.78rem">No macros yet.</div>';
    return;
  }
  for (const [name, info] of entries) {
    const item = document.createElement('div');
    item.className = 'macro-item';
    item.innerHTML = `
      <div>
        <strong style="color:var(--accent)">${name}</strong><br/>
        <code>${info.raw || ''}</code>
      </div>
      <button class="macro-del" data-name="${name}" title="Delete">×</button>
    `;
    list.appendChild(item);
  }
  list.querySelectorAll('.macro-del').forEach(btn => {
    btn.addEventListener('click', async () => {
      await fetch(`/api/macros/${btn.dataset.name}`, { method: 'DELETE' });
      refreshMacros();
    });
  });
}

window.saveMacro = async function () {
  const name  = document.getElementById('macro-name').value.trim();
  const steps = document.getElementById('macro-steps').value.trim();
  if (!name || !steps) return;
  await fetch('/api/macros', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, steps }),
  });
  document.getElementById('macro-name').value  = '';
  document.getElementById('macro-steps').value = '';
  refreshMacros();
};

// ============================================================
//  Voice recording (mic → Voxtral STT → command)
// ============================================================
const micBtn = document.getElementById('mic-btn');
let micActive = false;
let micStream = null;
let audioCtx  = null;
let processor = null;
let micChunks = [];

micBtn.addEventListener('click', async () => {
  if (!micActive) { await startMic(); } else { await stopMic(); }
});

async function startMic() {
  if (!navigator.mediaDevices?.getUserMedia) {
    addMsg('sys', 'Microphone not supported in this browser.');
    return;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(micStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    micChunks = [];

    processor.onaudioprocess = (e) => {
      micChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    src.connect(processor);
    processor.connect(audioCtx.destination);

    micActive = true;
    micBtn.classList.add('recording');
    micBtn.title = 'Click to stop recording';
    setStatus('Recording… click mic to stop', true);
  } catch (err) {
    addMsg('sys', `Mic error: ${err.message}`);
  }
}

async function stopMic() {
  micActive = false;
  micBtn.classList.remove('recording');
  micBtn.disabled = true;
  micBtn.title = 'Click to record voice command';

  processor.disconnect();
  micStream.getTracks().forEach(t => t.stop());
  const sourceSampleRate = audioCtx.sampleRate;
  await audioCtx.close();

  if (micChunks.length === 0) {
    micBtn.disabled = false;
    setStatus('Ready');
    return;
  }

  // Merge float32 chunks
  const totalLen = micChunks.reduce((s, c) => s + c.length, 0);
  const merged   = new Float32Array(totalLen);
  let offset = 0;
  for (const chunk of micChunks) { merged.set(chunk, offset); offset += chunk.length; }

  // Float32 → Int16
  const pcm16 = new Int16Array(merged.length);
  for (let i = 0; i < merged.length; i++) {
    const s = Math.max(-1, Math.min(1, merged[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  setStatus('Transcribing…', true);
  addMsg('sys', 'Transcribing voice…');

  try {
    const res  = await fetch(`/api/voice?sample_rate=${Math.round(sourceSampleRate)}`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body:    pcm16.buffer,
    });
    const data = await res.json();
    if (data.text && data.text.trim()) {
      msgInput.value = data.text.trim();
      addMsg('sys', `Voice: "${data.text.trim()}"`);
      sendCommand();
    } else {
      addMsg('sys', 'Could not understand audio — try again.');
      setStatus('Ready');
    }
  } catch (err) {
    addMsg('sys', `Voice error: ${err.message}`);
    setStatus('Ready');
  } finally {
    micBtn.disabled = false;
  }
}
