import * as THREE from 'three';
import { FilesetResolver, FaceLandmarker, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

const video = document.getElementById('webcam');
const canvas = document.getElementById('output');
const loading = document.getElementById('loading');

// 1. Initialise Three.js Scene, Camera & Advanced Rendering
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 7.5);

// Standard lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(5, 10, 7.5);
scene.add(dirLight);

// 2. Build Procedural Skeletal "Smiley with Body" Avatar
const avatar = new THREE.Group();
avatar.position.y = -1.2;
scene.add(avatar);

// Torso (The capsule body)
const bodyGeom = new THREE.CapsuleGeometry(0.85, 1.6, 8, 16);
const bodyMat = new THREE.MeshStandardMaterial({ color: 0x3d7beb, roughness: 0.4 }); // Playful blue torso
const torso = new THREE.Mesh(bodyGeom, bodyMat);
torso.position.y = 0.8;
avatar.add(torso);

// The Smiley Head (The Sphere)
const headGeom = new THREE.SphereGeometry(0.85, 32, 32);
const headMat = new THREE.MeshStandardMaterial({ color: 0xffd32a, roughness: 0.3 }); // Vibrant yellow smiley
const head = new THREE.Mesh(headGeom, headMat);
head.position.set(0, 1.9, 0);
avatar.add(head);

// Custom Face features
const eyeGeom = new THREE.SphereGeometry(0.09, 16, 16);
const eyeMat = new THREE.MeshStandardMaterial({ color: 0x1e272e, roughness: 0.5 }); // Dark eyes
const leftEye = new THREE.Mesh(eyeGeom, eyeMat);
leftEye.position.set(-0.28, 0.12, 0.72);
head.add(leftEye);

const rightEye = new THREE.Mesh(eyeGeom, eyeMat);
rightEye.position.set(0.28, 0.12, 0.72);
head.add(rightEye);

// Elastic mouth
const mouthGeom = new THREE.TorusGeometry(0.2, 0.05, 8, 24, Math.PI);
const mouthMat = new THREE.MeshStandardMaterial({ color: 0xb5342a, roughness: 0.6 });
const mouth = new THREE.Mesh(mouthGeom, mouthMat);
mouth.position.set(0, -0.15, 0.73);
mouth.rotation.x = Math.PI; // Face downward/happy curve
head.add(mouth);

// Articulated Arms, Hands & Bones
function createSkeletalArm(isLeft) {
    const arm = new THREE.Group();
    const shoulderOffset = isLeft ? -0.95 : 0.95;
    arm.position.set(shoulderOffset, 1.35, 0);

    // Shoulder & Joints
    const upperArmGeom = new THREE.CylinderGeometry(0.12, 0.1, 1.0, 8);
    const armMat = new THREE.MeshStandardMaterial({ color: 0xffd32a, roughness: 0.5 });
    const upperArm = new THREE.Mesh(upperArmGeom, armMat);
    upperArm.position.y = -0.5;
    arm.add(upperArm);

    const forearm = new THREE.Group();
    forearm.position.set(0, -1.0, 0);
    const forearmGeom = new THREE.CylinderGeometry(0.1, 0.08, 0.9, 8);
    const forearmMesh = new THREE.Mesh(forearmGeom, armMat);
    forearmMesh.position.y = -0.45;
    forearm.add(forearmMesh);
    upperArm.add(forearm);

    // 21 skeletal joints for wrist and finger endpoints (Dynamic Spheres)
    const fingersGroup = new THREE.Group();
    fingersGroup.position.set(0, -0.9, 0);
    forearm.add(fingersGroup);

    const skeletalJoints = [];
    for (let i = 0; i < 21; i++) {
        const jointGeom = new THREE.SphereGeometry(0.045, 8, 8);
        const jointMat = new THREE.MeshBasicMaterial({ color: 0xffd32a });
        const joint = new THREE.Mesh(jointGeom, jointMat);
        joint.visible = false; // Visibility is turned on whenever tracker activates
        fingersGroup.add(joint);
        skeletalJoints.push(joint);
    }

    // Connect fingers via visual structural rods
    const linkMaterial = new THREE.LineBasicMaterial({ color: 0xffd32a, linewidth: 2 });
    const lineIndices = [
        [0,1],[1,2],[2,3],[3,4],        // Thumb
        [0,5],[5,6],[6,7],[7,8],        // Index
        [5,9],[9,10],[10,11],[11,12],   // Middle
        [9,13],[13,14],[14,15],[15,16], // Ring
        [13,17],[17,18],[18,19],[19,20] // Pinky
    ];
    
    const lines = [];
    lineIndices.forEach(() => {
        const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(lineGeo, linkMaterial);
        line.visible = false;
        fingersGroup.add(line);
        lines.push({ line, indices: lineIndices[lines.length] });
    });

    return { root: arm, upperArm, forearm, fingersGroup, joints: skeletalJoints, lines };
}

const leftArm = createSkeletalArm(true);
const rightArm = createSkeletalArm(false);
avatar.add(leftArm.root);
avatar.add(rightArm.root);

// 3. MediaPipe Core & Parallel Model Async Loader
let faceLandmarker, handLandmarker;
async function setupAI() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    
    // Face tracking parameters
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", delegate: "GPU" },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO"
    });

    // Parallel Hand & Fine Finger tracking
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2
    });
}

async function setupWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    return new Promise((resolve) => video.onloadedmetadata = () => resolve(video));
}

// Data Smoothing Variable initialization (Linear interpolation values)
const lerpSpeed = 0.18;
let smoothedHeadRot = new THREE.Euler();
let smoothedHeadPos = new THREE.Vector3();

// 4. Transform Landmarks into 3D Joints mapping
function mapHandToSkeleton(handData, armSkeleton, isLeftHand) {
    armSkeleton.joints.forEach(j => j.visible = true);
    armSkeleton.lines.forEach(l => l.line.visible = true);

    const landmarks = handData.landmarks;

    // Direct geometric point-to-point map to position wrist and hand
    const wristMP = landmarks[0];
    const targetWristX = -(wristMP.x - 0.5) * (isLeftHand ? 2.5 : 2.5) + (isLeftHand ? -0.8 : 0.8);
    const targetWristY = -(wristMP.y - 0.5) * 3.5 + 0.8;
    const targetWristZ = -wristMP.z * 2.5;

    // Rotate shoulder and arm joints toward mapped hand coordinates
    const targetArmVec = new THREE.Vector3(targetWristX, targetWristY, targetWristZ);
    armSkeleton.root.lookAt(targetArmVec.clone().multiplyScalar(1.5));
    
    // Smooth the positions
    armSkeleton.fingersGroup.position.x += (targetWristX - armSkeleton.fingersGroup.position.x) * lerpSpeed;
    armSkeleton.fingersGroup.position.y += (targetWristY - armSkeleton.fingersGroup.position.y) * lerpSpeed;
    armSkeleton.fingersGroup.position.z += (targetWristZ - armSkeleton.fingersGroup.position.z) * lerpSpeed;

    // Apply exact finger movement offsets locally
    landmarks.forEach((landmark, index) => {
        const joint = armSkeleton.joints[index];
        const jX = -(landmark.x - wristMP.x) * 2.5;
        const jY = -(landmark.y - wristMP.y) * 2.5;
        const jZ = -(landmark.z - wristMP.z) * 2.5;
        
        joint.position.set(jX, jY, jZ);
    });

    // Re-draw connection skeleton lines
    armSkeleton.lines.forEach(({ line, indices }) => {
        const jStart = armSkeleton.joints[indices[0]].position;
        const jEnd = armSkeleton.joints[indices[1]].position;
        
        const positions = new Float32Array([jStart.x, jStart.y, jStart.z, jEnd.x, jEnd.y, jEnd.z]);
        line.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        line.geometry.computeBoundingBox();
        line.geometry.computeBoundingSphere();
    });
}

// 5. Unified Animation Rendering (Run frame tasks 60 times a second)
let lastVideoTime = -1;
function updateRenderLoop() {
    requestAnimationFrame(updateRenderLoop);

    if (video.currentTime !== lastVideoTime && faceLandmarker && handLandmarker) {
        lastVideoTime = video.currentTime;
        const now = performance.now();

        // Pass webcam frames through double tracking layers
        const faceResults = faceLandmarker.detectForVideo(video, now);
        const handResults = handLandmarker.detectForVideo(video, now);

        // a. Interpret Head and Expressions Tracking
        if (faceResults.facialTransformationMatrixes && faceResults.facialTransformationMatrixes.length > 0) {
            const matrixData = faceResults.facialTransformationMatrixes[0].data;
            const matrix = new THREE.Matrix4().fromArray(matrixData);
            
            const targetPos = new THREE.Vector3().setFromMatrixPosition(matrix);
            const targetRot = new THREE.Euler().setFromRotationMatrix(matrix);

            // Interpolated coordinate transition (Lerp) stops jittering
            smoothedHeadRot.x += (targetRot.x - smoothedHeadRot.x) * lerpSpeed;
            smoothedHeadRot.y += (targetRot.y - smoothedHeadRot.y) * lerpSpeed;
            smoothedHeadRot.z += (targetRot.z - smoothedHeadRot.z) * lerpSpeed;
            head.rotation.copy(smoothedHeadRot);

            smoothedHeadPos.x += (targetPos.x - smoothedHeadPos.x) * lerpSpeed;
            smoothedHeadPos.y += (targetPos.y - smoothedHeadPos.y) * lerpSpeed;
            head.position.x = smoothedHeadPos.x * 0.05;
            head.position.y = 1.9 + (smoothedHeadPos.y * 0.05);

            // Expressions (Eye Blink, Jaw/Mouth Open/Smile) mapping directly
            if (faceResults.faceBlendshapes && faceResults.faceBlendshapes.length > 0) {
                const shapes = faceResults.faceBlendshapes[0].categories;
                
                const leftBlink = shapes.find(s => s.categoryName === 'eyeBlinkLeft')?.score || 0;
                const rightBlink = shapes.find(s => s.categoryName === 'eyeBlinkRight')?.score || 0;
                const jawOpen = shapes.find(s => s.categoryName === 'jawOpen')?.score || 0;
                const smileL = shapes.find(s => s.categoryName === 'mouthSmileLeft')?.score || 0;
                const smileR = shapes.find(s => s.categoryName === 'mouthSmileRight')?.score || 0;

                // Adjust Y scales of parts dynamically
                leftEye.scale.y = Math.max(0.08, 1 - (leftBlink * 1.5));
                rightEye.scale.y = Math.max(0.08, 1 - (rightBlink * 1.5));
                
                // Open and reshape mouth visual based on tracking
                mouth.scale.set(1 + (smileL + smileR) * 0.5, 1 + jawOpen * 2.2, 1);
            }
        }

        // b. Handle dual hands and fingers joint assignment mapping
        leftArm.joints.forEach(j => j.visible = false);
        leftArm.lines.forEach(l => l.line.visible = false);
        rightArm.joints.forEach(j => j.visible = false);
        rightArm.lines.forEach(l => l.line.visible = false);

        if (handResults.handedness && handResults.handedness.length > 0) {
            handResults.handedness.forEach((handness, i) => {
                const isLeft = handness[0].label === "Right"; // OpenCV/Mediapipe flips handedness
                const mappedArm = isLeft ? leftArm : rightArm;
                
                mapHandToSkeleton({
                    landmarks: handResults.landmarks[i]
                }, mappedArm, isLeft);
            });
        }
    }

    renderer.render(scene, camera);
}

// Support resizing gracefully
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// 6. Launch Application Flow
async function init() {
    await setupAI();
    await setupWebcam();
    video.play();
    loading.style.display = 'none';
    updateRenderLoop();
}

init();
