import * as THREE from 'three';
import { FilesetResolver, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

// 1. Target the HTML elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('output');
const loading = document.getElementById('loading');

// 2. Set up the Three.js 3D Scene
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

// Add basic lighting so we can see the object
const light = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(light);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(0, 1, 1);
scene.add(directionalLight);

// Create a 3D box to act as our "head"
// We paint the front face red so you can see which way it is looking
const geometry = new THREE.BoxGeometry(1.5, 2, 1.5);
const materials = [
    new THREE.MeshLambertMaterial({ color: 0xcccccc }), // right
    new THREE.MeshLambertMaterial({ color: 0xcccccc }), // left
    new THREE.MeshLambertMaterial({ color: 0xcccccc }), // top
    new THREE.MeshLambertMaterial({ color: 0xcccccc }), // bottom
    new THREE.MeshLambertMaterial({ color: 0xff0000 }), // front (red face)
    new THREE.MeshLambertMaterial({ color: 0xcccccc }), // back
];
const headMesh = new THREE.Mesh(geometry, materials);
scene.add(headMesh);

// 3. Set up the MediaPipe AI Face Tracker
let faceLandmarker;
async function setupMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU" // Uses your local graphics card
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO",
        numFaces: 1
    });
}

// 4. Request Webcam Access
async function setupWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => resolve(video);
    });
}

// 5. The Render Loop (Runs 60 times a second)
let lastVideoTime = -1;
function renderLoop() {
    requestAnimationFrame(renderLoop);

    // If the video has a new frame, process it
    if (video.currentTime !== lastVideoTime && faceLandmarker) {
        lastVideoTime = video.currentTime;
        
        // Analyze the frame with AI
        const results = faceLandmarker.detectForVideo(video, performance.now());
        
        // If a face is found, apply the rotation to our 3D box
        if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
            const matrixData = results.facialTransformationMatrixes[0].data;
            const headMatrix = new THREE.Matrix4().fromArray(matrixData);
            
            // Extract just the rotation from the AI matrix and apply it to the Three.js mesh
            headMesh.rotation.setFromRotationMatrix(headMatrix);
        }
    }
    
    // Draw the 3D scene to the screen
    renderer.render(scene, camera);
}

// Keep the canvas sized correctly if the user resizes their browser window
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// 6. Start Everything
async function init() {
    await setupMediaPipe();
    await setupWebcam();
    video.play();
    loading.style.display = 'none'; // Hide the loading text once ready
    renderLoop();
}

init();
