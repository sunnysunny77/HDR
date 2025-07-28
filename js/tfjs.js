import * as tf from '@tensorflow/tfjs';

let model;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictionDiv = document.getElementById('prediction');

function setupCanvas() {
  // Handle high DPI screens
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);

  // Fill background black
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  // Prevent touch scrolling on the canvas
  canvas.style.touchAction = 'none';
}

setupCanvas();

let drawing = false;

canvas.addEventListener('pointerdown', e => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
  e.preventDefault();
});

canvas.addEventListener('pointermove', e => {
  if (drawing) {
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    e.preventDefault();
  }
});

canvas.addEventListener('pointerup', e => {
  drawing = false;
  e.preventDefault();
});

canvas.addEventListener('pointercancel', e => {
  drawing = false;
  e.preventDefault();
});

canvas.addEventListener('pointerleave', e => {
  drawing = false;
  e.preventDefault();
});

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  predictionDiv.innerText = 'Prediction: ?';
});

predictBtn.addEventListener('click', predict);

async function loadModel() {
  await tf.setBackend('cpu');  // Use CPU backend
  await tf.ready();
  model = await tf.loadGraphModel('tfjs_model/model.json');
  console.log('Model loaded.');
}

function preprocessCanvas() {
  // Use canvas.clientWidth/Height for accurate resize on high DPI
  return tf.browser.fromPixels(canvas, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(255.0)
    .expandDims(0); // shape: [1, 28, 28, 1]
}

async function predict() {
  if (!model) {
    alert('Model not loaded yet.');
    return;
  }

  const input = preprocessCanvas();
  const output = model.predict(input);
  const result = await output.argMax(-1).data();
  predictionDiv.innerText = `Prediction: ${result[0]}`;
}

export const tfjs = () => {

  loadModel();
}