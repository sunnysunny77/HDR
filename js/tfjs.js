import * as tf from "@tensorflow/tfjs";

const labels = [
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
  "U", "V", "W", "X", "Y", "Z",
  "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"
];

let model;
let drawing = false;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const CANVAS_SIZE = 280;
canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;

ctx.fillStyle = "black";
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictionDiv = document.getElementById("prediction");

export const tfjs = async () => {
  predictionDiv.innerText = "Loading model...";
  try {
    await tf.setBackend("webgl");
    await tf.ready();
  } catch {
    await tf.setBackend("cpu");
    await tf.ready();
  }
  try {
    model = await tf.loadGraphModel("tfjs_model/model.json");
    predictionDiv.innerText = "Model ready. Draw and click Predict.";
    console.log("Model loaded.");
  } catch (error) {
    predictionDiv.innerText = "Failed to load model.";
    console.error("Model loading error:", error);
  }
};

predictBtn.addEventListener("click", async () => {
  if (!model) {
    alert("Model not loaded yet.");
    return;
  }

  predictBtn.disabled = true;
  predictionDiv.innerText = "Predicting...";
  await tf.nextFrame();

  try {
    const img = tf.browser.fromPixels(canvas).toFloat().div(255);
    const brightness = img.mean(2);
    const mask = brightness.greater(0.1);
    const coords = await tf.whereAsync(mask);

    if (coords.shape[0] === 0) throw new Error("Empty canvas");

    const coordsData = await coords.array();
    const ys = coordsData.map(([y]) => y);
    const xs = coordsData.map(([, x]) => x);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const minX = Math.min(...xs), maxX = Math.max(...xs);

    const cropped = img.slice([minY, minX, 0], [maxY - minY + 1, maxX - minX + 1, 3]);

    const maxSide = Math.max(cropped.shape[0], cropped.shape[1]);
    const scale = 20 / maxSide;
    const newH = Math.round(cropped.shape[0] * scale);
    const newW = Math.round(cropped.shape[1] * scale);
    const resized = tf.image.resizeBilinear(cropped, [newH, newW]);

    const topPad = Math.floor((28 - newH) / 2);
    const leftPad = Math.floor((28 - newW) / 2);
    const padded = tf.pad(resized, [[topPad, 28 - newH - topPad], [leftPad, 28 - newW - leftPad], [0, 0]]);

    const gray = padded.mean(2).expandDims(0).expandDims(3);

    const prediction = model.predict(gray);
    const maxIndex = prediction.argMax(-1).dataSync()[0];
    const maxVal = prediction.max(-1).dataSync()[0];

    predictionDiv.innerText = `Prediction: ${labels[maxIndex]} (Confidence: ${maxVal.toFixed(4)})`;

    img.dispose();
    brightness.dispose();
    mask.dispose();
    coords.dispose();
    cropped.dispose();
    resized.dispose();
    padded.dispose();
    gray.dispose();
    prediction.dispose();

  } catch (error) {
    predictionDiv.innerText = "Prediction failed.";
    if (error.message !== "Empty canvas") console.error(error);
  } finally {
    predictBtn.disabled = false;
  }
});

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  return { x, y };
}

canvas.addEventListener("pointerdown", e => {
  if (["mouse", "pen", "touch"].includes(e.pointerType)) {
    drawing = true;
    const { x, y } = getCanvasCoords(e);
    ctx.strokeStyle = "white";
    ctx.lineWidth = Math.max(10, Math.min(canvas.width, canvas.height) / 20);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(x, y);
    e.preventDefault();
  }
});

canvas.addEventListener("pointermove", e => {
  if (drawing) {
    const { x, y } = getCanvasCoords(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    e.preventDefault();
  }
});

["pointerup", "pointercancel", "pointerleave"].forEach(evt => {
  canvas.addEventListener(evt, () => {
    drawing = false;
  });
});

clearBtn.addEventListener("click", () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  predictionDiv.innerHTML = `
    <b>Predicted:</b> ? <br/>
    <b>Confidence:</b> ?
  `;
});
