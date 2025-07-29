import * as tf from "@tensorflow/tfjs";

let model;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictionDiv = document.getElementById("prediction");

function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  canvas.style.width = `${canvas.clientWidth}px`;
  canvas.style.height = `${canvas.clientHeight}px`;
  ctx.scale(dpr, dpr);

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  canvas.style.touchAction = "none";
}

setupCanvas();

let drawing = false;

canvas.addEventListener("pointerdown", e => {
  if (["mouse", "pen", "touch"].includes(e.pointerType)) {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    e.preventDefault();
  }
});

canvas.addEventListener("pointermove", e => {
  if (drawing) {
    ctx.strokeStyle = "white";
    ctx.lineWidth = Math.max(10, canvas.clientWidth / 30);
    ctx.lineCap = "round";
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    e.preventDefault();
  }
});

["pointerup", "pointercancel", "pointerleave"].forEach(evt => {
  canvas.addEventListener(evt, e => {
    drawing = false;
    e.preventDefault();
  });
});

clearBtn.addEventListener("click", () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  predictionDiv.innerText = "Prediction: ?";
});

predictBtn.addEventListener("click", predict);

function preprocessCanvas() {
  return tf.browser.fromPixels(canvas, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(255.0)
    .expandDims(0); // shape: [1, 28, 28, 1]
}

async function predict() {
  if (!model) {
    alert("Model not loaded yet.");
    return;
  }

  predictBtn.disabled = true;
  predictionDiv.innerText = "Predicting...";
  await tf.nextFrame();

  const { prediction, confidence } = tf.tidy(() => {
    const input = preprocessCanvas();
    const output = model.predict(input).softmax();
    const prediction = output.argMax(-1).dataSync()[0];
    const confidence = output.max().dataSync()[0];
    return { prediction, confidence };
  });

  const labelMap = "0123456789";  // only digits 0-9
  const label = labelMap[prediction] ?? "?";
  predictionDiv.innerText = `Prediction: ${label} (${(confidence * 100).toFixed(2)}%)`;

  predictBtn.disabled = false;
}

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
    model = await tf.loadGraphModel("tfjs_model/model.json"); // simplified model path
    predictionDiv.innerText = "Model ready. Draw and click Predict.";
    console.log("Model loaded.");
  } catch (error) {
    predictionDiv.innerText = "Failed to load model.";
    console.error("Model loading error:", error);
  }
};
