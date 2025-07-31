import * as tf from "@tensorflow/tfjs";

let model;
let drawing = false;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictionDiv = document.getElementById("prediction");
const container = document.getElementById("container");

predictBtn.addEventListener("click", async () => {
  if (!model) {
    alert("Model not loaded yet.");
    return;
  }

  predictBtn.disabled = true;
  predictionDiv.innerText = "Predicting...";
  await tf.nextFrame();

  let processed;
  try {
    processed = tf.tidy(() => {
      const imgTensor = tf.browser.fromPixels(canvas);
      const gray = imgTensor.mean(2);
      const resized = tf.image.resizeBilinear(gray.expandDims(-1), [28, 28], true);
      const normalized = resized.div(255);
      return normalized.expandDims(0);
    });

    const output = model.predict(processed);
    const data = await output.data();

    const maxIndex = data.indexOf(Math.max(...data));
    const maxVal = data[maxIndex];

    predictionDiv.innerText = `Prediction: ${maxIndex} (Confidence: ${maxVal.toFixed(4)})`;

    output.dispose();
  } catch (err) {
    predictionDiv.innerText = "Prediction failed.";
    console.error(err);
  } finally {
    processed?.dispose();
    predictBtn.disabled = false;
  }
});


//assume is called loads
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
    ctx.lineWidth = Math.max(10, Math.min(canvas.width, canvas.height) / 20);
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

const resizeCanvas = () => {

  const rect = container.getBoundingClientRect();

  const size = Math.min(rect.width, 280);

  canvas.width = size;
  canvas.height = size;

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};

resizeCanvas();

window.addEventListener("resize", () => {

  resizeCanvas();
  predictionDiv.innerText = "Prediction: ?";
});
