import * as tf from "@tensorflow/tfjs";

// keras.datasets.mnist ai does all preprocessing look good, and drawing?

let model;
let drawing = false;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const CANVAS_SIZE = 280;
canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;
ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";
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
      const size = canvas.width;
      const imageData = ctx.getImageData(0, 0, size, size);
      const imgTensor = tf.browser.fromPixels(imageData);
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
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionDiv.innerText = "Prediction: ?";
});

const resizeCanvas = () => {
  const rect = container.getBoundingClientRect();
  const newSize = Math.min(rect.width, 280);

  // Backup current image
  const oldImage = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Create a temporary canvas to scale the old image
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.putImageData(oldImage, 0, 0);

  // Resize actual canvas
  canvas.width = newSize;
  canvas.height = newSize;

  // Fill black background
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw old image, scaled to new canvas size
  ctx.drawImage(tempCanvas, 0, 0, newSize, newSize);
};

resizeCanvas();

window.addEventListener("resize", () => {

  resizeCanvas();
  predictionDiv.innerText = "Prediction: ?";
});
