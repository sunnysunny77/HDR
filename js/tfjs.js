import * as tf from "@tensorflow/tfjs";

let model;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictionDiv = document.getElementById("prediction");

function resizeCanvas() {
  const maxWidth = 400; // max width in px
  const remInPx = parseFloat(getComputedStyle(document.documentElement).fontSize); // 1rem in px
  const padding = 20 + remInPx * 2; // 20px + 2rem

  const viewportWidth = window.innerWidth;
  const canvasWidth = Math.min(viewportWidth - padding, maxWidth);

  // Set canvas display size (CSS)
  canvas.style.width = `${canvasWidth}px`;
  canvas.style.height = `${canvasWidth}px`;

  // Set canvas internal pixel size
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvasWidth * dpr;
  canvas.height = canvasWidth * dpr;

  // Reset and apply transform
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);

  // Fill background black
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvasWidth, canvasWidth);
}

resizeCanvas();
window.addEventListener("resize", () => {
  resizeCanvas();
});

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
  // Use an offscreen canvas to avoid scaling artifacts
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");

  // Draw scaled-down version of the main canvas
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  return tf.tidy(() => {
    const img = tf.browser.fromPixels(tempCanvas, 1)
      .toFloat()
      .div(255.0);

    // Invert colors (white digit on black → black digit on white)
    const inverted = tf.sub(1, img);
    return inverted.expandDims(0); // shape: [1, 28, 28, 1]
  });
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

  const labelMap = "0123456789"; // update if you have more classes
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
    model = await tf.loadGraphModel("tfjs_model/model.json");
    predictionDiv.innerText = "Model ready. Draw and click Predict.";
    console.log("Model loaded.");
  } catch (error) {
    predictionDiv.innerText = "Failed to load model.";
    console.error("Model loading error:", error);
  }
};
