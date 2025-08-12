import * as tf from "@tensorflow/tfjs";

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
    const [maxIndex, maxVal] = tf.tidy(() => {
      let img = tf.browser.fromPixels(canvas);
      img = tf.image.resizeBilinear(img, [28, 28]);
      img = img.mean(2).toFloat();
      img = img.div(255);
      const input = img.expandDims(0).expandDims(-1);
      const prediction = model.predict(input);
      const maxIndexTensor = prediction.argMax(-1);
      const maxValTensor = prediction.max(-1);
      return [maxIndexTensor.dataSync()[0], maxValTensor.dataSync()[0]];
    });

    predictionDiv.innerText = `Prediction: ${maxIndex} (Confidence: ${maxVal.toFixed(4)})`;
  } catch (err) {

    predictionDiv.innerText = "Prediction failed.";
    console.error(err);
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
    ctx.beginPath();
    ctx.moveTo(x, y);
    e.preventDefault();
  }
});

canvas.addEventListener("pointermove", e => {

  if (drawing) {

    const { x, y } = getCanvasCoords(e);
    ctx.strokeStyle = "white";
    ctx.lineWidth = Math.max(10, Math.min(canvas.width, canvas.height) / 20);
    ctx.lineCap = "round";
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
