import * as tf from "@tensorflow/tfjs";

const labels = [
  "0","1","2","3","4","5","6","7","8","9",
  "A","B","C","D","E","F","G","H","I","J",
  "K","L","M","N","O","P","Q","R","S","T",
  "U","V","W","X","Y","Z",
];

const filter = ["0","O","5","S"];

let model;

let drawing = [false, false, false, false];

let currentLabels = [];

const SIZE = 140;

const canvases = [
  document.getElementById("canvas-0"),
  document.getElementById("canvas-1"),
  document.getElementById("canvas-2"),
  document.getElementById("canvas-3"),
];
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const message = document.getElementById("message");
const output = document.getElementById("output");

const getRandomLabel = () => {
  const allowed = labels.filter(l => !filter.includes(l));
  return allowed[Math.floor(Math.random() * allowed.length)];
};

const setRandomLabels = () => {
  currentLabels = [getRandomLabel(), getRandomLabel(), getRandomLabel(), getRandomLabel()];
  output.innerHTML = currentLabels.map(label => `<div>${label}</div>`).join("");
};

const contexts = canvases.map(c => {
  c.width = SIZE;
  c.height = SIZE;
  return c.getContext("2d");
});

const clear = (text, reset) => {
  contexts.forEach(ctx => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, SIZE, SIZE);
  });
  if (reset) setRandomLabels();
  message.innerText = text;
};

clearBtn.addEventListener("click", () => {
  clear("Draw the required characters", true);
});

const processCanvas = async (canvas) => {
  const img = tf.browser.fromPixels(canvas, 3).toFloat().div(255.0);

  const mask = img.greater(0.1);
  const coords = await tf.whereAsync(mask);

  if (coords.shape[0] === 0) {
    img.dispose();
    mask.dispose();
    coords.dispose();
    return null;
  }

  const ys = coords.slice([0, 0], [-1, 1]).squeeze();
  const xs = coords.slice([0, 1], [-1, 1]).squeeze();

  const minY = ys.min().arraySync();
  const maxY = ys.max().arraySync();
  const minX = xs.min().arraySync();
  const maxX = xs.max().arraySync();

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;

  let imgTensor = img.mean(2).expandDims(2);
  img.dispose();
  mask.dispose();
  coords.dispose();
  ys.dispose();
  xs.dispose();

  imgTensor = imgTensor.slice([minY, minX, 0], [height, width, 1]);

  const scale = 20 / Math.max(height, width);
  const newHeight = Math.round(height * scale);
  const newWidth = Math.round(width * scale);

  imgTensor = imgTensor.resizeBilinear([newHeight, newWidth]);

  const top = Math.floor((28 - newHeight) / 2);
  const bottom = 28 - newHeight - top;
  const left = Math.floor((28 - newWidth) / 2);
  const right = 28 - newWidth - left;

  imgTensor = imgTensor.pad([[top, bottom], [left, right], [0, 0]]).expandDims(0);

  const prediction = model.predict(imgTensor);
  const maxIndex = prediction.argMax(-1).dataSync()[0];

  prediction.dispose();
  imgTensor.dispose();

  return maxIndex;
};

predictBtn.addEventListener("click", async () => {
  if (!model) {
    alert("Model not loaded yet.");
    return;
  }

  predictBtn.disabled = true;
  message.innerText = "Checking...";
  await tf.nextFrame();

  let correct = true;

  for (let i = 0; i < 4; i++) {
    const maxIndex = await processCanvas(canvases[i]);
    if (maxIndex === null || currentLabels[i] !== labels[maxIndex]) {
      correct = false;
      break;
    }
  }

  correct ? (message.innerText = "Correct") : clear("Incorrect", false);
  predictBtn.disabled = false;
});

const getCanvasCoords = (e, canvas) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
};

canvases.forEach((canvas, i) => {
  const ctx = contexts[i];
  canvas.addEventListener("pointerdown", e => {
    if (["mouse","pen","touch"].includes(e.pointerType)) {
      drawing[i] = true;
      const { x, y } = getCanvasCoords(e, canvas);
      ctx.strokeStyle = "white";
      ctx.lineWidth = Math.max(10, canvas.width / 16);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(x, y);
      e.preventDefault();
    }
  });
  canvas.addEventListener("pointermove", e => {
    if (drawing[i]) {
      const { x, y } = getCanvasCoords(e, canvas);
      ctx.lineTo(x, y);
      ctx.stroke();
      e.preventDefault();
    }
  });
  ["pointerup","pointercancel","pointerleave"].forEach(evt =>
    canvas.addEventListener(evt, () => (drawing[i] = false))
  );
});

export const tfjs = async () => {
  message.innerText = "Loading model...";
  try {
    await tf.setBackend("webgl");
    await tf.ready();
  } catch {
    await tf.setBackend("cpu");
    await tf.ready();
  }
  try {
    model = await tf.loadGraphModel("tfjs_model/model.json");
    clear("Draw the required characters", true);
  } catch (error) {
    message.innerText = "Failed to load model.";
    console.error("Model loading error:", error);
  }
};