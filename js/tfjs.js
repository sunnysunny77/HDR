import * as tf from "@tensorflow/tfjs";

let drawing = [false, false, false, false];

const SIZE = 140;
const INVERT = false;

const host = "http://localhost:3001";

const canvases = Array.from(document.querySelectorAll(".quad"));
const clearBtn = document.querySelector("#clearBtn");
const predictBtn = document.querySelector("#predictBtn");
const message = document.querySelector("#message");
const output = document.querySelector("#output");

const contexts = canvases.map(canvas => {
  canvas.width = SIZE;
  canvas.height = SIZE;
  return canvas.getContext("2d");
});

const setRandomLabels = async () => {
  try {
    const res = await fetch(`${host}/labels`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    output.innerHTML = `${data.images.map(img => `<img src="${img}" alt="label" />`).join("")}`;
  } catch (err) {
    console.error(err);
    message.innerText = "Error";
  }
};

const clear = async (text, reset) => {
  contexts.forEach(ctx => {
        if (INVERT) {
          ctx.fillStyle = "white";
          ctx.fillRect(0, 0, SIZE, SIZE);
        } else {
          ctx.clearRect(0, 0, SIZE, SIZE)
        }
    });
  if (reset) await setRandomLabels();
  message.innerText = text;
};

clearBtn.addEventListener("click", () => {
  clear("Draw a capital letter in the boxes", true);
});

predictBtn.addEventListener("click", async () => {
  try {
    predictBtn.disabled = true;
    message.innerText = "Checking";
    const tensors = canvases.map(canvas =>{
      const img = tf.browser.fromPixels(canvas, 1).toFloat().div(255.0);
      return INVERT ? tf.sub(1.0, img) : img;
    });
    const images = tensors.map(tensor => ({
      data: Array.from(new Uint8Array(tensor.mul(255).dataSync())),
      shape: tensor.shape
    }));
    const res = await fetch(`${host}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images }),
    });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    let correct = true;
    data.predictions.forEach(prediction => {
      if (prediction.predictedLabel !== prediction.correctLabel) correct = false;
    });
    clear(correct ? "Correct" : "Incorrect", true);
  } catch (err) {
    console.error(err);
    message.innerText = "Error";
  } finally {
    predictBtn.disabled = false;
  }
});

const getCanvasCoords = (event, canvas) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return { x: (event.clientX - rect.left) * scaleX, y: (event.clientY - rect.top) * scaleY };
};

canvases.forEach((canvas, i) => {
  const ctx = contexts[i];
  canvas.addEventListener("pointerdown", event => {
    if (["mouse","pen","touch"].includes(event.pointerType)) {
      drawing[i] = true;
      const { x, y } = getCanvasCoords(event, canvas);
      ctx.strokeStyle = INVERT ? "black" : "white";
      ctx.lineWidth = Math.max(10, canvas.width / 16);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(x, y);
      event.preventDefault();
    }
  });
  canvas.addEventListener("pointermove", event => {
    if (drawing[i]) {
      const { x, y } = getCanvasCoords(event, canvas);
      ctx.lineTo(x, y);
      ctx.stroke();
      event.preventDefault();
    }
  });
  ["pointerup","pointercancel","pointerleave"].forEach(event =>
    canvas.addEventListener(event, () => (drawing[i] = false))
  );
});

export const tfjs = async () => {
  clear("Draw a capital letter in the boxes", true);
};
