import * as tf from "@tensorflow/tfjs";

let drawing = false;

const CANVAS_WIDTH = 280;
const CANVAS_HEIGHT = 140;
const INVERT = false;

const host = "https://hdr.localhost:3000/api";

const canvas = document.querySelector(".quad");
const resetBtn = document.querySelector("#resetBtn");
const predictBtn = document.querySelector("#predictBtn");
const clearBtn = document.querySelector("#clearBtn");
const message = document.querySelector("#message");
const output = document.querySelector("#output");

canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;
const ctx = canvas.getContext("2d");

const setRandomLabels = async () => {
  try {
    const res = await fetch(`${host}/labels`, {
      credentials: "include"
    });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    output.innerHTML = `<img src="${data.image}" alt="label" />`;
  } catch (err) {
    console.error(err);
    message.innerText = "Error";
  }
};

const clear = () => {
  if (INVERT) {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  } else {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }
};

resetBtn.addEventListener("click",  async () => {
  clear();
  await setRandomLabels();
  message.innerText = "Draw the word";
});

clearBtn.addEventListener("click", () => {
  clear();
  message.innerText = "Draw the word";
});

const resizeCanvas = (obj) => {
  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = 112;
  resizedCanvas.height = 28;
  const resizedCtx = resizedCanvas.getContext("2d");

  resizedCtx.drawImage(
    obj,
    0, 0, CANVAS_WIDTH, CANVAS_HEIGHT,
    0, 0, 112, 28
  );

  return resizedCanvas;
};

const invertCanvas = (ctx) => {
  const image = ctx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  const invertedCanvas = document.createElement("canvas");
  invertedCanvas.width = CANVAS_WIDTH;
  invertedCanvas.height = CANVAS_HEIGHT;
  const invertedCtx = invertedCanvas.getContext("2d");

  const invertedData = ctx.createImageData(image.width, image.height);

  for (let i = 0; i < image.data.length; i += 4) {
    invertedData.data[i] = 255 - image.data[i];
    invertedData.data[i + 1] = 255 - image.data[i + 1];
    invertedData.data[i + 2] = 255 - image.data[i + 2];
    invertedData.data[i + 3] = image.data[i + 3];
  };

  invertedCtx.putImageData(invertedData, 0, 0);

  return invertedCanvas;
};

predictBtn.addEventListener("click", async () => {
  try {
    predictBtn.disabled = true;
    message.innerHTML = "<img class='spinner' src='./images/spinner.gif' width='30' height='30' alt='spinner' />";

    const obj = INVERT ? invertCanvas(ctx) : canvas;
    const resized = resizeCanvas(obj);
    const img = tf.browser.fromPixels(resized, 1).toFloat().div(255.0);

    const image = {
      data: Array.from(new Uint8Array(img.mul(255).dataSync())),
      shape: img.shape
    };

    img.dispose();

    const res = await fetch(`${host}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: image }),
      credentials: "include"
    });

    if (!res.ok) throw new Error(res.statusText);

    const data = await res.json();

    message.innerText = data.correct ? "Correct" : "Incorrect";
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
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
};

canvas.addEventListener("pointerdown", event => {
  if (["mouse", "pen", "touch"].includes(event.pointerType)) {
    drawing = true;
    const { x, y } = getCanvasCoords(event, canvas);
    ctx.strokeStyle = INVERT ? "black" : "white";
    const minDim = Math.min(canvas.width, canvas.height);
    ctx.lineWidth = Math.max(1, Math.round(minDim / 18));
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(x, y);
    event.preventDefault();
  }
});

canvas.addEventListener("pointermove", event => {
  if (drawing) {
    const { x, y } = getCanvasCoords(event, canvas);
    ctx.lineTo(x, y);
    ctx.stroke();
    event.preventDefault();
  }
});

["pointerup", "pointercancel", "pointerleave"].forEach(event =>
  canvas.addEventListener(event, () => (drawing = false))
);

export const tfjs = async () => {
  if (INVERT) {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  } else {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }
  await setRandomLabels();
  message.innerText = "Draw the word";
};
