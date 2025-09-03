let drawing = [false, false, false, false];

const SIZE = 140;

const host = "https://localhost:3001"

const canvases = Array.from(document.querySelectorAll(".quad"));
const clearBtn = document.querySelector("#clearBtn");
const predictBtn = document.querySelector("#predictBtn");
const message = document.querySelector("#message");
const output = document.querySelector("#output");

const contexts = canvases.map(c => {
  c.width = SIZE;
  c.height = SIZE;
  return c.getContext("2d");
});

const setRandomLabels = async () => {
  try {
    const res = await fetch(`${host}/labels`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    output.innerHTML = data.labels.map(label => `<div>${label}</div>`).join("");
  } catch (err) {
    console.error(err);
    message.innerText = "Error fetching labels";
  }
};

const clear = async (text, reset) => {
  contexts.forEach(ctx => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, SIZE, SIZE);
  });
  if (reset) await setRandomLabels();
  message.innerText = text;
};

clearBtn.addEventListener("click", () => {
  clear("Draw the required characters", true);
});

predictBtn.addEventListener("click", async () => {
  try {
    predictBtn.disabled = true;
    message.innerText = "Checking...";

    const images = canvases.map(c => c.toDataURL("image/png").split(",")[1]);

    const response = await fetch(`${host}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images }),
    });

    if (!response.ok) {
      message.innerText = `Server error: ${response.status}`;
      return;
    }

    const data = await response.json();
    let allCorrect = true;

    data.predictions.forEach(p => {
      if (p.predictedLabel !== p.correctLabel) allCorrect = false;
    });

    message.innerText = allCorrect ? "All Correct!" : "Some answers are incorrect";

  } catch (err) {
    console.error(err);
    message.innerText = "Error";
  } finally {
    predictBtn.disabled = false;
  }
});

const getCanvasCoords = (e, canvas) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
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
  clear("Draw the required characters", true);
};
