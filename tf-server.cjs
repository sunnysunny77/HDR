const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const cors = require("cors");
const { createCanvas } = require("canvas");

const PORT = 3001;
const app = express();

const allowedOrigins = ["https://hdr.localhost:3000", "https://hdr.sunnyhome.site"];

app.use(cors({origin: (origin, callback) => {
  if (!origin) return callback(null, true);
  if (allowedOrigins.includes(origin)) {
    callback(null, true);
  } else {
    callback(new Error("Not allowed by CORS"));
  };
}}));
app.use(express.json({ limit: "10mb" }));

let model;

const labels = [
  "THAT", "WITH", "THIS", "THEY", "FROM", "HAVE", "WILL", "WERE", "BEEN", "MORE",
  "SAID", "WHEN", "ONLY", "THEM", "LAST", "THAN", "INTO", "MADE", "SOME", "MOST",
  "WHAT", "GOOD", "TIME", "EVEN", "ALSO", "WEST", "MANY", "SUCH", "OVER", "LIFE",
  "THEN", "USED", "FILM", "WORK", "MUCH", "MAKE", "LIKE", "MUST", "PLAY", "WEEK",
  "YOUR", "TAKE", "NEED", "MISS", "WELL", "DOWN", "YEAR", "HERE", "PART", "DAYS",
  "VERY", "HOME", "FOUR", "JUST", "LONG", "BACK", "SEEN", "EACH", "WENT", "FULL",
  "CASE", "NEXT", "BOTH", "DONE", "TOLD", "WANT", "COME", "SHOW", "NATO", "LEFT",
  "LORD", "SAME", "KNOW", "DOES", "VIEW", "GIRL", "HELP", "FACT", "TEXT", "SIDE",
  "SAYS", "REAL", "MIND", "LOVE", "GIVE", "LIVE", "EVER", "FACE", "ROOM", "BOOK",
  "FIVE", "HALF", "BEST", "EAST", "PLAN", "EASY", "FIND", "FREE", "REST"
];

let currentLabel = "";

const loadModel = async () => {
  if (!model) {
    model = await tf.loadGraphModel("file://tfjs_model/model.json");
    console.log("Model loaded");
  };
};

const drawLabel = (label) => {
  const width = 250;
  const height = 100;
  const fill = "white";
  const dotCount = 350;
  const lineStyle = "rgba(0,0,0,0.34)";
  const lineWidth = 0.5;
  const fontSize = 30;
  const font = `bold ${fontSize}px Sans`;
  const spacing = 0.5;

  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < dotCount; i++) {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    const alpha = Math.random() < 0.5 ? 0.3 : 0.6;
    const radius = Math.random() * 3 + 1;
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.beginPath();
    ctx.arc(Math.random() * canvas.width, Math.random() * canvas.height, radius, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.strokeStyle = lineStyle;
  ctx.lineWidth = lineWidth;
  for (let j = 0; j < 2; j++) {
    ctx.beginPath();
    ctx.moveTo(0, Math.random() * canvas.height);
    for (let x = 0; x < canvas.width; x += 5) {
      ctx.lineTo(
        x,
        (canvas.height / 2) + Math.sin(x / 5 + Math.random() * 2) * 12 + (Math.random() * 20 - 10)
      );
    }
    ctx.stroke();
  }

  ctx.font = font;
  let totalWidth = 0;
  const charWidths = [];
  for (const char of label) {
    const w = ctx.measureText(char).width;
    charWidths.push(w);
    totalWidth += w * (1 + spacing);
  }

  let x = (width - totalWidth) / 2;

  for (let i = 0; i < label.length; i++) {
    const char = label[i];
    const angle = (Math.random() - 0.5) * 0.6;
    const offsetY = (Math.random() - 0.5) * 18;
    const min = 50;
    const max = 150;
    const r = Math.floor(Math.random() * (max - min) + min);
    const g = Math.floor(Math.random() * (max - min) + min);
    const b = Math.floor(Math.random() * (max - min) + min);
    ctx.save();
    ctx.fillStyle = `rgba(${r},${g},${b},1)`;
    ctx.translate(x, height / 2 + offsetY);
    ctx.rotate(angle);
    ctx.fillText(char, 0, 0);
    ctx.restore();

    x += charWidths[i] * (1 + spacing);
  }

  return canvas.toDataURL();
};

app.get("/labels", (req, res) => {
  currentLabel = labels[Math.floor(Math.random() * labels.length)];
  res.json({image: drawLabel(currentLabel)});
});

const processImageNode = async (data, shape) => {
  const tensor = tf.tensor(data, shape, "float32").div(255.0);
  const input = tensor.resizeBilinear([28, 112]).expandDims(0);
  const prediction = model.predict(input);
  const maxIndex = prediction.argMax(-1).dataSync()[0];

  tensor.dispose();
  input.dispose();
  prediction.dispose();

  return maxIndex;
};

app.post("/classify", async (req, res) => {
  try {
    if (!model) await loadModel();

    const { image } = req.body;

    if (!currentLabel) {
      throw new Error("Error");
    }

   const predIndex = await processImageNode(image.data, image.shape);

    res.json({ correct: currentLabel === labels[predIndex] });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error" });
  }
});

app.listen(PORT, () => {
  console.log(`Server live: http://localhost:${PORT}`);
});
