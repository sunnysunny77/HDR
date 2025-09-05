const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const cors = require("cors");
const { createCanvas } = require("canvas");

const app = express();
const allowedOrigins = ["https://hdr.localhost:3000", "https://hdr.sunnyhome.site"];

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  }
}));
app.use(express.json({ limit: "10mb" }));

const PORT = 3001;
let model;

const CLASS_LABELS = [
  "A","B","C","D","E","F","G","H","I","J","K","L","M",
  "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
];

const PHONETIC_MAP = {
  "A": "ALPHA",   "B": "BRAVO",   "C": "CHARLIE", "D": "DELTA",
  "E": "ECHO",    "F": "FOXTROT", "G": "GOLF",    "H": "HOTEL",
  "I": "INDIA",   "J": "JULIET",  "K": "KILO",    "L": "LIMA",
  "M": "MIKE",    "N": "NOVEMBER","O": "OSCAR",   "P": "PAPA",
  "Q": "QUEBEC",  "R": "ROMEO",   "S": "SIERRA",  "T": "TANGO",
  "U": "UNIFORM", "V": "VICTOR",  "W": "WHISKEY", "X": "XRAY",
  "Y": "YANKEE",  "Z": "ZULU"
};

let activeLabels = [];

const loadModel = async () => {
  model = await tf.loadGraphModel("file://tfjs_model/model.json");
  console.log("Model loaded");
};
loadModel();

const drawPhoneticLabel = (label) => {
  const word = PHONETIC_MAP[label];
  const canvasSize = 122;
  const canvas = createCanvas(canvasSize, canvasSize);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const dotCount = 100;
  for (let i = 0; i < dotCount; i++) {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    const alpha = Math.random() < 0.5 ? 0.3 : 0.8;
    const radius = Math.random() * 3 + 1;
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.beginPath();
    ctx.arc(Math.random() * canvas.width, Math.random() * canvas.height, radius, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.strokeStyle = "rgba(0,0,0,0.2)";
  ctx.lineWidth = 0.5;
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

  const fontSize = 18;
  ctx.font = `bold ${fontSize}px Sans`;

  // Calculate total word width with spacing
  let totalWidth = 0;
  for (let char of word) {
    totalWidth += ctx.measureText(char).width * 0.8; // spacing factor
  }

  // Start x so word is horizontally centered
  let x = (canvas.width - totalWidth) / 2;

  for (let char of word) {
    const angle = (Math.random() - 0.5) * 0.6;
    const offsetY = (Math.random() - 0.5) * 18;
    const color = `rgba(${Math.floor(Math.random() * 256)},${Math.floor(Math.random() * 256)},${Math.floor(Math.random() * 256)},1)`;

    ctx.save();
    ctx.fillStyle = color;
    ctx.translate(x, canvas.height / 2 + offsetY);
    ctx.rotate(angle);
    ctx.fillText(char, 0, 0);
    ctx.restore();

    x += ctx.measureText(char).width * 0.8; // move x for next char
  }


  return canvas.toDataURL();
};

const getRandomLabels = (count = 4) => {
  return Array.from({ length: count }, () =>
    CLASS_LABELS[Math.floor(Math.random() * CLASS_LABELS.length)]
  );
};

const processImageNode = async (imageBuffer) => {
  let img = tf.node.decodeImage(imageBuffer, 3).toFloat().div(255.0);
  const mask = img.greater(0.1);
  const coords = await tf.whereAsync(mask);

  if (coords.shape[0] === 0) {
    img.dispose(); mask.dispose(); coords.dispose();
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
  img.dispose(); mask.dispose(); coords.dispose(); ys.dispose(); xs.dispose();

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

app.post("/classify", async (req, res) => {
  try {
    if (!model) return res.status(503).json({ error: "Model not loaded yet" });
    const { images } = req.body;
    if (!(images?.length > 0)) return res.status(400).json({ error: "No images sent" });
    if (!activeLabels || activeLabels.length !== images.length) {
      return res.status(400).json({ error: "Server labels not set or mismatch" });
    }

    const results = await Promise.all(images.map(async (base64, i) => {
      const buffer = Buffer.from(base64, "base64");
      const predIndex = await processImageNode(buffer);
      const predictedLabel = predIndex !== null ? CLASS_LABELS[predIndex] : null;

      return {
        correctLabel: activeLabels[i],
        predictedLabel
      };
    }));

    res.json({ predictions: results });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error during classification", details: err.message });
  }
});

app.get("/labels", (req, res) => {
  activeLabels = getRandomLabels(4);
  const labelImages = activeLabels.map(label => drawPhoneticLabel(label));
  res.json({
    labels: activeLabels,
    images: labelImages
  });
});

app.listen(PORT, () => {
  console.log(`Server live: http://localhost:${PORT}`);
});
