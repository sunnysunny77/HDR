const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const cors = require("cors");
const { createCanvas } = require("canvas");

const tf_app = express();
const allowedOrigins = ["https://hdr.localhost:3000", "https://hdr.sunnyhome.site"];

tf_app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  }
}));
tf_app.use(express.json({ limit: "10mb" }));

const port = 3001;
let model;
const labels = ["0","1","2","3","4","5","6","7","8","9"];
let currentLabels = [];
const numberWords = {
  "0": "zero",
  "1": "one",
  "2": "two",
  "3": "three",
  "4": "four",
  "5": "five",
  "6": "six",
  "7": "seven",
  "8": "eight",
  "9": "nine"
};

const loadModel = async () => {
  model = await tf.loadGraphModel("file://tfjs_model/model.json");
  console.log("Model loaded");
};
loadModel();

const drawLabel = (digit) => {
  const word = numberWords[digit];
  const canvas = createCanvas(50, 50);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const padding = 4;
  const chars = word.split("");
  const charWidth = (canvas.width - padding * 2) / chars.length;

  let fontSize = 8;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  while (true) {
    ctx.font = `bold ${fontSize}px sans-serif`;
    if (ctx.measureText(word).width >= canvas.width - padding * 2 || fontSize >= 20) break;
    fontSize++;
  }

  chars.forEach((char, i) => {
    const x = padding + i * charWidth + charWidth / 2;
    const y = canvas.height / 2 + (Math.random() * 6 - 3);
    const angle = (Math.random() * 40 - 20) * Math.PI / 180;

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = `rgb(${Math.random()*150|0}, ${Math.random()*150|0}, ${Math.random()*150|0})`;
    ctx.fillText(char, 0, 0);
    ctx.restore();
  });

  for (let i = 0; i < 50; i++) {
    ctx.fillStyle = `rgb(${Math.random()*255|0}, ${Math.random()*255|0}, ${Math.random()*255|0})`;
    ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, 1, 1);
  }

  for (let i = 0; i < 3; i++) {
    ctx.strokeStyle = `rgb(${Math.random()*255|0}, ${Math.random()*255|0}, ${Math.random()*255|0})`;
    ctx.beginPath();
    ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height);
    ctx.lineTo(Math.random() * canvas.width, Math.random() * canvas.height);
    ctx.stroke();
  }

  return canvas.toDataURL();
};


const getRandomLabels = (count = 4) => {
  return Array.from({ length: count }, () =>
    labels[Math.floor(Math.random() * labels.length)]
  );
};

const classifyImages = async (images, providedLabels, labels) => {
  return Promise.all(
    images.map(async (base64, i) => {
      const buffer = Buffer.from(base64, "base64");
      const predIndex = await processImageNode(buffer);
      const predLabel = predIndex !== null ? labels[predIndex] : null;

      return {
        correctLabel: providedLabels[i],
        predictedLabel: predLabel
      };
    })
  );
};

const processImageNode = async (imageBuffer) => {
  let img = tf.node.decodeImage(imageBuffer, 3).toFloat().div(255.0);
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

tf_app.post("/classify", async (req, res) => {
  try {
    if (!model) return res.status(503).json({ error: "Model not loaded yet" });
    const { images } = req.body;
    if (!(images?.length > 0)) return res.status(400).json({ error: "No images sent" });
    if (!currentLabels || currentLabels.length !== images.length) {
      return res.status(400).json({ error: "Server labels not set or mismatch" });
    }
    const results = await classifyImages(images, currentLabels, labels);
    res.json({ predictions: results });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error during classification", details: err.message });
  }
});

tf_app.get("/labels", (req, res) => {
  currentLabels = getRandomLabels(4);
  const labelImages = currentLabels.map(label => drawLabel(label));
  res.json({
    labels: currentLabels,
    images: labelImages
  });
});

tf_app.listen(port, () => {
  console.log(`Server live: http://localhost:${port}`);
});
