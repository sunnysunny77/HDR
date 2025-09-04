const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const cors = require("cors");

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

const loadModel = async () => {
  model = await tf.loadGraphModel("file://tfjs_model/model.json");
  console.log("Model loaded");
};
loadModel();

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
    if (!currentLabels || currentLabels.length !== images.length) return res.status(400).json({ error: "Server labels not set or mismatch" });
    const results = await classifyImages(images, currentLabels, labels);
    res.json({ predictions: results });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error during classification", details: err.message });
  }
});

tf_app.get("/labels", (req, res) => {
  currentLabels = getRandomLabels(4);
  res.json({ labels: currentLabels });
});

tf_app.listen(port, () => {
  console.log(`Server live: http://localhost:${port}`);
});
