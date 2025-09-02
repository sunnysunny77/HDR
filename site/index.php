<!DOCTYPE html>
<html lang="en" data-overlayscrollbars-initialize>
<head>
  <meta charset="utf-8" />
  <meta name="description" content="HDR" />
  <meta name="keywords" content="HDR" />
  <meta name="author" content="D>C" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HDR</title>
  <link href="./css/app.min.css" rel="stylesheet" type="text/css" />
  <link rel="manifest" href="manifest.json" />
  <link rel="apple-touch-icon" href="images/pwa-logo-small.webp" />
</head>

<body data-overlayscrollbars-initialize>

  <div class="container" id="container">

    <h1>Handwritten recognition</h1>

    <div id="canvas-wrapper">

      <canvas class="quad" id="canvas-0"></canvas>
      <canvas class="quad" id="canvas-1"></canvas>
      <canvas class="quad" id="canvas-2"></canvas>
      <canvas class="quad" id="canvas-3"></canvas>

    </div>

    <div class="buttons">

      <button id="clearBtn">Reset</button>

      <button id="predictBtn">Submit</button>

    </div>

    <div id="output" class="label-grid">

      <div></div>
      <div></div>
      <div></div>
      <div></div>

    </div>

    <div id="message">

      Loading model...

    </div>

  </div>

  <script src="./js/app.min.js" defer></script>

</body>

</html>