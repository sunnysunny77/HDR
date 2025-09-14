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

<body data-overlayscrollbars-initialize class="d-flex justify-content-center">

  <div class="hr-container d-flex flex-column" id="container">

    <h1 class="text-center mb-4">Handwritten recognition</h1>

    <div class="mb-4 d-flex" id="canvas-wrapper">

      <canvas class="quad"></canvas>
      <div class="fill"></div>

    </div>

    <div class="d-flex flex-wrap justify-content-center mb-4">

      <button class="btn btn-success mx-2 p-1 button" id="resetBtn">New</button>

      <button class="btn btn-success mx-2 p-1 button" id="clearBtn">Clear</button>

    </div>

    <div class="output-container mb-3">

      <div id="output" class="label-grid d-flex justify-content-center align-items-center">

        <img class="spinner" src="./images/spinner.gif" width="70" height="70" alt="spinner" />

      </div>

    </div>

    <div class="text-center alert alert-success w-100 d-flex justify-content-center align-items-center p-0 mb-3" role="alert" id="message">

      <img class="spinner" src="./images/spinner.gif" width="30" height="30" alt="spinner" />

    </div>

    <div class="d-flex flex-wrap justify-content-center">

      <button class="btn btn-success w-100 p-1" id="predictBtn">Send</button>

    </div>

  </div>

  <script src="./js/app.min.js" defer></script>

</body>

</html>