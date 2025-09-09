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

  <div class="hr-container d-flex flex-column" id="container">

    <h1 class="text-center mb-2">Handwritten recognition</h1>

    <div class="mb-4" id="canvas-wrapper">

      <canvas class="quad"></canvas>
      <div class="fill"></div>

    </div>

   <div class="d-flex flex-wrap justify-content-center mb-4">

      <button class="btn btn-success m-2 button" id="clearBtn">Reset</button>

      <button class="btn btn-success m-2 button" id="predictBtn">Submit</button>

    </div>

    <div class="output-container mb-3">

      <div id="output" class="label-grid">

        <img class="spinner" src="./images/spinner.gif" width="250" height="100" alt="spinner" />

      </div>

    </div>

    <div class="text-center alert alert-success p-2 w-100 mb-0" role="alert" id="message">

      Loading model

    </div>

  </div>

  <script src="./js/app.min.js" defer></script>

</body>

</html>