<!DOCTYPE html>
<html lang="en" data-overlayscrollbars-initialize>
<head>
  <meta charset="utf-8" />
  <meta name="description" content="Hand drawn recognition" />
  <meta name="keywords" content="Hand drawn recognition" />
  <meta name="author" content="D>C" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HDR</title>
  <link href="./css/app.min.css" rel="stylesheet" type="text/css" />
  <link rel="manifest" href="manifest.json" />
  <link rel="apple-touch-icon" href="images/pwa-logo-small.webp" />
</head>

<body data-overlayscrollbars-initialize>

  <main class="d-flex justify-content-center">

    <div class="hr-container d-flex flex-column" id="container">

      <h1 class="text-center mb-4">Hand drawn recognition</h1>

      <div class="mb-4 d-flex" id="canvas-wrapper">

        <canvas class="quad"></canvas>
        <div class="fill"></div>

      </div>

      <div class="row justify-content-evenly mb-4 g-0">

        <div class="col-8 col-sm-4 d-flex justify-content-center">

          <button class="btn button button-small btn-light w-100 my-1 p-1 button" id="resetBtn">New</button>

        </div>

        <div class="col-8 col-sm-4 d-flex justify-content-center">

          <button class="btn button button-small btn-light w-100 my-1 p-1 button" id="clearBtn">Clear</button>

        </div>

      </div>

      <div class="output-container mb-3">

        <div id="output" class="label-grid d-flex justify-content-center align-items-center">

          <img class="spinner" src="./images/spinner.gif" width="35" height="35" alt="spinner" />

        </div>

      </div>

      <div class="text-center alert w-100 d-flex justify-content-center align-items-center p-0 mb-3" role="alert" id="message">

        <img class="spinner" src="./images/spinner.gif" width="35" height="35" alt="spinner" />

      </div>

      <div class="d-flex flex-wrap justify-content-center">

        <button class="btn button btn-light button-large w-100 d-flex flex-warp align-items-center justify-content-center p-1" id="predictBtn">Send<span class="d-flex flex-warp align-items-center justify-content-center ms-2"><i class="fa-solid fa-arrow-right"></i></span></button>

      </div>

    </div>

  </main>

  <script src="./js/app.min.js" defer></script>

</body>

</html>