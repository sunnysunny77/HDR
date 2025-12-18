#!/bin/bash

set -e

set -x

source "${INIT_CWD}/.env"

npm run watch-js &

S1=$!

npm run php-server &

S2=$!

npm run https-server &

S3=$!

npm run express-server &

S4=$!

npm run watch-css &

S5=$!

open "https://${CN}:3000" &

S6=$!

wait
 
kill "${S1}" "${S2}" "${S3}" "${S4}" "${S5}" "${S6}"