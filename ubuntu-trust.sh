#!/bin/bash

source "${INIT_CWD}/.env"

certfile="${INIT_CWD}/certs/ca.crt"
certname="${CN} Root CA"

sudo cp "${certfile}" /usr/local/share/ca-certificates/"${CN}.crt"
sudo update-ca-certificates

mkdir -p "${HOME}/.pki/nssdb"
certutil -A -n "${certname}" -t "TCu,Cu,Tu" -i "${certfile}" -d sql:"${HOME}/.pki/nssdb"
