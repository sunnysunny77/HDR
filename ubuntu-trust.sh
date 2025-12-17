#!/bin/bash
set -euo pipefail

source "${INIT_CWD}/.env"

if [[ -z "${CN:-}" ]]; then
    echo "ERROR: CN is not set in .env"
    exit 1
fi

certfile="${INIT_CWD}/certs/ca.crt"
certname="${CN} Root CA"

echo "Installing system-wide certificate..."
sudo cp "${certfile}" /usr/local/share/ca-certificates/"${CN}.crt"
sudo update-ca-certificates

echo "Installing user NSS DB certificate (~/.pki/nssdb)..."
mkdir -p "${HOME}/.pki/nssdb"
certutil -A -n "${certname}" -t "TCu,Cu,Tu" -i "${certfile}" -d sql:"${HOME}/.pki/nssdb"
