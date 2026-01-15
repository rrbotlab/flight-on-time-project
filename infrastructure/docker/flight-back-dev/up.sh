#!/bin/sh
#v3

# 1. Descobre o diretório onde o script está localizado
# SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 2. Localiza o .env a partir dessa pasta (mesmo que chamado de longe)
ENV_FILE="$SCRIPT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "$ENV_FILE not found!"
    exit 1
fi

docker stop flight-back-dev || true
docker rm flight-back-dev || true
docker run \
--restart unless-stopped \
--name flight-back-dev \
--network vm3 \
--ip 172.18.0.12 \
-e FLIGHTONTIME_DATASOURCE_DEV="$FLIGHTONTIME_DATASOURCE_DEV" \
-e FLIGHTONTIME_USERNAME_DEV="$FLIGHTONTIME_USERNAME_DEV" \
-e FLIGHTONTIME_PASSWORD_DEV="$FLIGHTONTIME_PASSWORD_DEV" \
-e FLIGHTONTIME_DATASCIENCE_BASEURL="$FLIGHTONTIME_DATASCIENCE_BASEURL" \
-e FLIGHTONTIME_JWT_SECRET_DEV="$FLIGHTONTIME_JWT_SECRET_DEV" \
-dit rrbotlab/flight-back:dev-v3
docker logs -f flight-back-dev

