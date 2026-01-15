#!/bin/sh
#v1
# --ip 172.18.0.7 \
docker stop flight-front-dev || true
docker rm flight-front-dev || true
docker run \
--restart unless-stopped \
--name flight-front-dev \
--network vm3 \
-dit rrbotlab/flight-front:dev-v1 
docker logs -f flight-front-dev
