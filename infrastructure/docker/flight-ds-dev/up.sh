#!/bin/sh
#v1
docker stop flight-ds-dev || true
docker rm flight-ds-dev || true
docker run \
--restart unless-stopped \
--name flight-ds-dev \
--network vm3 \
--ip 172.18.0.5 \
-dit rrbotlab/flight-ds:dev-v1
docker logs -f flight-ds-dev

