#!/bin/bash
cd vision/nn_builder

docker build -f Dockerfile.x86 -t meraquetech/mardi_tech:nn-builder-x86 .

mkdir -p $PWD/saved_graphs $PWD/.cache

docker run --rm -d \
  -p 5050:5050 \
  --name=nn-builder \
  -v $PWD/saved_graphs:/app/saved_graphs \
  -v $PWD/.cache:/app/.cache \
  meraquetech/mardi_tech:nn-builder-x86

echo "Waiting for container to start..."
sleep 2

docker logs -f nn-builder

# ── Useful commands ───────────────────────────────────────────────────────────
# docker exec -it nn-builder bash
# docker rm -f nn-builder
#
# UI: http://<host-ip>:5050/
