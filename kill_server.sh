#!/bin/bash
# xhost +local:
# xhost +local:
# docker rm -f $(docker ps -aq)
# docker compose down
# docker compose -f docker-compose.rviz2.yaml down

# sudo chown -R $USER:$USER ${HOME}/ZED_BAG
# sudo chown -R $USER:$USER ${HOME}/yolo_models
docker compose down
# docker compose -f docker-compose.prime.yaml down