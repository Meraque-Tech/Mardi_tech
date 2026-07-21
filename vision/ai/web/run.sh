#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.train_web.yml"
ACTION="${1:-start}"

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    echo "[ERROR] Docker Compose is not installed."
    echo "Install the Docker Compose plugin and try again."
    exit 1
fi

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

compose() {
    "${COMPOSE[@]}" -f "$COMPOSE_FILE" "$@"
}

case "$ACTION" in
    start)
        echo "[YOLOv8 Web UI] Starting container..."
        compose up -d
        compose ps
        echo
        echo "Web UI: http://localhost:8000"
        echo "Logs:   ./run.sh logs"
        echo "Stop:   ./run.sh stop"
        ;;
    build)
        echo "[YOLOv8 Web UI] Building and starting container..."
        compose up -d --build
        compose ps
        echo
        echo "Web UI: http://localhost:8000"
        ;;
    restart)
        echo "[YOLOv8 Web UI] Restarting container..."
        compose restart
        compose ps
        ;;
    stop)
        echo "[YOLOv8 Web UI] Stopping container..."
        compose down
        ;;
    logs)
        compose logs -f --tail=200
        ;;
    status)
        compose ps
        ;;
    *)
        echo "Usage: $0 [start|build|restart|stop|logs|status]"
        exit 1
        ;;
esac
