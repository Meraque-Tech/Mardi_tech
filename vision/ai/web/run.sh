#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.train_web.yml"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"
RUNS_DIR="$REPO_ROOT/runs"
DATASETS_DIR="$SCRIPT_DIR/datasets"
LOGS_DIR="$SCRIPT_DIR/logs"
ACTION="${1:-start}"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed or is not on PATH."
    exit 1
fi

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

if ! [[ "$HOST_UID" =~ ^[0-9]+$ && "$HOST_GID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] HOST_UID and HOST_GID must be numeric."
    exit 1
fi

compose() {
    "${COMPOSE[@]}" -f "$COMPOSE_FILE" "$@"
}

ensure_env_file() {
    if [[ -f "$ENV_FILE" ]]; then
        return
    fi

    if [[ ! -f "$ENV_EXAMPLE" ]]; then
        echo "[ERROR] Missing $ENV_FILE and $ENV_EXAMPLE."
        exit 1
    fi

    cp -- "$ENV_EXAMPLE" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "[YOLOv8 Web UI] Created $ENV_FILE from .env.example."
    echo "[YOLOv8 Web UI] Add any required credentials before using dataset services."
}

ensure_runtime_dirs() {
    local directory

    for directory in "$RUNS_DIR" "$DATASETS_DIR" "$LOGS_DIR"; do
        if [[ -e "$directory" && ! -d "$directory" ]]; then
            echo "[ERROR] Runtime path exists but is not a directory: $directory"
            exit 1
        fi

        if ! mkdir -p -- "$directory"; then
            echo "[ERROR] Cannot create runtime directory: $directory"
            echo "Fix its ownership and run this command again:"
            echo "  sudo chown -R $(id -un):$(id -gn) \"$directory\""
            exit 1
        fi

        if [[ ! -w "$directory" || ! -x "$directory" ]]; then
            echo "[ERROR] Runtime directory is not writable: $directory"
            echo "Fix its ownership and run this command again:"
            echo "  sudo chown -R $(id -un):$(id -gn) \"$directory\""
            exit 1
        fi
    done
}

check_docker_daemon() {
    if ! docker info >/dev/null 2>&1; then
        echo "[ERROR] Docker is installed but the Docker daemon is unavailable."
        echo "Start Docker and run this command again."
        exit 1
    fi
}

check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        if ! nvidia-smi -L >/dev/null 2>&1; then
            echo "[WARNING] nvidia-smi could not detect a GPU."
            echo "The CUDA container may not start correctly."
        fi
    else
        echo "[WARNING] nvidia-smi is not installed."
        echo "Verify that NVIDIA drivers and the NVIDIA Container Toolkit are installed."
    fi
}

wait_for_ready() {
    local attempt

    if ! command -v curl >/dev/null 2>&1; then
        echo "[WARNING] curl is not installed; skipping the web UI readiness check."
        return 0
    fi

    for ((attempt = 1; attempt <= 30; attempt++)); do
        if curl --silent --show-error --fail --max-time 2 \
            http://127.0.0.1:8000/openapi.json >/dev/null 2>&1; then
            echo "[YOLOv8 Web UI] Ready: http://localhost:8000"
            return 0
        fi
        sleep 2
    done

    echo "[ERROR] The container started but the web UI did not become ready."
    compose ps
    echo
    echo "Recent container logs:"
    compose logs --tail=100
    exit 1
}

prepare_start() {
    check_docker_daemon
    ensure_env_file
    ensure_runtime_dirs
    check_gpu
}

case "$ACTION" in
    start)
        prepare_start
        echo "[YOLOv8 Web UI] Starting container..."
        compose up -d
        compose ps
        wait_for_ready
        echo
        echo "Logs:   ./run.sh logs"
        echo "Stop:   ./run.sh stop"
        ;;
    build)
        prepare_start
        echo "[YOLOv8 Web UI] Building and starting container..."
        compose up -d --build
        compose ps
        wait_for_ready
        echo
        ;;
    restart)
        check_docker_daemon
        ensure_runtime_dirs
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
