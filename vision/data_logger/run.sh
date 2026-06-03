#!/bin/bash
set -e

# ─────────────────────────────────────────────
# run.sh — build and start data_logger
# Usage:
#   ./run.sh            auto-detect platform
#   ./run.sh x86        force x86 mode
#   ./run.sh jetson     force Jetson Nano mode
#   ./run.sh stop       stop running container
#   ./run.sh logs       tail container logs
# ─────────────────────────────────────────────

COMPOSE_X86="docker-compose.x86.yml"
COMPOSE_JETSON="docker-compose.jetson.yml"

# ── Detect platform ──────────────────────────
detect_platform() {
    local arch
    arch=$(uname -m)
    if [[ "$arch" == "aarch64" ]]; then
        # Check for Jetson by reading the board model
        if grep -qi "jetson\|tegra" /proc/device-tree/model 2>/dev/null; then
            echo "jetson"
        else
            echo "jetson"   # any ARM64 — use Jetson image
        fi
    else
        echo "x86"
    fi
}

# ── Helpers ──────────────────────────────────
print_banner() {
    echo ""
    echo "  ╔══════════════════════════════════╗"
    echo "  ║       Data Logger — $1"
    echo "  ╚══════════════════════════════════╝"
    echo ""
}

# ── Main ─────────────────────────────────────
PLATFORM="${1:-auto}"

if [[ "$PLATFORM" == "auto" ]]; then
    PLATFORM=$(detect_platform)
    echo "[run.sh] Auto-detected platform: $PLATFORM"
fi

case "$PLATFORM" in

    x86)
        print_banner "x86 / Ubuntu 22.04"
        COMPOSE_FILE="$COMPOSE_X86"
        ;;

    jetson)
        print_banner "Jetson Nano / JetPack 4.6"
        COMPOSE_FILE="$COMPOSE_JETSON"

        # Check nvidia-container-runtime
        if ! docker info 2>/dev/null | grep -q "nvidia"; then
            echo "[WARN] nvidia runtime not detected in Docker."
            echo "       Run: sudo apt install nvidia-container-runtime"
            echo "            sudo systemctl restart docker"
            echo ""
        fi
        ;;

    stop)
        echo "[run.sh] Stopping all data_logger containers..."
        docker compose -f "$COMPOSE_X86"    down 2>/dev/null || true
        docker compose -f "$COMPOSE_JETSON" down 2>/dev/null || true
        echo "[run.sh] Done."
        exit 0
        ;;

    logs)
        PLATFORM=$(detect_platform)
        if [[ "$PLATFORM" == "jetson" ]]; then
            docker compose -f "$COMPOSE_JETSON" logs -f
        else
            docker compose -f "$COMPOSE_X86" logs -f
        fi
        exit 0
        ;;

    *)
        echo "Usage: $0 [x86|jetson|stop|logs]"
        exit 1
        ;;
esac

# ── Build + start ────────────────────────────
# echo "[run.sh] Building image..."
# docker compose -f "$COMPOSE_FILE" build

echo "[run.sh] Starting container..."
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo "  ✔  API server running at http://localhost:5000"
echo "  ✔  UI:             http://localhost:5000"
echo "  ✔  Logs folder:    $(pwd)/logs"
echo ""
echo "  Stop:  ./run.sh stop"
echo "  Logs:  ./run.sh logs"
echo ""
