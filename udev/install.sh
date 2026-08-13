#!/usr/bin/env bash
# Installs udev rules from this directory into /etc/udev/rules.d and reloads them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for rule in "$SCRIPT_DIR"/*.rules; do
    sudo cp "$rule" /etc/udev/rules.d/
    echo "Installed $(basename "$rule")"
done

sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Done. Check with: ls -la /dev/gnss_rtk"
