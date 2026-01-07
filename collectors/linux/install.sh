#!/bin/bash
# DuckHunt v2.0 - Linux Installation Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

INSTALL_DIR="/usr/local/lib/duckhunt"
CONFIG_DIR="/etc/duckhunt"
LOG_DIR="/var/log/duckhunt"
DATA_DIR="/var/lib/duckhunt"

echo -e "${GREEN}DuckHunt v2.0 - Linux Installation${NC}"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo)${NC}"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

echo "Installing Python dependencies..."
pip3 install evdev numpy scipy || {
    echo -e "${YELLOW}Warning: Failed to install some Python packages${NC}"
}

# Create directories
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"

# Copy files
echo "Copying files..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cp -r "$SCRIPT_DIR/core" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/collectors" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/enforcement" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/config"/* "$CONFIG_DIR/"

# Set permissions
chmod +x "$INSTALL_DIR/collectors/linux/evdev_reader.py"
chmod 755 "$LOG_DIR"
chmod 755 "$DATA_DIR"

# Install systemd service
echo "Installing systemd service..."
cp "$INSTALL_DIR/collectors/linux/systemd/duckhunt.service" /etc/systemd/system/
systemctl daemon-reload

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review configuration: $CONFIG_DIR/duckhunt.v2.conf"
echo "  2. Start service: sudo systemctl start duckhunt"
echo "  3. Enable on boot: sudo systemctl enable duckhunt"
echo "  4. Check status: sudo systemctl status duckhunt"
echo "  5. View logs: sudo journalctl -u duckhunt -f"
echo ""
echo "The system will run in learning mode for the first 2 weeks"
echo "Monitor false positives and adjust configuration as needed"
