#!/usr/bin/env bash

# ==========================================
# Configuration
# ==========================================
REMOTE="amort"
REMOTE_DIR="/home/hardikmedhi/PhD/plots/ds/"
LOCAL_DIR="/data/PhD/thesis/plots/ds/"
LOG_FILE="/data/PhD/thesis/logs/sync_plots.log"

# ==========================================
# Network Delay
# ==========================================
# Wait 60 seconds to ensure the network interface is fully up and connected
# sleep 60

echo "----------------------------------------" >> "$LOG_FILE"
echo "Starting boot sync at $(date)" >> "$LOG_FILE"

# Ensure local directory exists
mkdir -p "$LOCAL_DIR"

# Rsync the data (-u skips files that are newer on the receiver)
if rsync -avzu --progress "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR" >> "$LOG_FILE" 2>&1; then
    echo "Sync complete at $(date)" >> "$LOG_FILE"
else
    echo "ERROR: Sync failed at $(date). Check connection to $REMOTE." >> "$LOG_FILE"
fi
