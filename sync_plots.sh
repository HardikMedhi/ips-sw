#!/usr/bin/env bash

# ==========================================
# Configuration
# ==========================================
REMOTE="amort"
REMOTE_DIR_DS="/home/hardikmedhi/PhD/plots/ds/"
REMOTE_DIR_PS="/home/hardikmedhi/PhD/plots/ps/"
LOCAL_DIR_DS="/data/PhD/thesis/plots/ds/"
LOCAL_DIR_PS="/data/PhD/thesis/plots/ps/"
LOG_FILE="/data/PhD/thesis/logs/sync_plots.log"

# ==========================================
# Network Delay
# ==========================================
# Wait 60 seconds to ensure the network interface is fully up and connected
# sleep 60

echo "----------------------------------------" >> "$LOG_FILE"
echo "Starting boot sync at $(date)" >> "$LOG_FILE"

# Ensure local directory exists
mkdir -p "$LOCAL_DIR_DS"
mkdir -p "$LOCAL_DIR_PS"

# Rsync the data (-u skips files that are newer on the receiver)
if rsync -avzu --progress "$REMOTE:$REMOTE_DIR_DS" "$LOCAL_DIR_DS" >> "$LOG_FILE" 2>&1; then
    echo "DS Sync complete at $(date)" >> "$LOG_FILE"
else
    echo "ERROR: DS Sync failed at $(date). Check connection to $REMOTE." >> "$LOG_FILE"
fi

if rsync -avzu --progress "$REMOTE:$REMOTE_DIR_PS" "$LOCAL_DIR_PS" >> "$LOG_FILE" 2>&1; then
    echo "PS Sync complete at $(date)" >> "$LOG_FILE"
else
    echo "ERROR: PS Sync failed at $(date). Check connection to $REMOTE." >> "$LOG_FILE"
fi
