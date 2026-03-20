#!/bin/bash
# GPU Utilization Verification Script
# Usage: ./check_gpu.sh <executable> [args...]
#
# This script monitors GPU utilization while running a specified executable
# to verify that it is actually using the GPU.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <executable> [args...]"
    echo "Example: $0 ./rambo_alpaka 100000 5489"
    exit 1
fi

EXECUTABLE="$1"
shift
ARGS="$@"

if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: '$EXECUTABLE' is not executable or does not exist."
    exit 1
fi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. NVIDIA drivers may not be installed."
    exit 1
fi

echo "========================================"
echo "GPU Utilization Verification Script"
echo "========================================"
echo ""
echo "Initial GPU state:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Start nvidia-smi monitoring in background
MONITOR_LOG=$(mktemp)
(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader >> "$MONITOR_LOG" 2>/dev/null
        sleep 0.1
    done
) &
MONITOR_PID=$!

echo "Running: $EXECUTABLE $ARGS"
echo "----------------------------------------"
START_TIME=$(date +%s.%N)

# Run the executable
"$EXECUTABLE" $ARGS
EXIT_CODE=$?

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc)

# Stop monitoring
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "----------------------------------------"
echo "Exit code: $EXIT_CODE"
printf "Execution time: %.3f seconds\n" "$DURATION"
echo ""

# Analyze GPU usage
if [ -s "$MONITOR_LOG" ]; then
    echo "GPU Usage Summary during execution:"
    echo "========================================"
    
    # Get max GPU utilization observed
    MAX_UTIL=$(awk -F',' '{gsub(/[ %]/,"",$3); if($3+0 > max) max=$3+0} END {print max}' "$MONITOR_LOG")
    AVG_UTIL=$(awk -F',' '{gsub(/[ %]/,"",$3); sum+=$3+0; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}' "$MONITOR_LOG")
    SAMPLES=$(wc -l < "$MONITOR_LOG")
    
    echo "Samples collected: $SAMPLES"
    echo "Max GPU utilization: ${MAX_UTIL}%"
    echo "Avg GPU utilization: ${AVG_UTIL}%"
    echo ""
    
    if [ "${MAX_UTIL:-0}" -gt 5 ]; then
        echo "✓ GPU WAS UTILIZED (max utilization > 5%)"
    else
        echo "⚠ GPU may NOT have been utilized (max utilization <= 5%)"
        echo "  This could mean:"
        echo "  - The computation ran too fast to capture GPU usage"
        echo "  - The code is running on CPU instead of GPU"
        echo "  - Try running with more events for a longer execution"
    fi
else
    echo "Warning: Could not collect GPU monitoring data"
fi

# Cleanup
rm -f "$MONITOR_LOG"

echo ""
echo "Final GPU state:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

exit $EXIT_CODE
