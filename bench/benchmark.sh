#!/bin/bash
# =============================================================================
# PHIRST GPU Performance Benchmark Script
# =============================================================================
# Runs existing ex_drell_yan_<backend> executables (no builds performed).
#
# Usage: ./benchmark.sh [num_events] [seed] [num_runs]
#        ./benchmark.sh             # Default: 10M events, seed 5489, 3 runs
#        ./benchmark.sh 1000000 5489 3
#
# Notes:
#  - This script does NOT build the project. It searches for executables named
#    ex_drell_yan_<backend> (lowercase backend suffix) in:
#      * directories in $PATH
#      * ./build-<backend>/ (relative to repo root)
#      * the repository tree (depth 4)
#
# Options:
#   --help, -h      Show this help message
#
# Examples (to build manually):
#   cmake -DPHIRST_BACKEND=CUDA  -S . -B build-cuda && cmake --build build-cuda
#   cmake -DPHIRST_BACKEND=KOKKOS -S . -B build-kokkos -DKokkos_ROOT=/path/to/kokkos
#   cmake -DPHIRST_BACKEND=SYCL  -S . -B build-sycl -DCMAKE_CXX_COMPILER=/path/to/dpcpp
# =============================================================================

# Don't exit on individual command failures
set +e

# Parse options
# By default perform builds unless --skip-build is provided
SKIP_BUILD=false
FORCE_BUILD=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --help|-h)
            head -25 "$0" | tail -23
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_EVENTS=${POSITIONAL_ARGS[0]:-10000000}
SEED=${POSITIONAL_ARGS[1]:-5489}
NUM_RUNS=${POSITIONAL_ARGS[2]:-3}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}PHIRST GPU Performance Benchmark${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""
echo -e "Events per run: ${YELLOW}${NUM_EVENTS}${NC}"
echo -e "Random seed: ${YELLOW}${SEED}${NC}"
echo -e "Number of runs: ${YELLOW}${NUM_RUNS}${NC}"
echo ""

# Build helper removed — this script does not perform builds. Build the project manually if needed.

# Function to run benchmark and extract throughput
run_benchmark() {
    local name=$1
    local executable=$2
    local runs=$3
    
    if [ ! -x "$executable" ]; then
        echo -e "${RED}Error: Executable $executable not found${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Running ${name} (${runs} runs)...${NC}"
    
    local total_throughput=0
    local min_throughput=999999999999
    local max_throughput=0
    local throughputs=()
    
    for ((i=1; i<=runs; i++)); do
        # Run and capture output
        output=$("$executable" "$NUM_EVENTS" "$SEED" 2>&1)
        
        # Extract throughput (events/sec)
        throughput=$(echo "$output" | grep "Throughput:" | tail -1 | awk '{print $2}')
        
        if [ -n "$throughput" ]; then
            throughputs+=($throughput)
            
            # Convert scientific notation to decimal for comparison
            throughput_dec=$(echo "$throughput" | awk '{printf "%.0f", $1}')
            total_throughput=$((total_throughput + throughput_dec))
            
            if [ "$throughput_dec" -lt "$min_throughput" ]; then
                min_throughput=$throughput_dec
            fi
            if [ "$throughput_dec" -gt "$max_throughput" ]; then
                max_throughput=$throughput_dec
            fi
            
            echo -e "  Run $i: ${throughput} events/sec"
        else
            echo -e "  ${RED}Run $i: Failed to extract throughput${NC}"
        fi
    done
    
    if [ ${#throughputs[@]} -gt 0 ]; then
        avg_throughput=$((total_throughput / ${#throughputs[@]}))
        echo -e "${GREEN}  Average: $(printf "%.2e" $avg_throughput) events/sec${NC}"
        echo -e "  Min: $(printf "%.2e" $min_throughput), Max: $(printf "%.2e" $max_throughput)"
        
        # Store result for summary
        eval "${name}_avg=$avg_throughput"
        eval "${name}_min=$min_throughput"
        eval "${name}_max=$max_throughput"
    fi
    
    echo ""
}

# Function to verify GPU utilization
verify_gpu() {
    local name=$1
    local executable=$2
    
    echo -e "${BLUE}Verifying GPU utilization for ${name}...${NC}"
    
    if [ -x "$SCRIPT_DIR/check_gpu.sh" ]; then
        "$SCRIPT_DIR/check_gpu.sh" "$executable" "$NUM_EVENTS" "$SEED" 2>&1 | grep -a -E "(GPU WAS|GPU was NOT)" || true
    else
        echo -e "${YELLOW}Warning: check_gpu.sh not found${NC}"
    fi
    echo ""
}

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Phase 1: Locating executables${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${YELLOW}Searching for ex_drell_yan_<backend> executables in PATH and the repository tree (depth 4)...${NC}"
echo ""

# Initialize flags and exe paths
SERIAL_OK=false
CUDA_OK=false
KOKKOS_OK=false
SYCL_OK=false
ALPAKA_OK=false
SERIAL_EXE=""
CUDA_EXE=""
KOKKOS_EXE=""
SYCL_EXE=""
ALPAKA_EXE=""

locate_executable() {
    local backend="$1"
    local backend_lc
    backend_lc=$(echo "$backend" | tr '[:upper:]' '[:lower:]')
    local name="ex_drell_yan_${backend_lc}"

    # Check PATH first
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi

    # Search project tree (limit depth)
    found=$(find "$SCRIPT_DIR/.." -maxdepth 4 -type f -name "$name" -perm /111 -print -quit 2>/dev/null || true)
    if [ -n "$found" ]; then
        printf '%s\n' "$found"
        return 0
    fi

    # Fallback: search PATH directories explicitly
    IFS=':'; for d in $PATH; do
        if [ -x "$d/$name" ]; then
            printf '%s\n' "$d/$name"
            return 0
        fi
    done
    return 1
}

BACKENDS=(SERIAL KOKKOS ALPAKA CUDA SYCL)
for b in "${BACKENDS[@]}"; do
    exe=$(locate_executable "$b")
    if [ -n "$exe" ]; then
        eval "${b}_OK=true"
        eval "${b}_EXE=\"$exe\""
        echo -e "${GREEN}✓ $b executable found: $exe${NC}"
    else
        eval "${b}_OK=false"
        echo -e "${RED}✗ $b executable not found${NC}"
    fi
done

echo ""
# Summarize detection results
found_entries=()
missing_entries=()
for b in "${BACKENDS[@]}"; do
    okvar="${b}_OK"
    exevar="${b}_EXE"
    if eval "[ \"\${${okvar}}\" = true ]"; then
        found_entries+=( "$b: $(eval echo \${${exevar}})" )
    else
        missing_entries+=( "$b" )
    fi
done

echo -e "${CYAN}Detection summary:${NC}"
if [ ${#found_entries[@]} -gt 0 ]; then
    echo -e "${GREEN}Found executables:${NC}"
    for e in "${found_entries[@]}"; do
        echo -e "  ${GREEN}${e}${NC}"
    done
else
    echo -e "${YELLOW}No ex_drell_yan_<backend> executables found.${NC}"
    echo -e "${RED}Please compile at least one backend and place the executable in PATH or in ./build-<backend>/.${NC}"
    echo -e "Quick examples to build (run from repo root):"
    echo -e "  cmake -DPHIRST_BACKEND=CUDA -S . -B build-cuda && cmake --build build-cuda"
    echo -e "  cmake -DPHIRST_BACKEND=KOKKOS -S . -B build-kokkos -DKokkos_ROOT=/path/to/kokkos && cmake --build build-kokkos"
    echo -e "  cmake -DPHIRST_BACKEND=SYCL -S . -B build-sycl -DCMAKE_CXX_COMPILER=/path/to/dpcpp && cmake --build build-sycl"
fi

if [ ${#missing_entries[@]} -gt 0 ]; then
    echo -e "${CYAN}Other backends not found: ${YELLOW}$(IFS=', '; echo "${missing_entries[*]}")${NC}"
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Phase 2: Performance Benchmarks${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Run benchmarks
SERIAL_avg=0
CUDA_avg=0
KOKKOS_avg=0
SYCL_avg=0
ALPAKA_avg=0

if $SERIAL_OK; then
    run_benchmark "SERIAL" "$SERIAL_EXE" "$NUM_RUNS"
fi

if $CUDA_OK; then
    run_benchmark "CUDA" "$CUDA_EXE" "$NUM_RUNS"
fi

if $KOKKOS_OK; then
    run_benchmark "KOKKOS" "$KOKKOS_EXE" "$NUM_RUNS"
fi

if $SYCL_OK; then
    run_benchmark "SYCL" "$SYCL_EXE" "$NUM_RUNS"
fi

if $ALPAKA_OK; then
    run_benchmark "ALPAKA" "$ALPAKA_EXE" "$NUM_RUNS"
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Summary${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Print summary table
printf "%-12s %15s %15s %15s\n" "Backend" "Avg (ev/s)" "Min" "Max"
printf "%-12s %15s %15s %15s\n" "--------" "----------" "---" "---"

if $SERIAL_OK && [ "$SERIAL_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "Serial" $SERIAL_avg $SERIAL_min $SERIAL_max
fi

if $KOKKOS_OK && [ "$KOKKOS_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "Kokkos" $KOKKOS_avg $KOKKOS_min $KOKKOS_max
fi

if $ALPAKA_OK && [ "$ALPAKA_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "Alpaka" $ALPAKA_avg $ALPAKA_min $ALPAKA_max
fi

if $CUDA_OK && [ "$CUDA_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "CUDA" $CUDA_avg $CUDA_min $CUDA_max
fi

if $SYCL_OK && [ "$SYCL_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "SYCL" $SYCL_avg $SYCL_min $SYCL_max
fi

echo ""

# Determine winner
max_avg=0
winner="None"

if [ "$SERIAL_avg" -gt "$max_avg" ]; then
    max_avg=$SERIAL_avg
    winner="Serial"
fi

if [ "$KOKKOS_avg" -gt "$max_avg" ]; then
    max_avg=$KOKKOS_avg
    winner="Kokkos"
fi

if [ "$ALPAKA_avg" -gt "$max_avg" ]; then
    max_avg=$ALPAKA_avg
    winner="Alpaka"
fi

if [ "$CUDA_avg" -gt "$max_avg" ]; then
    max_avg=$CUDA_avg
    winner="CUDA"
fi

if [ "$SYCL_avg" -gt "$max_avg" ]; then
    max_avg=$SYCL_avg
    winner="SYCL"
fi

echo -e "${GREEN}Fastest backend: ${winner} ($(printf "%.2e" $max_avg) events/sec)${NC}"
echo ""

# Calculate relative performance
if [ "$max_avg" -gt 0 ]; then
    echo "Relative performance:"
    if $SERIAL_OK && [ "$SERIAL_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $SERIAL_avg * 100 / $max_avg" | bc)
        printf "  Serial: %5s%%\n" "$rel"
    fi
    if $KOKKOS_OK && [ "$KOKKOS_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $KOKKOS_avg * 100 / $max_avg" | bc)
        printf "  Kokkos: %5s%%\n" "$rel"
    fi
    if $ALPAKA_OK && [ "$ALPAKA_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $ALPAKA_avg * 100 / $max_avg" | bc)
        printf "  Alpaka: %5s%%\n" "$rel"
    fi
    if $CUDA_OK && [ "$CUDA_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $CUDA_avg * 100 / $max_avg" | bc)
        printf "  CUDA:   %5s%%\n" "$rel"
    fi
    if $SYCL_OK && [ "$SYCL_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $SYCL_avg * 100 / $max_avg" | bc)
        printf "  SYCL:   %5s%%\n" "$rel"
    fi  
fi

echo ""
echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Benchmark complete.${NC}"
echo -e "${CYAN}======================================${NC}"
