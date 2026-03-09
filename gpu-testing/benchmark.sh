#!/bin/bash
# =============================================================================
# RAMBO GPU Performance Benchmark Script
# =============================================================================
# Compares performance of Base, Kokkos, Alpaka, CUDA, and SYCL implementations
#
# Usage: ./benchmark.sh [options] [num_events] [seed] [num_runs]
#        ./benchmark.sh                    # Default: 10M events, seed 5489, 3 runs
#        ./benchmark.sh 100000000          # 100M events
#        ./benchmark.sh 10000000 42        # Custom seed
#        ./benchmark.sh 10000000 5489 5    # 5 runs
#        ./benchmark.sh --skip-build       # Skip build phase, run existing executables
#
# Options:
#   --skip-build    Skip the build phase, only run benchmarks on existing executables
#   --help          Show this help message
#
# Environment Variables (for build phase):
#   KOKKOS_ROOT   - Path to Kokkos installation
#   ALPAKA_ROOT   - Path to Alpaka installation  
#   SYCL_CXX      - Path to SYCL compiler (clang++)
#
# Example:
#   # Build each project manually with correct modules, then benchmark:
#   ./benchmark.sh --skip-build 10000000 5489 5
# =============================================================================

# Don't exit on individual command failures
set +e

# Parse options
SKIP_BUILD=true
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            SKIP_BUILD=false
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
echo -e "${CYAN}RAMBO GPU Performance Benchmark${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""
echo -e "Events per run: ${YELLOW}${NUM_EVENTS}${NC}"
echo -e "Random seed: ${YELLOW}${SEED}${NC}"
echo -e "Number of runs: ${YELLOW}${NUM_RUNS}${NC}"
echo ""

# Function to build a project
build_project() {
    local name=$1
    local dir=$2
    local cmake_args=${3:-""}
    
    echo -e "${BLUE}Building ${name}...${NC}"
    
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: Directory $dir not found${NC}"
        return 1
    fi
    
    cd "$dir"
    rm -rf build
    mkdir -p build
    cd build
    
    if cmake $cmake_args .. > /dev/null 2>&1; then
        if make -j4 > /dev/null 2>&1; then
            echo -e "${GREEN}✓ ${name} built successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ ${name} build failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ ${name} cmake failed${NC}"
        return 1
    fi
}

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
echo -e "${CYAN}Phase 1: Building Projects${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Build all projects
BASE_OK=false
KOKKOS_OK=false
ALPAKA_OK=false
CUDA_OK=false
SYCL_OK=false

# Unified "one" implementations
ONE_SERIAL_OK=false
ONE_CUDA_OK=false
ONE_KOKKOS_OK=false
ONE_SYCL_OK=false
ONE_ALPAKA_OK=false

if $SKIP_BUILD; then
    echo -e "${YELLOW}Skipping build phase (--build to build)${NC}"
    echo -e "Checking for existing executables..."
    echo ""
    
    # Check for existing executables
    if [ -x "$SCRIPT_DIR/base/build/rambo_base" ]; then
        echo -e "${GREEN}✓ Base executable found${NC}"
        BASE_OK=true
    else
        echo -e "${RED}✗ Base executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/kokkos/build/rambo_kokkos" ]; then
        echo -e "${GREEN}✓ Kokkos executable found${NC}"
        KOKKOS_OK=true
    else
        echo -e "${RED}✗ Kokkos executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/alpaka/build/rambo_alpaka" ]; then
        echo -e "${GREEN}✓ Alpaka executable found${NC}"
        ALPAKA_OK=true
    else
        echo -e "${RED}✗ Alpaka executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/cuda/build/rambo_cuda" ]; then
        echo -e "${GREEN}✓ CUDA executable found${NC}"
        CUDA_OK=true
    else
        echo -e "${RED}✗ CUDA executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/sycl/build/rambo_sycl" ]; then
        echo -e "${GREEN}✓ SYCL executable found${NC}"
        SYCL_OK=true
    else
        echo -e "${RED}✗ SYCL executable not found${NC}"
    fi
    
    # Check unified "one" executables
    echo ""
    echo -e "${YELLOW}Unified implementations:${NC}"
    
    if [ -x "$SCRIPT_DIR/one/build-serial/rambo_one" ]; then
        echo -e "${GREEN}✓ One-Serial executable found${NC}"
        ONE_SERIAL_OK=true
    else
        echo -e "${RED}✗ One-Serial executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/one/build-cuda/rambo_one" ]; then
        echo -e "${GREEN}✓ One-CUDA executable found${NC}"
        ONE_CUDA_OK=true
    else
        echo -e "${RED}✗ One-CUDA executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/one/build-kokkos/rambo_one" ]; then
        echo -e "${GREEN}✓ One-Kokkos executable found${NC}"
        ONE_KOKKOS_OK=true
    else
        echo -e "${RED}✗ One-Kokkos executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/one/build-sycl/rambo_one" ]; then
        echo -e "${GREEN}✓ One-SYCL executable found${NC}"
        ONE_SYCL_OK=true
    else
        echo -e "${RED}✗ One-SYCL executable not found${NC}"
    fi
    
    if [ -x "$SCRIPT_DIR/one/build-alpaka/rambo_one" ]; then
        echo -e "${GREEN}✓ One-Alpaka executable found${NC}"
        ONE_ALPAKA_OK=true
    else
        echo -e "${RED}✗ One-Alpaka executable not found${NC}"
    fi
else
    # Base always builds (no dependencies)
    if build_project "Base (Serial)" "$SCRIPT_DIR/base"; then
        BASE_OK=true
    fi

    # Kokkos - use KOKKOS_ROOT if set, otherwise try to find it
    KOKKOS_CMAKE_ARGS=""
    if [ -n "$KOKKOS_ROOT" ]; then
        KOKKOS_CMAKE_ARGS="-DKokkos_ROOT=$KOKKOS_ROOT"
    fi
    if build_project "Kokkos" "$SCRIPT_DIR/kokkos" "$KOKKOS_CMAKE_ARGS"; then
        KOKKOS_OK=true
    fi

    # Alpaka - use ALPAKA_ROOT if set (alias alpaka_ROOT for CMake)
    ALPAKA_CMAKE_ARGS="-DALPAKA_BACKEND=CUDA"
    if [ -n "$ALPAKA_ROOT" ]; then
        ALPAKA_CMAKE_ARGS="$ALPAKA_CMAKE_ARGS -Dalpaka_ROOT=$ALPAKA_ROOT"
    fi
    if build_project "Alpaka (CUDA)" "$SCRIPT_DIR/alpaka" "$ALPAKA_CMAKE_ARGS"; then
        ALPAKA_OK=true
    fi

    # CUDA - auto-detects nvcc
    if build_project "CUDA" "$SCRIPT_DIR/cuda"; then
        CUDA_OK=true
    fi

    # SYCL - use SYCL_CXX if set
    SYCL_CMAKE_ARGS=""
    if [ -n "$SYCL_CXX" ]; then
        SYCL_CMAKE_ARGS="-DCMAKE_CXX_COMPILER=$SYCL_CXX"
    fi
    if build_project "SYCL" "$SCRIPT_DIR/sycl" "$SYCL_CMAKE_ARGS"; then
        SYCL_OK=true
    fi
fi  # End of build phase

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Phase 2: GPU Verification${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if $KOKKOS_OK; then
    verify_gpu "Kokkos" "$SCRIPT_DIR/kokkos/build/rambo_kokkos"
fi

if $ALPAKA_OK; then
    verify_gpu "Alpaka" "$SCRIPT_DIR/alpaka/build/rambo_alpaka"
fi

if $CUDA_OK; then
    verify_gpu "CUDA" "$SCRIPT_DIR/cuda/build/rambo_cuda"
fi

if $SYCL_OK; then
    verify_gpu "SYCL" "$SCRIPT_DIR/sycl/build/rambo_sycl"
fi

# Unified "one" GPU verification
if $ONE_CUDA_OK; then
    verify_gpu "One-CUDA" "$SCRIPT_DIR/one/build-cuda/rambo_one"
fi

if $ONE_KOKKOS_OK; then
    verify_gpu "One-Kokkos" "$SCRIPT_DIR/one/build-kokkos/rambo_one"
fi

if $ONE_SYCL_OK; then
    verify_gpu "One-SYCL" "$SCRIPT_DIR/one/build-sycl/rambo_one"
fi

if $ONE_ALPAKA_OK; then
    verify_gpu "One-Alpaka" "$SCRIPT_DIR/one/build-alpaka/rambo_one"
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Phase 3: Performance Benchmarks${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Run benchmarks
BASE_avg=0
KOKKOS_avg=0
ALPAKA_avg=0
CUDA_avg=0
SYCL_avg=0
ONE_SERIAL_avg=0
ONE_CUDA_avg=0
ONE_KOKKOS_avg=0
ONE_SYCL_avg=0
ONE_ALPAKA_avg=0

if $BASE_OK; then
    run_benchmark "BASE" "$SCRIPT_DIR/base/build/rambo_base" "$NUM_RUNS"
fi

if $KOKKOS_OK; then
    run_benchmark "KOKKOS" "$SCRIPT_DIR/kokkos/build/rambo_kokkos" "$NUM_RUNS"
fi

if $ALPAKA_OK; then
    run_benchmark "ALPAKA" "$SCRIPT_DIR/alpaka/build/rambo_alpaka" "$NUM_RUNS"
fi

if $CUDA_OK; then
    run_benchmark "CUDA" "$SCRIPT_DIR/cuda/build/rambo_cuda" "$NUM_RUNS"
fi

if $SYCL_OK; then
    run_benchmark "SYCL" "$SCRIPT_DIR/sycl/build/rambo_sycl" "$NUM_RUNS"
fi

# Unified "one" benchmarks
echo ""
echo -e "${YELLOW}--- Unified Implementations ---${NC}"
echo ""

if $ONE_SERIAL_OK; then
    run_benchmark "ONE_SERIAL" "$SCRIPT_DIR/one/build-serial/rambo_one" "$NUM_RUNS"
fi

if $ONE_CUDA_OK; then
    run_benchmark "ONE_CUDA" "$SCRIPT_DIR/one/build-cuda/rambo_one" "$NUM_RUNS"
fi

if $ONE_KOKKOS_OK; then
    run_benchmark "ONE_KOKKOS" "$SCRIPT_DIR/one/build-kokkos/rambo_one" "$NUM_RUNS"
fi

if $ONE_SYCL_OK; then
    run_benchmark "ONE_SYCL" "$SCRIPT_DIR/one/build-sycl/rambo_one" "$NUM_RUNS"
fi

if $ONE_ALPAKA_OK; then
    run_benchmark "ONE_ALPAKA" "$SCRIPT_DIR/one/build-alpaka/rambo_one" "$NUM_RUNS"
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Summary${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Print summary table
printf "%-12s %15s %15s %15s\n" "Backend" "Avg (ev/s)" "Min" "Max"
printf "%-12s %15s %15s %15s\n" "--------" "----------" "---" "---"

if $BASE_OK && [ "$BASE_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "Base" $BASE_avg $BASE_min $BASE_max
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
echo "Unified implementations:"
printf "%-12s %15s %15s %15s\n" "--------" "----------" "---" "---"

if $ONE_SERIAL_OK && [ "$ONE_SERIAL_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "One-Serial" $ONE_SERIAL_avg $ONE_SERIAL_min $ONE_SERIAL_max
fi

if $ONE_CUDA_OK && [ "$ONE_CUDA_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "One-CUDA" $ONE_CUDA_avg $ONE_CUDA_min $ONE_CUDA_max
fi

if $ONE_KOKKOS_OK && [ "$ONE_KOKKOS_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "One-Kokkos" $ONE_KOKKOS_avg $ONE_KOKKOS_min $ONE_KOKKOS_max
fi

if $ONE_SYCL_OK && [ "$ONE_SYCL_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "One-SYCL" $ONE_SYCL_avg $ONE_SYCL_min $ONE_SYCL_max
fi

if $ONE_ALPAKA_OK && [ "$ONE_ALPAKA_avg" -gt 0 ]; then
    printf "%-12s %15.2e %15.2e %15.2e\n" "One-Alpaka" $ONE_ALPAKA_avg $ONE_ALPAKA_min $ONE_ALPAKA_max
fi

echo ""

# Determine winner
max_avg=0
winner="None"

if [ "$BASE_avg" -gt "$max_avg" ]; then
    max_avg=$BASE_avg
    winner="Base"
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

if [ "$ONE_SERIAL_avg" -gt "$max_avg" ]; then
    max_avg=$ONE_SERIAL_avg
    winner="One-Serial"
fi

if [ "$ONE_CUDA_avg" -gt "$max_avg" ]; then
    max_avg=$ONE_CUDA_avg
    winner="One-CUDA"
fi

if [ "$ONE_KOKKOS_avg" -gt "$max_avg" ]; then
    max_avg=$ONE_KOKKOS_avg
    winner="One-Kokkos"
fi

if [ "$ONE_SYCL_avg" -gt "$max_avg" ]; then
    max_avg=$ONE_SYCL_avg
    winner="One-SYCL"
fi

if [ "$ONE_ALPAKA_avg" -gt "$max_avg" ]; then
    max_avg=$ONE_ALPAKA_avg
    winner="One-Alpaka"
fi

echo -e "${GREEN}Fastest backend: ${winner} ($(printf "%.2e" $max_avg) events/sec)${NC}"
echo ""

# Calculate relative performance
if [ "$max_avg" -gt 0 ]; then
    echo "Relative performance:"
    if $BASE_OK && [ "$BASE_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $BASE_avg * 100 / $max_avg" | bc)
        printf "  Base:   %5.1f%%\n" $rel
    fi
    if $KOKKOS_OK && [ "$KOKKOS_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $KOKKOS_avg * 100 / $max_avg" | bc)
        printf "  Kokkos: %5.1f%%\n" $rel
    fi
    if $ALPAKA_OK && [ "$ALPAKA_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $ALPAKA_avg * 100 / $max_avg" | bc)
        printf "  Alpaka: %5.1f%%\n" $rel
    fi
    if $CUDA_OK && [ "$CUDA_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $CUDA_avg * 100 / $max_avg" | bc)
        printf "  CUDA:   %5.1f%%\n" $rel
    fi
    if $SYCL_OK && [ "$SYCL_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $SYCL_avg * 100 / $max_avg" | bc)
        printf "  SYCL:       %5.1f%%\n" $rel
    fi
    echo ""
    echo "Unified:"
    if $ONE_SERIAL_OK && [ "$ONE_SERIAL_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $ONE_SERIAL_avg * 100 / $max_avg" | bc)
        printf "  One-Serial: %5.1f%%\n" $rel
    fi
    if $ONE_CUDA_OK && [ "$ONE_CUDA_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $ONE_CUDA_avg * 100 / $max_avg" | bc)
        printf "  One-CUDA:   %5.1f%%\n" $rel
    fi
    if $ONE_KOKKOS_OK && [ "$ONE_KOKKOS_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $ONE_KOKKOS_avg * 100 / $max_avg" | bc)
        printf "  One-Kokkos: %5.1f%%\n" $rel
    fi
    if $ONE_SYCL_OK && [ "$ONE_SYCL_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $ONE_SYCL_avg * 100 / $max_avg" | bc)
        printf "  One-SYCL:   %5.1f%%\n" $rel
    fi
    if $ONE_ALPAKA_OK && [ "$ONE_ALPAKA_avg" -gt 0 ]; then
        rel=$(echo "scale=1; $ONE_ALPAKA_avg * 100 / $max_avg" | bc)
        printf "  One-Alpaka: %5.1f%%\n" $rel
    fi
fi

echo ""
echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Benchmark complete.${NC}"
echo -e "${CYAN}======================================${NC}"
