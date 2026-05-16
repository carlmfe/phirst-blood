#!/bin/bash
# =============================================================================
# PHIRST GPU Performance Benchmark Script
# =============================================================================
# Runs existing ex_drell_yan_<backend> and ex_eggholder_<backend> executables (no builds performed).
#
# Usage: ./benchmark.sh [num_events] [seed] [num_runs] [--test all|drell-yan|eggholder] [--mode all|vegas|flat]
#                       [--check] [--sigma N] [--no-color]
#        ./benchmark.sh             # Default: 10M events, seed 5489, 3 runs, all tests, all modes
#        ./benchmark.sh 1000000 5489 3 --test eggholder --mode vegas
#        ./benchmark.sh --check --sigma 3   # Exit with error if backends differ by >3σ or >5x throughput gap
#
# Notes:
#  - This script does NOT build the project. It searches for executables named
#    ex_<test>_<backend> (lowercase backend suffix) in:
#      * directories in $PATH
#      * ./build-<backend>/ (relative to repo root)
#      * the repository tree (depth 4)
#
# Options:
#   --help, -h      Show this help message
#   --test <name>   Which test to run: 'all', 'drell-yan', or 'eggholder' (default: all)
#   --mode <name>   Which integration mode to run: 'all', 'vegas', or 'flat' (default: all)
#   --check         Exit with code 1 if any consistency check fails (for CI use)
#   --sigma N       Sigma threshold for physics result comparison (default: 5)
#   --no-color      Suppress ANSI color codes (useful for CI logs and artifacts)
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
TEST_NAME="all"
MODE_NAME="all"
CHECK_RESULTS=false  # --check: exit 1 if consistency checks fail
NSIGMA=5             # --sigma N: sigma threshold for physics result comparison
NO_COLOR=false       # --no-color: suppress ANSI escape codes

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
        --test)
            TEST_NAME="$2"
            shift 2
            ;;
        --mode)
            MODE_NAME="$2"
            shift 2
            ;;
        --check)
            CHECK_RESULTS=true
            shift
            ;;
        --no-color)
            NO_COLOR=true
            shift
            ;;
        --sigma)
            NSIGMA="$2"
            shift 2
            ;;
        --help|-h)
            head -33 "$0" | tail -31
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
if $NO_COLOR; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}PHIRST GPU Performance Benchmark${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""
echo -e "Events per run: ${YELLOW}${NUM_EVENTS}${NC}"
echo -e "Random seed: ${YELLOW}${SEED}${NC}"
echo -e "Number of runs: ${YELLOW}${NUM_RUNS}${NC}"
echo -e "Test suite: ${YELLOW}${TEST_NAME}${NC}"
echo -e "Integration mode: ${YELLOW}${MODE_NAME}${NC}"
echo ""

# Build helper removed — this script does not perform builds. Build the project manually if needed.

# Function to run benchmark and extract throughput
run_benchmark() {
    local backend_name=$1
    local executable=$2
    local runs=$3
    local test_label=$4
    local mode=$5

    if [ ! -x "$executable" ]; then
        return 1
    fi
    
    local mode_flag="1"
    local mode_str="VEGAS"
    if [ "$mode" == "flat" ]; then
        mode_flag="0"
        mode_str="Flat MC"
    fi

    echo -e "${BLUE}Running ${test_label} on ${backend_name} [${mode_str}] (${runs} runs)...${NC}"
    
    local total_throughput=0
    local min_throughput=999999999999
    local max_throughput=0
    local throughputs=()
    local last_mean=""
    local last_err=""
    
    for ((i=1; i<=runs; i++)); do
        # Run and capture output
        output=$("$executable" "$NUM_EVENTS" "$SEED" "$mode_flag" 2>&1)
        
        # Extract throughput and physics result
        throughput=$(echo "$output" | grep "Throughput:" | tail -1 | awk '{print $2}')
        mean_val=$(echo "$output" | grep "^  Mean:" | tail -1 | awk '{print $2}')
        err_val=$(echo "$output" | grep "^  Error:" | tail -1 | awk '{print $2}')
        
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
            
            result_str=""
            [ -n "$mean_val" ] && result_str=" | Mean: ${mean_val} ± ${err_val}"
            echo -e "  Run $i: ${throughput} events/sec${result_str}"
            [ -n "$mean_val" ] && last_mean="$mean_val" && last_err="$err_val"
        else
            echo -e "  ${RED}Run $i: Failed to extract throughput${NC}"
        fi
    done
    
    if [ ${#throughputs[@]} -gt 0 ]; then
        avg_throughput=$((total_throughput / ${#throughputs[@]}))
        echo -e "${GREEN}  Average: $(printf "%.2e" $avg_throughput) events/sec${NC}"
        echo -e "  Min: $(printf "%.2e" $min_throughput), Max: $(printf "%.2e" $max_throughput)"
        
        # Store result for summary
        local var_prefix=$(echo "${test_label}_${mode}_${backend_name}" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
        eval "${var_prefix}_AVG=$avg_throughput"
        eval "${var_prefix}_MIN=$min_throughput"
        eval "${var_prefix}_MAX=$max_throughput"
        [ -n "$last_mean" ] && eval "${var_prefix}_MEAN=$last_mean"
        [ -n "$last_err"  ] && eval "${var_prefix}_ERR=$last_err"
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

# Check cross-backend consistency in physics results and throughput.
# Compares all pairs of backends that produced results using pairwise sigma,
# and flags GPU backends that appear suspiciously slow (possible CPU fallback).
consistency_check() {
    local t=$1
    local m=$2
    local t_upper=$(echo "$t" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    local m_upper=$(echo "$m" | tr '[:lower:]' '[:upper:]')
    local mode_str="VEGAS"; [ "$m" == "flat" ] && mode_str="Flat MC"
    local fail=false
    local has_data=false

    echo -e "${CYAN}--- Consistency: ${t} [${mode_str}] ---${NC}"

    # Collect backends that produced a physics result (mean ± error)
    local backends_with_mean=()
    for b in "${BACKENDS[@]}"; do
        meanval=$(eval echo "\${${t_upper}_${m_upper}_${b}_MEAN}")
        [ -n "$meanval" ] && backends_with_mean+=("$b")
    done

    if [ ${#backends_with_mean[@]} -ge 2 ]; then
        has_data=true
        echo "Physics results (mean ± error):"
        for b in "${backends_with_mean[@]}"; do
            printf "  %-8s: %s ± %s\n" \
                "$b" \
                "$(eval echo "\${${t_upper}_${m_upper}_${b}_MEAN}")" \
                "$(eval echo "\${${t_upper}_${m_upper}_${b}_ERR}")"
        done
        echo ""

        echo "Pairwise sigma (|ΔMean| / √(err₁²+err₂²), threshold: ${NSIGMA}σ):"
        for ((i=0; i<${#backends_with_mean[@]}; i++)); do
            for ((j=i+1; j<${#backends_with_mean[@]}; j++)); do
                local ba="${backends_with_mean[$i]}"
                local bb="${backends_with_mean[$j]}"
                local ma=$(eval echo "\${${t_upper}_${m_upper}_${ba}_MEAN}")
                local ea=$(eval echo "\${${t_upper}_${m_upper}_${ba}_ERR}")
                local mb=$(eval echo "\${${t_upper}_${m_upper}_${bb}_MEAN}")
                local eb=$(eval echo "\${${t_upper}_${m_upper}_${bb}_ERR}")
                sigma=$(awk -v m1="$ma" -v e1="$ea" -v m2="$mb" -v e2="$eb" 'BEGIN {
                    if (e1+0==0 || e2+0==0) { print "N/A"; exit }
                    diff = (m1>m2) ? m1-m2 : m2-m1
                    printf "%.1f", diff / sqrt(e1*e1 + e2*e2)
                }')
                if [ "$sigma" == "N/A" ]; then
                    echo -e "  ${YELLOW}$ba vs $bb: N/A (zero error)${NC}"
                elif awk -v s="$sigma" -v t="$NSIGMA" 'BEGIN{exit !(s+0>t+0)}'; then
                    echo -e "  ${RED}✗ $ba vs $bb: ${sigma}σ — exceeds ${NSIGMA}σ threshold${NC}"
                    fail=true
                else
                    echo -e "  ${GREEN}✓ $ba vs $bb: ${sigma}σ${NC}"
                fi
            done
        done
        echo ""
    fi

    # Throughput outlier check: flag non-SERIAL backends below 20% of the fastest GPU
    local max_gpu_tp=0
    for b in "${BACKENDS[@]}"; do
        [ "$b" == "SERIAL" ] && continue
        avg=$(eval echo "\${${t_upper}_${m_upper}_${b}_AVG:-0}")
        [ "${avg:-0}" -gt "$max_gpu_tp" ] 2>/dev/null && max_gpu_tp="${avg:-0}"
    done

    if [ "$max_gpu_tp" -gt 0 ]; then
        has_data=true
        echo "Throughput outlier check (GPU backends, threshold: 20% of fastest):"
        for b in "${BACKENDS[@]}"; do
            [ "$b" == "SERIAL" ] && continue
            [ "$(eval echo "\${${t_upper}_${m_upper}_${b}_OK}")" != "true" ] && continue
            avg=$(eval echo "\${${t_upper}_${m_upper}_${b}_AVG:-0}")
            [ "${avg:-0}" -eq 0 ] && continue
            pct=$(awk -v a="$avg" -v m="$max_gpu_tp" 'BEGIN{printf "%.0f", a*100/m}')
            if [ "$pct" -lt 20 ]; then
                echo -e "  ${RED}✗ $b: ${pct}% of fastest GPU — possible CPU fallback?${NC}"
                fail=true
            else
                echo -e "  ${GREEN}✓ $b: ${pct}%${NC}"
            fi
        done
        echo ""
    fi

    $has_data || echo -e "${YELLOW}  No data available for consistency check.${NC}"
    $fail && return 1
    return 0
}

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Phase 1: Locating executables${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

locate_executable() {
    local test_name="$1"
    local backend="$2"
    local backend_lc=$(echo "$backend" | tr '[:upper:]' '[:lower:]')
    local test_lc=$(echo "$test_name" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    local name="ex_${test_lc}_${backend_lc}"

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

BACKENDS=(SERIAL KOKKOS ALPAKA CUDA SYCL HIP)
TESTS_TO_RUN=()
MODES_TO_RUN=()

if [ "$TEST_NAME" == "all" ] || [ "$TEST_NAME" == "drell-yan" ]; then
    TESTS_TO_RUN+=("drell-yan")
fi
if [ "$TEST_NAME" == "all" ] || [ "$TEST_NAME" == "eggholder" ]; then
    TESTS_TO_RUN+=("eggholder")
fi

if [ "$MODE_NAME" == "all" ] || [ "$MODE_NAME" == "vegas" ]; then
    MODES_TO_RUN+=("vegas")
fi
if [ "$MODE_NAME" == "all" ] || [ "$MODE_NAME" == "flat" ]; then
    MODES_TO_RUN+=("flat")
fi

# Initialize summary variables
for t in "${TESTS_TO_RUN[@]}"; do
    t_upper=$(echo "$t" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    for m in "${MODES_TO_RUN[@]}"; do
        m_upper=$(echo "$m" | tr '[:lower:]' '[:upper:]')
        for b in "${BACKENDS[@]}"; do
            eval "${t_upper}_${m_upper}_${b}_OK=false"
            eval "${t_upper}_${m_upper}_${b}_EXE=\"\""
            eval "${t_upper}_${m_upper}_${b}_AVG=0"
            eval "${t_upper}_${m_upper}_${b}_MEAN=\"\""
            eval "${t_upper}_${m_upper}_${b}_ERR=\"\""
        done
    done
done

for t in "${TESTS_TO_RUN[@]}"; do
    echo -e "${YELLOW}Searching for ${t} executables...${NC}"
    t_upper=$(echo "$t" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    for b in "${BACKENDS[@]}"; do
        exe=$(locate_executable "$t" "$b")
        if [ -n "$exe" ]; then
            for m in "${MODES_TO_RUN[@]}"; do
                m_upper=$(echo "$m" | tr '[:lower:]' '[:upper:]')
                eval "${t_upper}_${m_upper}_${b}_OK=true"
                eval "${t_upper}_${m_upper}_${b}_EXE=\"$exe\""
            done
            echo -e "${GREEN}✓ $b: $exe${NC}"
        else
            echo -e "${RED}✗ $b not found${NC}"
        fi
    done
    echo ""
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Phase 2: Performance Benchmarks${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

for t in "${TESTS_TO_RUN[@]}"; do
    t_upper=$(echo "$t" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    for m in "${MODES_TO_RUN[@]}"; do
        m_upper=$(echo "$m" | tr '[:lower:]' '[:upper:]')

        mode_str="VEGAS"
        if [ "$m" == "flat" ]; then
            mode_str="Flat MC"
        fi

        echo -e "${CYAN}--- Benchmark: ${t} [${mode_str}] ---${NC}"
        for b in "${BACKENDS[@]}"; do
            okvar="${t_upper}_${m_upper}_${b}_OK"
            exevar="${t_upper}_${m_upper}_${b}_EXE"
            if eval "[ \"\${${okvar}}\" = true ]"; then
                run_benchmark "$b" "$(eval echo \${${exevar}})" "$NUM_RUNS" "$t" "$m"
            fi
        done
    done
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Summary${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

for t in "${TESTS_TO_RUN[@]}"; do
    t_upper=$(echo "$t" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    for m in "${MODES_TO_RUN[@]}"; do
        m_upper=$(echo "$m" | tr '[:lower:]' '[:upper:]')

        mode_str="VEGAS"
        if [ "$m" == "flat" ]; then
            mode_str="Flat MC"
        fi

        echo -e "${CYAN}Test: ${t} [${mode_str}]${NC}"

        # Check if any backends were run for this test/mode
        any_run=false
        for b in "${BACKENDS[@]}"; do
            avgvar="${t_upper}_${m_upper}_${b}_AVG"
            if [ "$(eval echo \${${avgvar}})" -gt 0 ]; then
                any_run=true
                break
            fi
        done

        if ! $any_run; then
            echo -e "${YELLOW}No benchmarks ran for this configuration.${NC}\n"
            continue
        fi

        printf "%-12s %15s %15s %15s\n" "Backend" "Avg (ev/s)" "Min" "Max"
        printf "%-12s %15s %15s %15s\n" "--------" "----------" "---" "---"

        for b in "${BACKENDS[@]}"; do
            okvar="${t_upper}_${m_upper}_${b}_OK"
            avgvar="${t_upper}_${m_upper}_${b}_AVG"
            minvar="${t_upper}_${m_upper}_${b}_MIN"
            maxvar="${t_upper}_${m_upper}_${b}_MAX"

            if eval "[ \"\${${okvar}}\" = true ]" && [ "$(eval echo \${${avgvar}})" -gt 0 ]; then
                printf "%-12s %15.2e %15.2e %15.2e\n" "$b" "$(eval echo \${${avgvar}})" "$(eval echo \${${minvar}})" "$(eval echo \${${maxvar}})"
            fi
        done
        echo ""

        # Determine winner for this test/mode
        max_avg=0
        winner="None"

        for b in "${BACKENDS[@]}"; do
            avgvar="${t_upper}_${m_upper}_${b}_AVG"
            avg_val=$(eval echo \${${avgvar}})
            if [ "$avg_val" -gt "$max_avg" ]; then
                max_avg=$avg_val
                winner="$b"
            fi
        done

        echo -e "${GREEN}Fastest backend: ${winner} ($(printf "%.2e" $max_avg) events/sec)${NC}"

        # Calculate relative performance
        if [ "$max_avg" -gt 0 ]; then
            echo "Relative performance:"
            for b in "${BACKENDS[@]}"; do
                okvar="${t_upper}_${m_upper}_${b}_OK"
                avgvar="${t_upper}_${m_upper}_${b}_AVG"
                avg_val=$(eval echo \${${avgvar}})
                if eval "[ \"\${${okvar}}\" = true ]" && [ "$avg_val" -gt 0 ]; then
                    rel=$(echo "scale=1; $avg_val * 100 / $max_avg" | bc)
                    printf "  %-8s: %5s%%\n" "$b" "$rel"
                fi
            done
        fi
        echo ""
    done
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Phase 3: Cross-backend Consistency${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "Physics sigma threshold: ${YELLOW}${NSIGMA}σ${NC} | Throughput outlier threshold: ${YELLOW}20%${NC} of fastest GPU"
echo ""

CONSISTENCY_FAILED=false
for t in "${TESTS_TO_RUN[@]}"; do
    for m in "${MODES_TO_RUN[@]}"; do
        consistency_check "$t" "$m" || CONSISTENCY_FAILED=true
    done
done

if $CONSISTENCY_FAILED; then
    echo -e "${YELLOW}⚠ Some consistency checks flagged issues — review warnings above.${NC}"
    $CHECK_RESULTS && { echo -e "${RED}Exiting with error (--check mode).${NC}"; exit 1; }
fi
echo ""

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Benchmark complete.${NC}"
echo -e "${CYAN}======================================${NC}"
