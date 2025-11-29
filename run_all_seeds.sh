#!/bin/bash

# Run train_all_models.sh for multiple seeds
# Usage: bash run_all_seeds.sh

# Note: We don't use 'set -e' here because we want to continue
# processing other seeds even if one fails

# ============================================================================
# Configuration
# ============================================================================
SEEDS=(1 2 3 4 5)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Log directory structure
LOG_DIR="logs"
SUMMARY_DIR="${LOG_DIR}/summaries"

# Create log directories
mkdir -p "${SUMMARY_DIR}"

# Master log file
MASTER_LOG="${SUMMARY_DIR}/all_seeds_summary_${TIMESTAMP}.log"

# ============================================================================
# Color definitions
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper functions
# ============================================================================
format_duration() {
    local seconds=$1
    echo "$((seconds / 60))m $((seconds % 60))s"
}

echo "================================================================================"
echo "Running Training for All Seeds"
echo "================================================================================"
echo "Seeds: ${SEEDS[@]}"
echo "Master log: ${MASTER_LOG}"
echo "Start time: $(date)"
echo "================================================================================"
echo ""

# Initialize master log
cat > "${MASTER_LOG}" <<EOF
===============================================================================
Multi-Seed Training Summary
===============================================================================
Start Time: $(date)
Seeds: ${SEEDS[@]}

===============================================================================
EOF

# Track overall statistics
TOTAL_START=$(date +%s)
SUCCESSFUL_SEEDS=0
FAILED_SEEDS=0

# Run training for each seed
for SEED in "${SEEDS[@]}"; do
    echo -e "${CYAN}================================================================================${NC}"
    echo -e "${CYAN}Running Training for Seed: ${SEED}${NC}"
    echo -e "${CYAN}================================================================================${NC}"
    echo ""
    
    # Record start time for this seed
    SEED_START=$(date +%s)
    SEED_START_TIME=$(date)
    
    # Run train_all_models.sh with this seed
    if bash train_all_models.sh "${SEED}" 2>&1 | tee -a "${MASTER_LOG}"; then
        # Training succeeded
        SEED_END=$(date +%s)
        SEED_DURATION=$((SEED_END - SEED_START))
        DURATION_STR=$(format_duration ${SEED_DURATION})
        SUCCESSFUL_SEEDS=$((SUCCESSFUL_SEEDS + 1))
        
        echo ""
        echo -e "${GREEN}✓ Seed ${SEED} training completed successfully${NC}"
        echo -e "${GREEN}  Duration: ${DURATION_STR}${NC}"
        
        # Log to master summary
        cat >> "${MASTER_LOG}" <<EOF

================================================================================
Seed: ${SEED}
================================================================================
Status: SUCCESS
Start: ${SEED_START_TIME}
Duration: ${DURATION_STR}

EOF
        
    else
        # Training failed
        SEED_END=$(date +%s)
        SEED_DURATION=$((SEED_END - SEED_START))
        DURATION_STR=$(format_duration ${SEED_DURATION})
        FAILED_SEEDS=$((FAILED_SEEDS + 1))
        
        echo ""
        echo -e "${RED}✗ Seed ${SEED} training FAILED${NC}"
        echo -e "${RED}  Duration: ${DURATION_STR}${NC}"
        
        # Log to master summary
        cat >> "${MASTER_LOG}" <<EOF

================================================================================
Seed: ${SEED}
================================================================================
Status: FAILED
Start: ${SEED_START_TIME}
Duration: ${DURATION_STR}
Error: See logs above for details

EOF
    fi
    
    echo ""
    echo -e "${BLUE}Progress: ${SUCCESSFUL_SEEDS} successful, ${FAILED_SEEDS} failed out of ${#SEEDS[@]} seeds${NC}"
    echo ""
done

# Final summary
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_DURATION_STR=$(format_duration ${TOTAL_DURATION})

cat >> "${MASTER_LOG}" <<EOF

===============================================================================
Final Summary
===============================================================================
End Time: $(date)
Total Duration: ${TOTAL_DURATION_STR}
Successful Seeds: ${SUCCESSFUL_SEEDS}/${#SEEDS[@]}
Failed Seeds: ${FAILED_SEEDS}/${#SEEDS[@]}

===============================================================================
EOF

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}All Seeds Training Complete${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${GREEN}Results:${NC}"
echo -e "  Successful: ${SUCCESSFUL_SEEDS}/${#SEEDS[@]}"
echo -e "  Failed: ${FAILED_SEEDS}/${#SEEDS[@]}"
echo -e "  Total Duration: ${TOTAL_DURATION_STR}"
echo ""
echo -e "${GREEN}Master log saved to: ${MASTER_LOG}${NC}"
echo ""
echo "Individual seed logs:"
for SEED in "${SEEDS[@]}"; do
    SEED_LOG=$(ls -t "${SUMMARY_DIR}/training_summary_seed${SEED}_"*.log 2>/dev/null | head -1 || echo "")
    if [ -n "$SEED_LOG" ]; then
        echo "  Seed ${SEED}: ${SEED_LOG}"
    fi
done
echo ""
echo -e "${BLUE}View master summary: cat ${MASTER_LOG}${NC}"
echo -e "${BLUE}View MLflow results: mlflow ui --backend-store-uri ./mlruns${NC}"
echo ""

# Exit with non-zero code if any seed failed
if [ ${FAILED_SEEDS} -gt 0 ]; then
    exit 1
else
    exit 0
fi
