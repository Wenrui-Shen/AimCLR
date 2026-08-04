#!/usr/bin/env bash
set -euo pipefail

# Run the four AimCLR Stage2 A5 ablations from the repository root.
#
# Usage:
#   bash run_stage2_a5_ablation.sh          # pretrain + LP for every variant
#   bash run_stage2_a5_ablation.sh all      # same as above
#   bash run_stage2_a5_ablation.sh pretrain # only the four Stage2 runs
#   bash run_stage2_a5_ablation.sh lp       # only the four linear evaluations
#
# Override the Python executable when needed:
#   PYTHON_BIN=/path/to/python bash run_stage2_a5_ablation.sh all

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${1:-all}"

case "${RUN_MODE}" in
    all|pretrain|lp)
        ;;
    *)
        echo "Usage: bash $0 [all|pretrain|lp]" >&2
        exit 2
        ;;
esac

if [[ ! -f main.py ]]; then
    echo "main.py was not found in ${SCRIPT_DIR}" >&2
    exit 1
fi

STAGE1_CHECKPOINT="./data/ntu60_cs/aimclr_joint/pretext/epoch300_model.pt"
if [[ "${RUN_MODE}" != "lp" && ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "Missing Stage1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
    exit 1
fi

NAMES=(
    "shared-jmb-q0"
    "resa-only"
    "ose-only-jmb-q0"
    "dual-projector-jmb-q0"
)

PRETRAIN_CONFIGS=(
    "config/ntu60/pretext/pretext_ose_resa_a4_stage2_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/pretext/pretext_ose_resa_a4_stage2_resa_only_xsub_joint.yaml"
    "config/ntu60/pretext/pretext_ose_resa_a4_stage2_ose_only_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/pretext/pretext_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml"
)

LINEAR_CONFIGS=(
    "config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_resa_only_xsub_joint.yaml"
    "config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_ose_only_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml"
)

run_pretraining() {
    local index
    for index in "${!NAMES[@]}"; do
        echo "[$((index + 1))/${#NAMES[@]}] Stage2 pretrain: ${NAMES[index]}"
        "${PYTHON_BIN}" main.py pretrain_ose_resa_stage2 \
            --config "${PRETRAIN_CONFIGS[index]}"
    done
}

run_linear_evaluation() {
    local index
    for index in "${!NAMES[@]}"; do
        echo "[$((index + 1))/${#NAMES[@]}] Linear evaluation: ${NAMES[index]}"
        "${PYTHON_BIN}" main.py linear_evaluation \
            --config "${LINEAR_CONFIGS[index]}"
    done
}

if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "pretrain" ]]; then
    run_pretraining
fi

if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "lp" ]]; then
    run_linear_evaluation
fi

echo "A5 Stage2 ${RUN_MODE} run completed successfully."
