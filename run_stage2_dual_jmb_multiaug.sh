#!/usr/bin/env bash
set -euo pipefail

# Dual JMB multi-augmentation validation.
#
# Default: the prespecified single follow-up seed (seed0) after seed1 K=2
# produced only a weak improvement. Two independently augmented JMB groups
# are used per iteration. Each group uses one online Joint branch and EMA
# Motion/Bone branches derived from the same augmented raw exemplar. The two
# normalized JMB prototypes are averaged and normalized again.
#
# Usage:
#   bash run_stage2_dual_jmb_multiaug.sh
#   bash run_stage2_dual_jmb_multiaug.sh all
#   bash run_stage2_dual_jmb_multiaug.sh pretrain
#   bash run_stage2_dual_jmb_multiaug.sh lp
#
# Optional:
#   SEED=1 AUGMENTATIONS=2 PYTHON_BIN=/path/to/python \
#     bash run_stage2_dual_jmb_multiaug.sh all

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${1:-all}"
SEED="${SEED:-0}"
AUGMENTATIONS="${AUGMENTATIONS:-2}"

case "${RUN_MODE}" in
    all|pretrain|lp)
        ;;
    *)
        echo "Usage: bash $0 [all|pretrain|lp]" >&2
        exit 2
        ;;
esac

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer: ${SEED}" >&2
    exit 2
fi
if ! [[ "${AUGMENTATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "AUGMENTATIONS must be a positive integer: ${AUGMENTATIONS}" >&2
    exit 2
fi

STAGE1_CHECKPOINT="./data/ntu60_cs/aimclr_joint/pretext/epoch300_model.pt"
PRETRAIN_CONFIG="config/ntu60/pretext/pretext_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml"
LINEAR_CONFIG="config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml"
EXEMPLAR_CACHE="./data/ntu60_cs/ose_resa_shared/exemplar_seed${SEED}.npy"
OUTPUT_ROOT="./data/ntu60_cs/aimclr_to_native_ose_resa_a4_dualproj_jmb_q0_mf_ma${AUGMENTATIONS}_100ep_joint_seed${SEED}"
PRETRAIN_DIR="${OUTPUT_ROOT}/pretext"
CHECKPOINT="${PRETRAIN_DIR}/epoch100_model.pt"
LP_DIR="${OUTPUT_ROOT}/linear_eval"
LP_LOG="${LP_DIR}/log.txt"
SUMMARY="${OUTPUT_ROOT}/lp_best_acc.csv"

require_empty_work_dir() {
    local work_dir="$1"
    local label="$2"
    if [[ -d "${work_dir}" ]] &&
            [[ -n "$(find "${work_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to reuse non-empty partial ${label} work_dir: ${work_dir}" >&2
        echo "Inspect it and choose a new SEED/AUGMENTATIONS value before retrying." >&2
        exit 1
    fi
}

extract_best_acc() {
    local best_line
    best_line="$(grep -Eo 'Best Top1: [0-9]+([.][0-9]+)?%' "${LP_LOG}" |
        tail -n 1 || true)"
    if [[ -z "${best_line}" ]]; then
        echo "Could not find Best Top1 in ${LP_LOG}" >&2
        exit 1
    fi
    best_line="${best_line#Best Top1: }"
    echo "${best_line%\%}"
}

run_pretrain() {
    if [[ -f "${CHECKPOINT}" ]]; then
        echo "Stage2 epoch100 already exists; skip: ${CHECKPOINT}"
        return
    fi
    if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
        echo "Missing Stage1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
        exit 1
    fi
    echo "Running ReSA/OSE unit tests before the formal Stage2 run"
    "${PYTHON_BIN}" -m unittest \
        tests.test_ose_resa_lmix \
        tests.test_ose_resa_stage2 \
        tests.test_ose_resa_prototypes
    require_empty_work_dir "${PRETRAIN_DIR}" "pretrain"

    echo "Stage2 Dual JMB multi-augmentation | seed ${SEED} | K=${AUGMENTATIONS}"
    "${PYTHON_BIN}" main.py pretrain_ose_resa_stage2 \
        --config "${PRETRAIN_CONFIG}" \
        --work_dir "${PRETRAIN_DIR}" \
        --ose_exemplar_seed "${SEED}" \
        --ose_exemplar_index_path "${EXEMPLAR_CACHE}" \
        --ose_exemplar_views "${AUGMENTATIONS}"
}

run_lp() {
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Missing Stage2 epoch100 checkpoint: ${CHECKPOINT}" >&2
        exit 1
    fi

    if [[ -f "${LP_LOG}" ]] && grep -q 'Eval epoch: 200' "${LP_LOG}"; then
        echo "LP epoch200 already complete; reuse: ${LP_LOG}"
    else
        require_empty_work_dir "${LP_DIR}" "LP"
        "${PYTHON_BIN}" main.py linear_evaluation \
            --config "${LINEAR_CONFIG}" \
            --work_dir "${LP_DIR}" \
            --weights "${CHECKPOINT}"
        if ! grep -q 'Eval epoch: 200' "${LP_LOG}"; then
            echo "LP did not reach epoch200: ${LP_LOG}" >&2
            exit 1
        fi
    fi

    local best_acc
    best_acc="$(extract_best_acc)"
    mkdir -p "${OUTPUT_ROOT}"
    {
        echo "seed,augmentations,variant,best_acc,checkpoint,lp_log"
        printf '%s,%s,%s,%s,%s,%s\n' \
            "${SEED}" "${AUGMENTATIONS}" "dual-jmb-q0-multiaug" \
            "${best_acc}" "${CHECKPOINT}" "${LP_LOG}"
    } > "${SUMMARY}"
    echo "LP best | seed ${SEED} | K=${AUGMENTATIONS} | ${best_acc}%"
    echo "Summary: ${SUMMARY}"
}

if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "pretrain" ]]; then
    run_pretrain
fi
if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "lp" ]]; then
    run_lp
fi
