#!/usr/bin/env bash
set -euo pipefail

# Gate 1 paired multi-seed validation for the current AimCLR Stage2 method.
#
# The legacy filename is kept so existing server commands do not break. A5
# seed0 is already complete; this script now runs the six missing experiments:
#
#   exemplar seed 1: OSE-only -> Shared -> Dual
#   exemplar seed 2: OSE-only -> Shared -> Dual
#
# Every seed uses one shared exemplar cache across the three variants. Each
# variant has its own pretrain/LP work directory. After every LP, the script
# extracts the final "Best Top1" value from log.txt and appends it to:
#
#   ./data/ntu60_cs/aimclr_stage2_gate1_lp_best_acc.log
#
# Usage:
#   bash run_stage2_a5_ablation.sh          # six pretrains + six LP runs
#   bash run_stage2_a5_ablation.sh all      # same as above
#   bash run_stage2_a5_ablation.sh pretrain # only six Stage2 pretrains
#   bash run_stage2_a5_ablation.sh lp       # only six LP runs + summary log
#
# Optional overrides:
#   PYTHON_BIN=/path/to/python bash run_stage2_a5_ablation.sh all
#   SUMMARY_LOG=/path/to/result.log bash run_stage2_a5_ablation.sh lp
#
# Completed epoch100 pretrains and completed epoch200 LPs are reused safely.
# A non-empty partial work_dir is never overwritten automatically.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${1:-all}"
SUMMARY_LOG="${SUMMARY_LOG:-./data/ntu60_cs/aimclr_stage2_gate1_lp_best_acc.log}"

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

SEEDS=(1 2)

NAMES=(
    "ose-only-jmb-q0"
    "shared-jmb-q0"
    "dual-projector-jmb-q0"
)

PRETRAIN_CONFIGS=(
    "config/ntu60/pretext/pretext_ose_resa_a4_stage2_ose_only_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/pretext/pretext_ose_resa_a4_stage2_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/pretext/pretext_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml"
)

LINEAR_CONFIGS=(
    "config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_ose_only_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_jmb_q0_mf_xsub_joint.yaml"
    "config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml"
)

OUTPUT_STEMS=(
    "aimclr_to_native_ose_only_a4_jmb_q0_mf_100ep_joint"
    "aimclr_to_native_ose_resa_a4_jmb_q0_mf_100ep_joint"
    "aimclr_to_native_ose_resa_a4_dualproj_jmb_q0_mf_100ep_joint"
)

TOTAL_RUNS=$((${#SEEDS[@]} * ${#NAMES[@]}))

exemplar_cache() {
    local seed="$1"
    echo "./data/ntu60_cs/ose_resa_shared/exemplar_seed${seed}.npy"
}

output_root() {
    local index="$1"
    local seed="$2"
    echo "./data/ntu60_cs/${OUTPUT_STEMS[index]}_seed${seed}"
}

require_empty_work_dir() {
    local work_dir="$1"
    local label="$2"
    if [[ -d "${work_dir}" ]] &&
            [[ -n "$(find "${work_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to reuse non-empty ${label} work_dir: ${work_dir}" >&2
        echo "Inspect the existing run and choose a new directory before retrying." >&2
        exit 1
    fi
}

initialize_summary_log() {
    mkdir -p "$(dirname -- "${SUMMARY_LOG}")"
    {
        echo "# AimCLR Stage2 Gate 1 paired multi-seed LP results"
        echo "# generated_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "# Stage2 checkpoint is fixed at epoch100; best_acc is the final Best Top1 reported by the 200-epoch LP."
        echo "seed,variant,best_acc,checkpoint,lp_log"
    } > "${SUMMARY_LOG}"
}

extract_best_acc() {
    local lp_log="$1"
    local best_line
    local best_acc

    if [[ ! -f "${lp_log}" ]]; then
        echo "LP log was not created: ${lp_log}" >&2
        return 1
    fi

    best_line="$(grep -Eo 'Best Top1: [0-9]+([.][0-9]+)?%' "${lp_log}" |
        tail -n 1 || true)"
    if [[ -z "${best_line}" ]]; then
        echo "Could not find a Best Top1 entry in ${lp_log}" >&2
        return 1
    fi

    best_acc="${best_line#Best Top1: }"
    best_acc="${best_acc%\%}"
    echo "${best_acc}"
}

run_pretraining() {
    local completed=0
    local seed
    local index
    local cache
    local root
    local work_dir

    for seed in "${SEEDS[@]}"; do
        cache="$(exemplar_cache "${seed}")"
        for index in "${!NAMES[@]}"; do
            completed=$((completed + 1))
            root="$(output_root "${index}" "${seed}")"
            work_dir="${root}/pretext"
            if [[ -f "${work_dir}/epoch100_model.pt" ]]; then
                echo "[${completed}/${TOTAL_RUNS}] Stage2 pretrain already complete; skip | seed ${seed} | ${NAMES[index]}"
                continue
            fi
            require_empty_work_dir "${work_dir}" "pretrain"

            echo "[${completed}/${TOTAL_RUNS}] Stage2 pretrain | seed ${seed} | ${NAMES[index]}"
            "${PYTHON_BIN}" main.py pretrain_ose_resa_stage2 \
                --config "${PRETRAIN_CONFIGS[index]}" \
                --work_dir "${work_dir}" \
                --ose_exemplar_seed "${seed}" \
                --ose_exemplar_index_path "${cache}"
        done
    done
}

run_linear_evaluation() {
    local completed=0
    local seed
    local index
    local root
    local checkpoint
    local work_dir
    local lp_log
    local best_acc

    initialize_summary_log
    for seed in "${SEEDS[@]}"; do
        for index in "${!NAMES[@]}"; do
            completed=$((completed + 1))
            root="$(output_root "${index}" "${seed}")"
            checkpoint="${root}/pretext/epoch100_model.pt"
            work_dir="${root}/linear_eval"
            lp_log="${work_dir}/log.txt"

            if [[ ! -f "${checkpoint}" ]]; then
                echo "Missing epoch100 Stage2 checkpoint: ${checkpoint}" >&2
                exit 1
            fi
            if [[ -f "${lp_log}" ]] &&
                    grep -q 'Eval epoch: 200' "${lp_log}"; then
                best_acc="$(extract_best_acc "${lp_log}")"
                echo "[${completed}/${TOTAL_RUNS}] LP already complete; reuse | seed ${seed} | ${NAMES[index]} | ${best_acc}%"
            else
                require_empty_work_dir "${work_dir}" "LP"

                echo "[${completed}/${TOTAL_RUNS}] Linear evaluation | seed ${seed} | ${NAMES[index]}"
                "${PYTHON_BIN}" main.py linear_evaluation \
                    --config "${LINEAR_CONFIGS[index]}" \
                    --work_dir "${work_dir}" \
                    --weights "${checkpoint}"

                if ! grep -q 'Eval epoch: 200' "${lp_log}"; then
                    echo "LP did not reach its required epoch200 evaluation: ${lp_log}" >&2
                    exit 1
                fi
                best_acc="$(extract_best_acc "${lp_log}")"
            fi
            printf '%s,%s,%s,%s,%s\n' \
                "${seed}" "${NAMES[index]}" "${best_acc}" \
                "${checkpoint}" "${lp_log}" >> "${SUMMARY_LOG}"
            echo "LP best | seed ${seed} | ${NAMES[index]} | ${best_acc}%"
        done
    done

    echo "LP best-accuracy summary: ${SUMMARY_LOG}"
}

if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "pretrain" ]]; then
    run_pretraining
fi

if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "lp" ]]; then
    run_linear_evaluation
fi

echo "AimCLR Stage2 Gate 1 ${RUN_MODE} run completed successfully."
