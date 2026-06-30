#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Cluster helper after uv sync and transformers patch.

Default command:
  OPENPI_CACHE=/home/data/yangzemin/cache scripts/cluster_next_step.sh --no-check-certificate

Commands:
  verify        Check LIBERO files and offline LeRobotDataset loading.
  prepare-pi0   Download/convert pi0_base if needed.
  smoke         Verify LIBERO, prepare pi0_base, then run a 10-step smoke train. Default.
  train-flow    Run full flow fine-tuning.
  train-idp     Run full IDP-Geo fine-tuning.

Options:
  --openpi-cache DIR       Large cache root. Default: $OPENPI_CACHE or /home/data/yangzemin/cache.
  --gpu IDS                Set CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1.
  --no-check-certificate   Pass through to wget helper scripts.
  --ca-certificate FILE    Pass through to wget helper scripts.
  --jobs N                 Parallel wget jobs for helper scripts. Default: 8.
  --batch-size N           Training global batch size. Smoke default: 8; full default: 32.
  --num-workers N          DataLoader workers. Smoke default: 4; full default: 8.
  --steps N                Training steps. Smoke default: 10; full default: 30000.
  --exp-name NAME          Experiment name override.
  --project-name NAME      W&B project name. Default: openpi-idp.
  --wandb-mode MODE        W&B mode. Smoke default: disabled; full default: env/current.
  --overwrite              Pass --overwrite to train_pytorch.py.
  --resume                 Pass --resume to train_pytorch.py.
  -h, --help               Show this help.

Notes:
  This script unsets HF_ENDPOINT/HF_HUB_ENDPOINT because the Huawei HF endpoint has timed out in practice.
  It sets HF_HUB_OFFLINE=1 for training so the already-downloaded dataset is used.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command="smoke"
if [[ $# -gt 0 && "$1" != -* ]]; then
  command="$1"
  shift
fi

openpi_cache="${OPENPI_CACHE:-/home/data/yangzemin/cache}"
gpu="${CUDA_VISIBLE_DEVICES:-}"
no_check_certificate=0
ca_certificate=""
jobs=8
batch_size=""
num_workers=""
steps=""
exp_name=""
project_name="openpi-idp"
wandb_mode=""
overwrite=0
resume=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --openpi-cache)
      openpi_cache="$2"
      shift 2
      ;;
    --gpu)
      gpu="$2"
      shift 2
      ;;
    --no-check-certificate)
      no_check_certificate=1
      shift
      ;;
    --ca-certificate)
      ca_certificate="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --num-workers)
      num_workers="$2"
      shift 2
      ;;
    --steps)
      steps="$2"
      shift 2
      ;;
    --exp-name)
      exp_name="$2"
      shift 2
      ;;
    --project-name)
      project_name="$2"
      shift 2
      ;;
    --wandb-mode)
      wandb_mode="$2"
      shift 2
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    --resume)
      resume=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$command" in
  verify|prepare-pi0|smoke|train-flow|train-idp)
    ;;
  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac

export OPENPI_CACHE="$openpi_cache"
export HF_HOME="${HF_HOME:-$OPENPI_CACHE/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OPENPI_CACHE/xdg}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OPENPI_CACHE/openpi}"
export OPENPI_CKPT_HOME="${OPENPI_CKPT_HOME:-$OPENPI_CACHE/openpi_checkpoints}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
unset HF_ENDPOINT HF_HUB_ENDPOINT

if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME" "$OPENPI_CKPT_HOME" "$XDG_CACHE_HOME"

download_args=(--jobs "$jobs")
if [[ "$no_check_certificate" == "1" ]]; then
  download_args+=(--no-check-certificate)
elif [[ -n "$ca_certificate" ]]; then
  download_args+=(--ca-certificate "$ca_certificate")
fi

pi0_base_pt="$OPENPI_DATA_HOME/pytorch/pi0_base_pytorch"

print_env() {
  echo "OPENPI_CACHE=$OPENPI_CACHE"
  echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
  echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
  echo "OPENPI_CKPT_HOME=$OPENPI_CKPT_HOME"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
}

verify_libero() {
  echo "== Verify LIBERO dataset =="
  "$repo_root/scripts/download_libero_wget.sh" --check-only "${download_args[@]}"
}

prepare_pi0() {
  echo "== Prepare pi0_base PyTorch checkpoint =="
  "$repo_root/scripts/download_pi0_base_wget.sh" "${download_args[@]}"
}

run_train() {
  local mode="$1"
  local default_exp="$2"
  local default_steps="$3"
  local default_batch="$4"
  local default_workers="$5"

  local run_exp="${exp_name:-$default_exp}"
  local run_steps="${steps:-$default_steps}"
  local run_batch="${batch_size:-$default_batch}"
  local run_workers="${num_workers:-$default_workers}"

  if [[ ! -f "$pi0_base_pt/model.safetensors" ]]; then
    echo "Missing pi0 base PyTorch checkpoint: $pi0_base_pt/model.safetensors" >&2
    echo "Run: scripts/cluster_next_step.sh prepare-pi0" >&2
    exit 1
  fi

  train_args=(
    pi0_libero
    --exp_name "$run_exp"
    --project_name "$project_name"
    --pytorch_weight_path "$pi0_base_pt"
    --assets_base_dir ./assets
    --checkpoint_base_dir "$OPENPI_CKPT_HOME"
    --batch_size "$run_batch"
    --num_workers "$run_workers"
    --num_train_steps "$run_steps"
    --pytorch_loss_type "$mode"
  )

  if [[ "$mode" == "idp_geo" ]]; then
    train_args+=(--pytorch_idp_tau 0.1 --pytorch_valid_action_dim 7)
  fi
  if [[ "$run_steps" -le 100 ]]; then
    train_args+=(--log_interval 1 --save_interval 5)
    export WANDB_MODE="${wandb_mode:-disabled}"
  else
    train_args+=(--log_interval 100 --save_interval 1000)
    if [[ -n "$wandb_mode" ]]; then
      export WANDB_MODE="$wandb_mode"
    fi
  fi
  if [[ "$overwrite" == "1" ]]; then
    train_args+=(--overwrite)
  fi
  if [[ "$resume" == "1" ]]; then
    train_args+=(--resume)
  fi

  echo "== Run train_pytorch.py =="
  echo "mode=$mode exp=$run_exp steps=$run_steps batch=$run_batch workers=$run_workers WANDB_MODE=${WANDB_MODE:-<default>}"
  (
    cd "$repo_root"
    HF_HUB_OFFLINE=1 uv run scripts/train_pytorch.py "${train_args[@]}"
  )
}

print_env

case "$command" in
  verify)
    verify_libero
    ;;
  prepare-pi0)
    prepare_pi0
    ;;
  smoke)
    verify_libero
    prepare_pi0
    overwrite=1
    run_train flow "${exp_name:-smoke_flow}" "${steps:-10}" "${batch_size:-8}" "${num_workers:-4}"
    ;;
  train-flow)
    run_train flow "${exp_name:-flow_full_30k}" "${steps:-30000}" "${batch_size:-32}" "${num_workers:-8}"
    ;;
  train-idp)
    run_train idp_geo "${exp_name:-idp_geo_full_30k}" "${steps:-30000}" "${batch_size:-32}" "${num_workers:-8}"
    ;;
esac
