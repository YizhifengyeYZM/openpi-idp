#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Download the physical-intelligence/libero LeRobot dataset with wget.

Recommended:
  OPENPI_CACHE=/home/data/yangzemin/cache scripts/download_libero_wget.sh

Options:
  --dest DIR          Exact dataset directory. Default:
                      $HF_LEROBOT_HOME/physical-intelligence/libero, or
                      $OPENPI_CACHE/huggingface/lerobot/physical-intelligence/libero.
  --base-url URL      File base URL. Default:
                      https://huggingface.co/datasets/physical-intelligence/libero/resolve/main
  --manifest FILE     File list. Default: docs/libero_hf_paths.txt.
  --check-only        Do not download; only check local files.
  --no-verify         Skip offline LeRobotDataset verification.
  -h, --help          Show this help.

Environment:
  OPENPI_CACHE        Preferred large cache root.
  HF_HOME             HuggingFace cache root.
  HF_LEROBOT_HOME     LeRobot cache root.
  LIBERO_DIR          Same as --dest.
  LIBERO_BASE_URL     Same as --base-url.
  HF_TOKEN            Optional token; only used as a wget Authorization header.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/docs/libero_hf_paths.txt"
base_url="${LIBERO_BASE_URL:-https://huggingface.co/datasets/physical-intelligence/libero/resolve/main}"
dest="${LIBERO_DIR:-}"
verify=1
check_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      dest="$2"
      shift 2
      ;;
    --base-url)
      base_url="$2"
      shift 2
      ;;
    --manifest)
      manifest="$2"
      shift 2
      ;;
    --check-only)
      check_only=1
      shift
      ;;
    --no-verify)
      verify=0
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

if [[ ! -f "$manifest" ]]; then
  echo "Manifest not found: $manifest" >&2
  exit 1
fi

if [[ -z "$dest" ]]; then
  if [[ -n "${HF_LEROBOT_HOME:-}" ]]; then
    dest="$HF_LEROBOT_HOME/physical-intelligence/libero"
  elif [[ -n "${OPENPI_CACHE:-}" ]]; then
    export HF_HOME="${HF_HOME:-$OPENPI_CACHE/huggingface}"
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
    dest="$HF_LEROBOT_HOME/physical-intelligence/libero"
  else
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
    dest="$HF_LEROBOT_HOME/physical-intelligence/libero"
    echo "OPENPI_CACHE is not set; defaulting to $dest" >&2
  fi
fi

mkdir -p "$dest"

case "$dest" in
  */physical-intelligence/libero)
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${dest%/physical-intelligence/libero}}"
    ;;
esac

echo "LIBERO destination: $dest"
echo "Manifest: $manifest"
echo "Base URL: ${base_url%/}"
echo "For later training shells: export HF_LEROBOT_HOME=\"$HF_LEROBOT_HOME\""
if env | grep -qi '_proxy='; then
  echo "Proxy variables detected:"
  env | grep -i '_proxy=' | sed 's/=.*/=<set>/'
fi

download_one() {
  local path="$1"
  local out="$dest/$path"
  local tmp="$out.partial"
  local url="${base_url%/}/$path"

  mkdir -p "$(dirname "$out")"

  if [[ -s "$out" ]]; then
    echo "[skip] $path"
    return 0
  fi

  echo "[download] $path"
  local args=(
    -c
    --tries=0
    --connect-timeout=30
    --read-timeout=120
    --waitretry=10
    --retry-connrefused
    -O "$tmp"
    "$url"
  )

  if [[ -n "${HF_TOKEN:-}" ]]; then
    args=(--header="Authorization: Bearer ${HF_TOKEN}" "${args[@]}")
  fi

  wget "${args[@]}"
  mv "$tmp" "$out"
}

if [[ "$check_only" -eq 0 ]]; then
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    download_one "$path"
  done < "$manifest"
fi

parquet_count="$(find "$dest/data" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
echo "Parquet files: $parquet_count / 1693"
if [[ ! -f "$dest/meta/info.json" ]]; then
  echo "Missing metadata: $dest/meta/info.json" >&2
  exit 1
fi

if [[ "$parquet_count" != "1693" ]]; then
  echo "Dataset is incomplete. Re-run this script to resume." >&2
  exit 1
fi

du -sh "$dest" || true

if [[ "$verify" -eq 1 ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; skipping LeRobotDataset verification."
    exit 0
  fi

  echo "Running offline LeRobotDataset verification..."
  (
    cd "$repo_root"
    unset HF_ENDPOINT HF_HUB_ENDPOINT
    export HF_HUB_OFFLINE=1
    export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
    uv run python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("physical-intelligence/libero")
print("num_episodes:", ds.num_episodes)
print("num_frames:", ds.num_frames)
print("tasks:", len(ds.meta.tasks))
PY
  )
fi

echo "LIBERO dataset is ready."
