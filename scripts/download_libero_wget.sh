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
  --no-check-certificate
                      Pass wget --no-check-certificate. Useful behind a corporate
                      HTTPS inspection gateway with an untrusted internal CA.
  --ca-certificate FILE
                      Pass wget --ca-certificate FILE.
  --max-passes N      Number of manifest passes before giving up. Default: 0,
                      meaning retry until complete.
  --retry-sleep SEC   Seconds to sleep between incomplete passes. Default: 60.
  --wget-tries N      Per-file wget tries in each pass. Default: 20.
  --jobs N            Parallel wget jobs per pass. Default: 1.
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
  WGET_NO_CHECK_CERTIFICATE=1
                      Same as --no-check-certificate.
  WGET_CA_CERTIFICATE Same as --ca-certificate.
  DOWNLOAD_MAX_PASSES Same as --max-passes.
  DOWNLOAD_RETRY_SLEEP
                      Same as --retry-sleep.
  WGET_TRIES          Same as --wget-tries.
  DOWNLOAD_JOBS       Same as --jobs.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/docs/libero_hf_paths.txt"
base_url="${LIBERO_BASE_URL:-https://huggingface.co/datasets/physical-intelligence/libero/resolve/main}"
dest="${LIBERO_DIR:-}"
no_check_certificate="${WGET_NO_CHECK_CERTIFICATE:-0}"
ca_certificate="${WGET_CA_CERTIFICATE:-}"
max_passes="${DOWNLOAD_MAX_PASSES:-0}"
retry_sleep="${DOWNLOAD_RETRY_SLEEP:-60}"
wget_tries="${WGET_TRIES:-20}"
jobs="${DOWNLOAD_JOBS:-1}"
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
    --no-check-certificate)
      no_check_certificate=1
      shift
      ;;
    --ca-certificate)
      ca_certificate="$2"
      shift 2
      ;;
    --max-passes)
      max_passes="$2"
      shift 2
      ;;
    --retry-sleep)
      retry_sleep="$2"
      shift 2
      ;;
    --wget-tries)
      wget_tries="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
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
if ! [[ "$jobs" =~ ^[0-9]+$ ]] || [[ "$jobs" -lt 1 ]]; then
  echo "--jobs must be a positive integer, got: $jobs" >&2
  exit 2
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
echo "Retry policy: max_passes=$max_passes, retry_sleep=${retry_sleep}s, wget_tries=$wget_tries, jobs=$jobs"
if [[ "$no_check_certificate" == "1" ]]; then
  echo "wget certificate verification: disabled (--no-check-certificate)"
elif [[ -n "$ca_certificate" ]]; then
  echo "wget CA certificate: $ca_certificate"
fi
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
    --tries="$wget_tries"
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
  if [[ "$no_check_certificate" == "1" ]]; then
    args=(--no-check-certificate "${args[@]}")
  elif [[ -n "$ca_certificate" ]]; then
    args=(--ca-certificate="$ca_certificate" "${args[@]}")
  fi

  if wget "${args[@]}"; then
    mv "$tmp" "$out"
    return 0
  fi

  echo "[warn] wget failed for $path; keeping partial file for a later retry." >&2
  return 1
}

missing_manifest_count() {
  local missing=0
  local path
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if [[ ! -f "$dest/$path" ]]; then
      missing=$((missing + 1))
    fi
  done < "$manifest"
  echo "$missing"
}

parquet_count() {
  find "$dest/data" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' '
}

if [[ "$check_only" -eq 0 ]]; then
  pass=1
  while true; do
    echo "Download pass $pass starting..."
    failures=0
    pids=()
    while IFS= read -r path; do
      [[ -z "$path" ]] && continue
      download_one "$path" &
      pids+=("$!")

      if [[ "${#pids[@]}" -ge "$jobs" ]]; then
        for pid in "${pids[@]}"; do
          if ! wait "$pid"; then
            failures=$((failures + 1))
          fi
        done
        pids=()
      fi
    done < "$manifest"
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failures=$((failures + 1))
      fi
    done

    missing_now="$(missing_manifest_count)"
    parquet_now="$(parquet_count)"
    echo "Download pass $pass finished: failures=$failures, missing_manifest_files=$missing_now, parquet=$parquet_now/1693"

    if [[ "$missing_now" == "0" ]]; then
      break
    fi

    if [[ "$max_passes" != "0" && "$pass" -ge "$max_passes" ]]; then
      echo "Reached --max-passes=$max_passes with missing files remaining." >&2
      break
    fi

    echo "Sleeping ${retry_sleep}s before retrying incomplete files..."
    sleep "$retry_sleep"
    pass=$((pass + 1))
  done
fi

missing_count="$(missing_manifest_count)"
parquet_count="$(parquet_count)"
echo "Manifest files missing: $missing_count"
echo "Parquet files: $parquet_count / 1693"
if [[ "$missing_count" != "0" ]]; then
  echo "Dataset is incomplete. Re-run this script to resume." >&2
  exit 1
fi
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
