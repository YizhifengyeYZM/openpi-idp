#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Download the PaliGemma tokenizer used by OpenPI.

Recommended:
  OPENPI_CACHE=/home/data/yangzemin/cache scripts/download_paligemma_tokenizer_wget.sh --no-check-certificate

Options:
  --dest FILE          Output file. Default:
                       $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model.
  --url URL            Tokenizer URL. Default:
                       https://storage.googleapis.com/big_vision/paligemma_tokenizer.model.
  --no-check-certificate
                       Pass wget --no-check-certificate.
  --ca-certificate FILE
                       Pass wget --ca-certificate FILE.
  --max-passes N       Number of attempts before giving up. Default: 0,
                       meaning retry until complete.
  --retry-sleep SEC    Seconds to sleep between attempts. Default: 60.
  --wget-tries N       Per-attempt wget tries. Default: 20.
  --jobs N             Accepted for compatibility with cluster_next_step.sh; ignored.
  --check-only         Do not download; only check local file.
  --no-verify          Skip sentencepiece load verification.
  -h, --help           Show this help.

Environment:
  OPENPI_CACHE         Preferred large cache root.
  OPENPI_DATA_HOME     OpenPI data/cache root.
  PALIGEMMA_TOKENIZER  Same as --dest.
  PALIGEMMA_TOKENIZER_URL
                       Same as --url.
  WGET_NO_CHECK_CERTIFICATE=1
                       Same as --no-check-certificate.
  WGET_CA_CERTIFICATE  Same as --ca-certificate.
  DOWNLOAD_JOBS        Accepted for compatibility; ignored.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
url="${PALIGEMMA_TOKENIZER_URL:-https://storage.googleapis.com/big_vision/paligemma_tokenizer.model}"
dest="${PALIGEMMA_TOKENIZER:-}"
no_check_certificate="${WGET_NO_CHECK_CERTIFICATE:-0}"
ca_certificate="${WGET_CA_CERTIFICATE:-}"
max_passes="${DOWNLOAD_MAX_PASSES:-0}"
retry_sleep="${DOWNLOAD_RETRY_SLEEP:-60}"
wget_tries="${WGET_TRIES:-20}"
check_only=0
verify=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      dest="$2"
      shift 2
      ;;
    --url)
      url="$2"
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

if [[ -z "${OPENPI_DATA_HOME:-}" ]]; then
  if [[ -n "${OPENPI_CACHE:-}" ]]; then
    export OPENPI_DATA_HOME="$OPENPI_CACHE/openpi"
  else
    export OPENPI_DATA_HOME="$HOME/.cache/openpi"
    echo "OPENPI_CACHE is not set; defaulting OPENPI_DATA_HOME to $OPENPI_DATA_HOME" >&2
  fi
fi

dest="${dest:-$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model}"
mkdir -p "$(dirname "$dest")"

echo "PaliGemma tokenizer: $dest"
echo "URL: $url"
echo "For later training shells: export OPENPI_DATA_HOME=\"$OPENPI_DATA_HOME\""
echo "Retry policy: max_passes=$max_passes, retry_sleep=${retry_sleep}s, wget_tries=$wget_tries"
if [[ "$no_check_certificate" == "1" ]]; then
  echo "wget certificate verification: disabled (--no-check-certificate)"
elif [[ -n "$ca_certificate" ]]; then
  echo "wget CA certificate: $ca_certificate"
fi

download_tokenizer() {
  local tmp="$dest.partial"

  if [[ -s "$dest" ]]; then
    echo "[skip] tokenizer already exists"
    return 0
  fi

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

  if [[ "$no_check_certificate" == "1" ]]; then
    args=(--no-check-certificate "${args[@]}")
  elif [[ -n "$ca_certificate" ]]; then
    args=(--ca-certificate="$ca_certificate" "${args[@]}")
  fi

  echo "[download] paligemma_tokenizer.model"
  if wget "${args[@]}"; then
    mv "$tmp" "$dest"
    return 0
  fi

  echo "[warn] wget failed; keeping partial file for a later retry." >&2
  return 1
}

if [[ "$check_only" -eq 0 ]]; then
  pass=1
  while true; do
    if download_tokenizer; then
      break
    fi

    if [[ "$max_passes" != "0" && "$pass" -ge "$max_passes" ]]; then
      echo "Reached --max-passes=$max_passes; tokenizer is still missing." >&2
      break
    fi

    echo "Sleeping ${retry_sleep}s before retrying tokenizer download..."
    sleep "$retry_sleep"
    pass=$((pass + 1))
  done
fi

if [[ ! -f "$dest" ]]; then
  echo "Missing tokenizer: $dest" >&2
  exit 1
fi

ls -lh "$dest"

if [[ "$verify" -eq 1 ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; skipping tokenizer verification."
    exit 0
  fi

  echo "Running sentencepiece verification..."
  (
    cd "$repo_root"
    TOKENIZER_PATH="$dest" uv run python - <<'PY'
import os
import sentencepiece

path = os.environ["TOKENIZER_PATH"]
with open(path, "rb") as f:
    tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
print("tokenizer vocab_size:", tokenizer.vocab_size())
PY
  )
fi

echo "PaliGemma tokenizer is ready."
