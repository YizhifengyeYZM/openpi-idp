#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Download OpenPI pi0_base JAX checkpoint with wget and convert it to PyTorch.

Recommended:
  OPENPI_CACHE=/home/data/yangzemin/cache scripts/download_pi0_base_wget.sh

Options:
  --jax-dir DIR       JAX checkpoint root. Default:
                      $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base, or
                      $OPENPI_CACHE/openpi/openpi-assets/checkpoints/pi0_base.
  --pt-dir DIR        Converted PyTorch checkpoint dir. Default:
                      $OPENPI_DATA_HOME/pytorch/pi0_base_pytorch.
  --base-url URL      File base URL. Default:
                      https://storage.googleapis.com/openpi-assets/checkpoints/pi0_base/params
  --manifest FILE     File list. Default: docs/pi0_base_jax_paths.txt.
  --no-check-certificate
                      Pass wget --no-check-certificate. Useful behind a corporate
                      HTTPS inspection gateway with an untrusted internal CA.
  --ca-certificate FILE
                      Pass wget --ca-certificate FILE.
  --download-only     Download JAX checkpoint, but do not convert.
  --force-convert     Re-run conversion even if model.safetensors already exists.
  --check-only        Do not download; only check local files.
  -h, --help          Show this help.

Environment:
  OPENPI_CACHE        Preferred large cache root.
  OPENPI_DATA_HOME    OpenPI data/cache root.
  PI0_BASE_JAX        Same as --jax-dir.
  PI0_BASE_PT         Same as --pt-dir.
  PI0_BASE_BASE_URL   Same as --base-url.
  WGET_NO_CHECK_CERTIFICATE=1
                      Same as --no-check-certificate.
  WGET_CA_CERTIFICATE Same as --ca-certificate.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/docs/pi0_base_jax_paths.txt"
base_url="${PI0_BASE_BASE_URL:-https://storage.googleapis.com/openpi-assets/checkpoints/pi0_base/params}"
jax_dir="${PI0_BASE_JAX:-}"
pt_dir="${PI0_BASE_PT:-}"
no_check_certificate="${WGET_NO_CHECK_CERTIFICATE:-0}"
ca_certificate="${WGET_CA_CERTIFICATE:-}"
download_only=0
force_convert=0
check_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jax-dir)
      jax_dir="$2"
      shift 2
      ;;
    --pt-dir)
      pt_dir="$2"
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
    --download-only)
      download_only=1
      shift
      ;;
    --force-convert)
      force_convert=1
      shift
      ;;
    --check-only)
      check_only=1
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

if [[ -z "${OPENPI_DATA_HOME:-}" ]]; then
  if [[ -n "${OPENPI_CACHE:-}" ]]; then
    export OPENPI_DATA_HOME="$OPENPI_CACHE/openpi"
  else
    export OPENPI_DATA_HOME="$HOME/.cache/openpi"
    echo "OPENPI_CACHE is not set; defaulting OPENPI_DATA_HOME to $OPENPI_DATA_HOME" >&2
  fi
fi

jax_dir="${jax_dir:-$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base}"
pt_dir="${pt_dir:-$OPENPI_DATA_HOME/pytorch/pi0_base_pytorch}"

mkdir -p "$jax_dir/params" "$(dirname "$pt_dir")"

echo "pi0 base JAX checkpoint: $jax_dir"
echo "pi0 base PyTorch output: $pt_dir"
echo "Manifest: $manifest"
echo "Base URL: ${base_url%/}"
echo "For later training shells: export OPENPI_DATA_HOME=\"$OPENPI_DATA_HOME\""
echo "For training: --pytorch_weight_path \"$pt_dir\""
if [[ "$no_check_certificate" == "1" ]]; then
  echo "wget certificate verification: disabled (--no-check-certificate)"
elif [[ -n "$ca_certificate" ]]; then
  echo "wget CA certificate: $ca_certificate"
fi

download_one() {
  local path="$1"
  local out="$jax_dir/params/$path"
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

  if [[ "$no_check_certificate" == "1" ]]; then
    args=(--no-check-certificate "${args[@]}")
  elif [[ -n "$ca_certificate" ]]; then
    args=(--ca-certificate="$ca_certificate" "${args[@]}")
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

missing=0
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  if [[ ! -f "$jax_dir/params/$path" ]]; then
    echo "Missing: $jax_dir/params/$path" >&2
    missing=1
  fi
done < "$manifest"

if [[ "$missing" -ne 0 ]]; then
  echo "pi0 base JAX checkpoint is incomplete. Re-run this script to resume." >&2
  exit 1
fi

du -sh "$jax_dir" || true

if [[ "$download_only" -eq 1 ]]; then
  echo "Downloaded JAX checkpoint only."
  exit 0
fi

if [[ -f "$pt_dir/model.safetensors" && "$force_convert" -eq 0 ]]; then
  echo "Converted checkpoint already exists: $pt_dir/model.safetensors"
  echo "Use --force-convert to regenerate it."
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; cannot convert checkpoint." >&2
  exit 1
fi

(
  cd "$repo_root"
  uv run examples/convert_jax_model_to_pytorch.py \
    --config_name pi0_libero \
    --checkpoint_dir "$jax_dir" \
    --output_path "$pt_dir"
)

test -f "$pt_dir/model.safetensors"
du -sh "$pt_dir" || true
echo "pi0 base PyTorch checkpoint is ready."
