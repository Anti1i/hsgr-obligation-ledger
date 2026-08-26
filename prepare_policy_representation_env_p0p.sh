#!/usr/bin/env bash

P0P_LOCAL_ENV_ROOT=""
P0P_ENV_PYTHON=""

p0p_prepare_env() {
  local project_scratch="$1"
  local archive="$project_scratch/env/venv-p0p-v1.tar"
  [ -s "$archive" ] || { echo "FATAL: P0p environment archive missing: $archive" >&2; return 1; }
  P0P_LOCAL_ENV_ROOT="$(mktemp -d "/tmp/hsgr-p0p-${SLURM_JOB_ID:-manual}-XXXXXX")"
  tar -xf "$archive" -C "$P0P_LOCAL_ENV_ROOT"
  P0P_ENV_PYTHON="$P0P_LOCAL_ENV_ROOT/venv/bin/python"
  [ -x "$P0P_ENV_PYTHON" ] || { echo "FATAL: extracted P0p Python missing" >&2; return 1; }
}

p0p_cleanup_env() {
  case "$P0P_LOCAL_ENV_ROOT" in
    /tmp/hsgr-p0p-*) rm -rf -- "$P0P_LOCAL_ENV_ROOT" ;;
    "") ;;
    *) echo "WARNING: refusing unexpected local env cleanup path: $P0P_LOCAL_ENV_ROOT" >&2 ;;
  esac
}
