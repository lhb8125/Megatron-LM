#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

case "${1:-}" in
    tiny)
        "${SCRIPT_DIR}/setup_env.sh"
        exec "${SCRIPT_DIR}/run_tiny_matrix.sh"
        ;;
    qwen30b)
        "${SCRIPT_DIR}/setup_env.sh"
        exec "${SCRIPT_DIR}/run_qwen_smoke.sh"
        ;;
    probe)
        "${SCRIPT_DIR}/setup_env.sh"
        exec "${SCRIPT_DIR}/version_probe.sh" "${2:-1}"
        ;;
    *)
        echo "usage: $0 {tiny|qwen30b|probe [world-size]}" >&2
        exit 2
        ;;
esac

