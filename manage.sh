#!/usr/bin/env bash
# Project task runner. Usage: ./manage.sh <test|rebuild|verify|all>
set -euo pipefail

PYTHON=".venv/bin/python"

case "${1:-help}" in
  test)     # run the test suite
    "$PYTHON" -m pytest -q
    ;;
  rebuild)  # rebuild every silver artifact from bronze + lexicons
    "$PYTHON" run_pipeline.py
    ;;
  verify)   # prove a rebuild matches disk byte-for-byte
    "$PYTHON" run_pipeline.py --check-idempotent
    ;;
  all)      # full confidence pass
    "$0" test && "$0" rebuild && "$0" verify
    ;;
  *)
    echo "usage: ./manage.sh <test|rebuild|verify|all>"
    exit 1
    ;;
esac
