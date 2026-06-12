#!/usr/bin/env bash
# Project task runner. Usage: ./manage.sh <test|rebuild|verify|all>
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"

case "${1:-help}" in
  test)     # run both tiers' test suites
    "$PYTHON" -m pytest -q
    ;;
  rebuild)  # rebuild silver, then gold data, then the model (model grid: minutes)
    "$PYTHON" 01-bronze/build.py
    "$PYTHON" 02-silver/build.py
    "$PYTHON" 03-gold/build.py
    ;;
  verify)   # prove all three rebuilds match disk byte-for-byte (refits the grid)
    "$PYTHON" 01-bronze/build.py --check-idempotent
    "$PYTHON" 02-silver/build.py --check-idempotent
    "$PYTHON" 03-gold/build.py --check-idempotent
    ;;
  all)      # full confidence pass
    "$0" test && "$0" rebuild && "$0" verify
    ;;
  *)
    echo "usage: ./manage.sh <test|rebuild|verify|all>"
    exit 1
    ;;
esac
