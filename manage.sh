#!/usr/bin/env bash
# Project task runner. Usage: ./manage.sh <test|rebuild|verify|all>
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"

case "${1:-help}" in
  test)     # run both tiers' test suites
    "$PYTHON" -m pytest -q
    ;;
  rebuild)  # rebuild silver from bronze, then gold from silver
    "$PYTHON" 01-bronze/build.py
    "$PYTHON" 02-silver/build.py
    ;;
  verify)   # prove both rebuilds match disk byte-for-byte
    "$PYTHON" 01-bronze/build.py --check-idempotent
    "$PYTHON" 02-silver/build.py --check-idempotent
    ;;
  all)      # full confidence pass
    "$0" test && "$0" rebuild && "$0" verify
    ;;
  *)
    echo "usage: ./manage.sh <test|rebuild|verify|all>"
    exit 1
    ;;
esac
