#!/usr/bin/env bash
# Project task runner. Usage: ./manage.sh <test|rebuild|verify|all>
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"

case "${1:-help}" in
  test)     # run the Python suites and the app's JavaScript suite
    "$PYTHON" -m pytest -q
    node --test "04-app/tests/*.test.js"
    ;;
  rebuild)  # silver -> gold data -> model (grid: minutes) -> app assets
    "$PYTHON" 01-bronze/build.py
    "$PYTHON" 02-silver/build.py
    "$PYTHON" 03-gold/build.py
    "$PYTHON" 03-gold/build_app.py
    ;;
  verify)   # prove every rebuild matches disk byte-for-byte (model refits the grid)
    "$PYTHON" 01-bronze/build.py --check-idempotent
    "$PYTHON" 02-silver/build.py --check-idempotent
    "$PYTHON" 03-gold/build.py --check-idempotent
    "$PYTHON" 03-gold/build_app.py --check-idempotent
    ;;
  all)      # full confidence pass
    "$0" test && "$0" rebuild && "$0" verify
    ;;
  *)
    echo "usage: ./manage.sh <test|rebuild|verify|all>"
    exit 1
    ;;
esac
