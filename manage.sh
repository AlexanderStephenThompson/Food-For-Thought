#!/usr/bin/env bash
# Project task runner. Usage: ./manage.sh <test|rebuild|verify|all>
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"

case "${1:-help}" in
  test)     # run the test suite
    "$PYTHON" -m pytest 01-bronze/tests -q
    ;;
  rebuild)  # rebuild every silver dataset from bronze + lexicons
    "$PYTHON" 01-bronze/build.py
    ;;
  verify)   # prove a rebuild matches disk byte-for-byte
    "$PYTHON" 01-bronze/build.py --check-idempotent
    ;;
  all)      # full confidence pass
    "$0" test && "$0" rebuild && "$0" verify
    ;;
  *)
    echo "usage: ./manage.sh <test|rebuild|verify|all>"
    exit 1
    ;;
esac
