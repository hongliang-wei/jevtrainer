#!/usr/bin/env bash
# Fetch the Intern-Decision evaluation bundle and convert every registered dataset into the cache.
set -uo pipefail
cd "$(dirname "$0")/.."
jt data fetch intern
jt data prepare "${1:-all}" 2>prepare_errors.log | tee prepare_summary.log
echo PREPARE_DONE
