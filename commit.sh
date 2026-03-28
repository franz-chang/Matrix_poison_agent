#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "[commit.sh] git not found" >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "[commit.sh] .git not found in ${ROOT_DIR}" >&2
  exit 1
fi

PUSH=0
if [[ "${1:-}" == "--push" ]]; then
  PUSH=1
  shift
fi

if [[ $# -gt 0 ]]; then
  MSG="$*"
else
  MSG="auto: $(date '+%Y-%m-%d %H:%M:%S')"
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "[commit.sh] no changes to commit"
  exit 0
fi

git add -A

if git diff --cached --quiet; then
  echo "[commit.sh] nothing staged after add"
  exit 0
fi

git commit -m "${MSG}"

echo "[commit.sh] committed $(git rev-parse --short HEAD)"

if [[ ${PUSH} -eq 1 ]]; then
  BRANCH="$(git branch --show-current)"
  if [[ -z "${BRANCH}" ]]; then
    BRANCH="main"
  fi
  git push -u origin "${BRANCH}"
  echo "[commit.sh] pushed to origin/${BRANCH}"
fi
