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

PUSH=1
SYNC=0
DRY_RUN=0
ALLOW_EMPTY=0
REMOTE="origin"
BRANCH=""
MSG=""

usage() {
  cat <<'USAGE'
Usage:
  ./commit.sh [options] [commit message]

Options:
  --no-push           Commit only, do not push.
  --push              Commit and push (default).
  --sync              If branch is behind remote, run pull --rebase --autostash first.
  --remote <name>     Remote name (default: origin).
  --branch <name>     Branch name (default: current branch, fallback main).
  --allow-empty       Allow creating an empty commit when no file changes.
  --dry-run           Print commands without executing.
  -h, --help          Show this help.

Examples:
  ./commit.sh
  ./commit.sh "feat: update training pipeline"
  ./commit.sh --sync "chore: sync + commit"
  ./commit.sh --no-push "wip: local checkpoint"
USAGE
}

run_cmd() {
  echo "+ $*"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-push)
      PUSH=0
      shift
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --sync)
      SYNC=1
      shift
      ;;
    --remote)
      if [[ $# -lt 2 ]]; then
        echo "[commit.sh] --remote requires a value" >&2
        exit 1
      fi
      REMOTE="$2"
      shift 2
      ;;
    --branch)
      if [[ $# -lt 2 ]]; then
        echo "[commit.sh] --branch requires a value" >&2
        exit 1
      fi
      BRANCH="$2"
      shift 2
      ;;
    --allow-empty)
      ALLOW_EMPTY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        MSG="$*"
      fi
      break
      ;;
    *)
      if [[ -z "${MSG}" ]]; then
        MSG="$1"
      else
        MSG+=" $1"
      fi
      shift
      ;;
  esac
done

if [[ -z "${MSG}" ]]; then
  MSG="auto: $(date '+%Y-%m-%d %H:%M:%S')"
fi

if ! git remote get-url "${REMOTE}" >/dev/null 2>&1; then
  echo "[commit.sh] remote not found: ${REMOTE}" >&2
  exit 1
fi

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git branch --show-current)"
  if [[ -z "${BRANCH}" ]]; then
    BRANCH="main"
  fi
fi

if [[ ${DRY_RUN} -eq 0 ]]; then
  git fetch "${REMOTE}" "${BRANCH}" --quiet || true
fi

UPSTREAM_REF="${REMOTE}/${BRANCH}"
BEHIND=0
AHEAD=0
if git rev-parse --verify --quiet "${UPSTREAM_REF}" >/dev/null; then
  COUNTS="$(git rev-list --left-right --count "${UPSTREAM_REF}...HEAD")"
  BEHIND="$(echo "${COUNTS}" | awk '{print $1}')"
  AHEAD="$(echo "${COUNTS}" | awk '{print $2}')"
fi

if [[ ${SYNC} -eq 1 && ${BEHIND} -gt 0 ]]; then
  run_cmd git pull --rebase --autostash "${REMOTE}" "${BRANCH}"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    COUNTS="$(git rev-list --left-right --count "${UPSTREAM_REF}...HEAD")"
    BEHIND="$(echo "${COUNTS}" | awk '{print $1}')"
    AHEAD="$(echo "${COUNTS}" | awk '{print $2}')"
  fi
fi

if [[ ${PUSH} -eq 1 && ${BEHIND} -gt 0 ]]; then
  echo "[commit.sh] local branch is behind ${UPSTREAM_REF} by ${BEHIND} commit(s)." >&2
  echo "[commit.sh] rerun with --sync to auto rebase, or pull/rebase manually first." >&2
  exit 1
fi

CHANGED=0
if [[ -n "$(git status --porcelain)" ]]; then
  CHANGED=1
fi

if [[ ${CHANGED} -eq 1 ]]; then
  run_cmd git add -A
  if [[ ${DRY_RUN} -eq 0 ]]; then
    if ! git diff --cached --quiet; then
      run_cmd git commit -m "${MSG}"
      echo "[commit.sh] committed $(git rev-parse --short HEAD)"
    else
      echo "[commit.sh] nothing staged after add"
    fi
  else
    echo "+ git commit -m ""${MSG}"""
  fi
else
  if [[ ${ALLOW_EMPTY} -eq 1 ]]; then
    run_cmd git commit --allow-empty -m "${MSG}"
    if [[ ${DRY_RUN} -eq 0 ]]; then
      echo "[commit.sh] committed empty commit $(git rev-parse --short HEAD)"
    fi
  else
    echo "[commit.sh] no file changes; skip commit"
  fi
fi

if [[ ${PUSH} -eq 1 ]]; then
  run_cmd git push -u "${REMOTE}" "${BRANCH}"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[commit.sh] dry-run mode: push command printed, nothing executed"
  else
    echo "[commit.sh] pushed to ${REMOTE}/${BRANCH}"
  fi
else
  echo "[commit.sh] push disabled (--no-push)"
fi
