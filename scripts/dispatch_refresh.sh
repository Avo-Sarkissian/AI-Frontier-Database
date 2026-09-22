#!/usr/bin/env bash
# Trigger the "Refresh model data" workflow from outside GitHub.
#
# GitHub's own cron has dropped most of this repo's scheduled runs since
# 2026-08-26 (README, "Live Data Refresh"). An external scheduler calling this
# endpoint is the only trigger that has proved reliable. cron-job.org sends the
# same request every hour; this script is the local smoke test for the token.
#
#   GITHUB_TOKEN=github_pat_... scripts/dispatch_refresh.sh
#
# The token must be a fine-grained personal access token scoped to this ONE
# repository with "Actions: Read and write". Nothing else.
set -euo pipefail
: "${GITHUB_TOKEN:?set GITHUB_TOKEN to a fine-grained token with Actions: write on this repo}"
REPO="${REPO:-Avo-Sarkissian/AI-Frontier-Database}"
REF="${REF:-main}"
body=$(mktemp)
trap 'rm -f "$body"' EXIT
code=$(curl -sS -o "$body" -w '%{http_code}' -X POST \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/$REPO/actions/workflows/refresh.yml/dispatches" \
  -d "{\"ref\":\"$REF\"}")
case "$code" in
  204|200) echo "dispatched refresh.yml on $REPO@$REF (HTTP $code) — watch the Actions tab" ;;
  *) echo "dispatch failed: HTTP $code" >&2; cat "$body" >&2; echo >&2; exit 1 ;;
esac
