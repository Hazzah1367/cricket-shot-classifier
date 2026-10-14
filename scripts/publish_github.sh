#!/usr/bin/env bash
# Run locally after gh auth login. Creates a private repo by default.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo 'Install GitHub CLI from https://cli.github.com'; exit 1; }
login="$(gh api user --jq .login)"
if [[ "$login" != 'Hazzah1367' ]]; then
  echo "Expected Hazzah1367; authenticated as $login. Switch accounts before publishing."
  exit 1
fi
project_dir="$(pwd -P)"
git_top="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ "$git_top" != "$project_dir" ]]; then
  git init -b main
fi
existing_origin="$(git remote get-url origin 2>/dev/null || true)"
if [[ -n "$existing_origin" && "$existing_origin" != 'https://github.com/Hazzah1367/cricket-shot-classifier.git' && "$existing_origin" != 'git@github.com:Hazzah1367/cricket-shot-classifier.git' ]]; then
  echo 'This folder has an unexpected origin; inspect it before publishing.'
  exit 1
fi
git add .
if ! git diff --cached --quiet; then
  git commit -m 'Build cricket shot classification MVP with audited image splits'
fi
if ! gh repo view Hazzah1367/cricket-shot-classifier >/dev/null 2>&1; then
  gh repo create Hazzah1367/cricket-shot-classifier --private --description 'Three-class cricket batting image classifier in PyTorch'
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin https://github.com/Hazzah1367/cricket-shot-classifier.git
fi
git push -u origin main
