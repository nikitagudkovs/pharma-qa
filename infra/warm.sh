#!/usr/bin/env bash
# Keep one instance warm (no cold start) or go back to scale-to-zero, without rebuilding.
# Usage: infra/warm.sh on | off
set -euo pipefail
export CLOUDSDK_ACTIVE_CONFIG_NAME=pharma-qa
cd "$(dirname "$0")"
case "${1:-}" in on) MIN=1 ;; off) MIN=0 ;; *) echo "usage: warm.sh on|off" >&2; exit 2 ;; esac
export GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"
terraform apply -input=false -auto-approve -var "image=$(terraform output -raw image)" -var "min_instances=${MIN}" | grep -E "Apply complete|Error"
echo "min instances = ${MIN}"
