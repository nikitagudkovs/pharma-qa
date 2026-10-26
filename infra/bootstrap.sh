#!/usr/bin/env bash
# One-time: create the GCP project, link billing, write terraform.tfvars.
# Usage: infra/bootstrap.sh <billing-account-id> [project-id]
set -euo pipefail
export CLOUDSDK_ACTIVE_CONFIG_NAME=pharma-qa   # never touches your default gcloud config
cd "$(dirname "$0")"

BILLING="${1:?usage: bootstrap.sh <billing-account-id> [project-id]}"
PROJECT="${2:-pharma-qa-$(openssl rand -hex 3)}"
REGION="europe-west1"

if ! gcloud projects describe "$PROJECT" >/dev/null 2>&1; then
  gcloud projects create "$PROJECT" --name="Pharma QA"
fi
gcloud billing projects link "$PROJECT" --billing-account="$BILLING"
gcloud config set project "$PROJECT"
# Terraform can't enable APIs until these two are on
gcloud services enable serviceusage.googleapis.com cloudresourcemanager.googleapis.com --project "$PROJECT"

cat > terraform.tfvars <<TFVARS
project_id      = "$PROJECT"
region          = "$REGION"
billing_account = "$BILLING"
TFVARS
echo "Bootstrapped $PROJECT. Next: infra/deploy.sh"
