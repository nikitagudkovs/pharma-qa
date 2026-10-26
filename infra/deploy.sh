#!/usr/bin/env bash
# Build the image in Cloud Build and deploy it with Terraform. Safe to rerun.
# Usage: infra/deploy.sh                    (scale to zero)
#        MIN_INSTANCES=1 infra/deploy.sh    (keep one instance warm, e.g. on a demo day)
set -euo pipefail
export CLOUDSDK_ACTIVE_CONFIG_NAME=pharma-qa
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
MIN="${MIN_INSTANCES:-0}"

PROJECT="$(sed -n 's/^project_id *= *"\(.*\)"/\1/p' terraform.tfvars)"
REGION="$(sed -n 's/^region *= *"\(.*\)"/\1/p' terraform.tfvars)"
export GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"

terraform init -input=false -upgrade >/dev/null

# 1. Foundation: APIs, registry, secret, IAM, budget (no service yet on a first run)
CURRENT_IMAGE="$(terraform output -raw image 2>/dev/null || true)"
terraform apply -input=false -auto-approve -var "image=${CURRENT_IMAGE}" -var "min_instances=${MIN}"

# 2. The API key must be in Secret Manager before the service can start
if [ -z "$(gcloud secrets versions list anthropic-api-key --project "$PROJECT" --filter='state=ENABLED' --format='value(name)' 2>/dev/null)" ]; then
  cat <<MSG

The anthropic-api-key secret has no value yet. Add it (the key is read straight
from .env, so it never appears on screen or in shell history), then rerun infra/deploy.sh. From the repo root:

  grep '^ANTHROPIC_API_KEY=' .env | cut -d= -f2- | tr -d '\\n' | gcloud secrets versions add anthropic-api-key --data-file=- --project $PROJECT --configuration pharma-qa

MSG
  exit 1
fi

# 3. Build for linux/amd64 on Google's side, tagged with the git commit
TAG="$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo -dirty)"
IMAGE="$(terraform output -raw image_repo)/app:${TAG}"
gcloud builds submit "$ROOT" --tag "$IMAGE" --project "$PROJECT" --region "$REGION"  # default machine = inside the free build-minutes

# 4. Deploy that image
terraform apply -input=false -auto-approve -var "image=${IMAGE}" -var "min_instances=${MIN}"
URL="$(terraform output -raw url)"
echo
echo "Deployed ${IMAGE}"
echo "  ${URL}"
echo -n "  health: " && curl -fsS --max-time 120 --retry 5 --retry-delay 10 --retry-all-errors "${URL}/api/examples" >/dev/null && echo "ok"
