terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }
}

# Credentials come from GOOGLE_OAUTH_ACCESS_TOKEN, which deploy.sh sets from the
# isolated `pharma-qa` gcloud configuration, so application-default credentials
# (and any other gcloud account on this machine) are never touched.
provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}
