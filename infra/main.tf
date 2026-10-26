locals {
  services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "billingbudgets.googleapis.com",
    "iam.googleapis.com",
  ]
  name = "pharma-qa"
}

data "google_project" "this" {}

resource "google_project_service" "apis" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# --- Image registry ----------------------------------------------------------

resource "google_artifact_registry_repository" "images" {
  repository_id = local.name
  location      = var.region
  format        = "DOCKER"

  # Keep only the two newest images so storage stays near the 0.5 GB free tier
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-two-newest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 2
    }
  }
  cleanup_policies {
    id     = "delete-older"
    action = "DELETE"
    condition {
      older_than = "86400s"
    }
  }

  depends_on = [google_project_service.apis]
}

# Cloud Build runs as the default compute service account in new projects;
# it needs to push images and write build logs.
resource "google_project_iam_member" "build_sa" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/logging.logWriter",
    "roles/storage.objectViewer",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"

  depends_on = [google_project_service.apis]
}

# --- Runtime identity and the API key secret ----------------------------------

resource "google_service_account" "run" {
  account_id   = "${local.name}-run"
  display_name = "Pharma Q&A Cloud Run runtime"
  depends_on   = [google_project_service.apis]
}

# The secret's value is added by hand (see README), never by Terraform, so the
# key never lands in Terraform state.
resource "google_secret_manager_secret" "anthropic_key" {
  secret_id = "anthropic-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "run_reads_key" {
  secret_id = google_secret_manager_secret.anthropic_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

# --- The service ----------------------------------------------------------------

resource "google_cloud_run_v2_service" "app" {
  count               = var.image == "" ? 0 : 1
  name                = local.name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.run.email
    max_instance_request_concurrency = 8
    timeout                          = "120s"

    scaling {
      # 0 = scale to zero (free while idle, but a cold start takes ~75 s);
      # 1 = keep one instance warm for a demo day (billed while warm)
      min_instance_count = var.min_instances
      max_instance_count = 2 # caps spend if the URL gets hammered
    }

    containers {
      image = var.image
      ports {
        container_port = 8000
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi" # torch + embedding model + index need ~1 GB
        }
        cpu_idle          = true # CPU billed only while handling requests
        startup_cpu_boost = true
      }
      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "ASK_RATE_LIMIT"
        value = "10"
      }
      # Torch sizes its thread pools from the host's cores, not the container's
      # vCPU limit. HF_HUB_OFFLINE: the model is baked into the image, so skip the
      # Hub update check on every cold start.
      dynamic "env" {
        for_each = { OMP_NUM_THREADS = "2", MKL_NUM_THREADS = "2", TOKENIZERS_PARALLELISM = "false", HF_HUB_OFFLINE = "1" }
        content {
          name  = env.key
          value = env.value
        }
      }
      startup_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds    = 5
        failure_threshold = 24 # allow up to 2 minutes for model load
      }
    }
  }

  depends_on = [google_secret_manager_secret_iam_member.run_reads_key]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.image == "" ? 0 : 1
  name     = google_cloud_run_v2_service.app[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- Budget alert: email the billing admins if spend passes 50% / 100% ----------

resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "${local.name} monthly"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      units = tostring(var.budget_amount)
    }
  }
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  depends_on = [google_project_service.apis]
}
