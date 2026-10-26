output "url" {
  value = var.image == "" ? null : google_cloud_run_v2_service.app[0].uri
}

output "image_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "secret" {
  value = google_secret_manager_secret.anthropic_key.secret_id
}

output "image" {
  description = "Image currently deployed; deploy.sh reuses it for foundation-only applies"
  value       = var.image
}
