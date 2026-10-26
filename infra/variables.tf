variable "project_id" {
  description = "GCP project that holds everything (created by bootstrap.sh)"
  type        = string
}

variable "region" {
  description = "Cloud Run region; europe-west1 is a Tier 1 (cheapest, largest free tier) region"
  type        = string
  default     = "europe-west1"
}

variable "billing_account" {
  description = "Billing account ID, used only for the budget alert"
  type        = string
}

variable "image" {
  description = "Container image to run. Empty on the first apply, before any image is built."
  type        = string
  default     = ""
}

variable "budget_amount" {
  description = "Monthly budget alert, in the billing account's currency"
  type        = number
  default     = 1
}

variable "min_instances" {
  description = "Instances kept warm. 0 is free while idle; set 1 for a demo day (MIN_INSTANCES=1 infra/deploy.sh)"
  type        = number
  default     = 0
}
