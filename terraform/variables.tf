variable "data_lake_bucket" {
  default = "dtc_flights_project_dlake"
}
# locals {
#   data_lake_bucket = "dtc_project_lake"
# }

variable "project" {
  description = "The project id that we created on gcp"
  default = "dtc-project-346013"
}

variable "region" {
  description = "Region for GCP resources. Choose as per your location: https://cloud.google.com/about/locations"
  default = "europe-west2"
  type = string
}

variable "zone" {
  description = "Zone for GCP resources. Choose as per your location: https://cloud.google.com/about/locations"
  default = "europe-west2-a"
  type = string
}

variable "storage_class" {
  description = "Storage class type for the bucket."
  default = "STANDARD"
}

variable "vm_instance" {
  description = "The name of our VM Instance or the Compute Engine"
  type = string
  default = "e2-instance"
}

variable "vm_instance_type" {
  description = "The instance type of the Compute Engine"
  type = string
  default = "e2-standard-2"
}

