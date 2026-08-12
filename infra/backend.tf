terraform {
  backend "s3" {
    bucket       = "liveedge-tfstate-590184129653"
    key          = "liveedge/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}
