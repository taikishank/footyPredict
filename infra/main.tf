data "aws_caller_identity" "current" {}

module "s3_model_artifacts" {
  source = "./modules/s3-model-artifacts"

  bucket_name = "liveedge-model-artifacts-${data.aws_caller_identity.current.account_id}"
}
