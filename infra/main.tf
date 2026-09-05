data "aws_caller_identity" "current" {}

module "s3_model_artifacts" {
  source = "./modules/s3-model-artifacts"

  bucket_name = "liveedge-model-artifacts-${data.aws_caller_identity.current.account_id}"
}

module "dynamodb_live_state" {
  source = "./modules/dynamodb-live-state"

  table_name = "liveedge-live-match-state"
}

module "sqs_live_events" {
  source = "./modules/sqs-live-events"

  queue_name = "liveedge-live-events"
}
