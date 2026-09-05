output "model_artifacts_bucket" {
  value = module.s3_model_artifacts.bucket_name
}

output "live_match_state_table" {
  value = module.dynamodb_live_state.table_name
}

output "live_events_queue_url" {
  value = module.sqs_live_events.queue_url
}
