output "queue_url" {
  value = aws_sqs_queue.live_events.id
}

output "queue_arn" {
  value = aws_sqs_queue.live_events.arn
}

output "dlq_url" {
  value = aws_sqs_queue.live_events_dlq.id
}
