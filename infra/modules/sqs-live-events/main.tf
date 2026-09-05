# Fans out in-match events (PROJECT_SPEC.md Phase 3), replacing
# ingestor-go/internal/live/local_publisher.go's log-only stand-in. Standard
# (not FIFO) queue - the current consumer just logs events, so ordering and
# exactly-once delivery aren't load-bearing yet; revisit if a real consumer
# ends up needing per-fixture ordering.
resource "aws_sqs_queue" "live_events_dlq" {
  name                      = "${var.queue_name}-dlq"
  message_retention_seconds = 1209600 # 14 days, max - gives time to inspect failures
}

resource "aws_sqs_queue" "live_events" {
  name                       = var.queue_name
  visibility_timeout_seconds = 30
  message_retention_seconds  = 86400 # 1 day - these are near-real-time events, not durable history

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.live_events_dlq.arn
    maxReceiveCount     = 5
  })
}
