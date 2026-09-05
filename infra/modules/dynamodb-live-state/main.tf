# Replaces the ingestor-go/internal/store/postgres.go live_match_state table
# (PROJECT_SPEC.md Phase 3). Like that table, this only ever holds each
# fixture's latest polled state - no history - so a single-key, on-demand
# table is enough: no sort key, no capacity planning.
resource "aws_dynamodb_table" "live_match_state" {
  name         = var.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "fixture_id"

  attribute {
    name = "fixture_id"
    type = "N"
  }

  # The ingestor sets expires_at on every write to ~6h past the poll, so a
  # fixture the live poller stops touching (match finished, host crashed
  # mid-match, etc.) ages out on its own instead of accumulating stale rows
  # forever - the "no history" table needs no manual cleanup job either.
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}
