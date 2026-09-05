output "table_name" {
  value = aws_dynamodb_table.live_match_state.name
}

output "table_arn" {
  value = aws_dynamodb_table.live_match_state.arn
}
