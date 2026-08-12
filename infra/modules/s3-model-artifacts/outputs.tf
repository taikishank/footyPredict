output "bucket_name" {
  value = aws_s3_bucket.models.id
}

output "bucket_arn" {
  value = aws_s3_bucket.models.arn
}
