#!/bin/sh
# Runs inside the localstack container on startup (mounted at
# /etc/localstack/init/ready.d/). Creates the same DynamoDB table and SQS
# queues that infra/modules/{dynamodb-live-state,sqs-live-events} create in
# real AWS, so local dev matches the deployed shape.
set -eu

awslocal dynamodb create-table \
  --table-name liveedge-live-match-state \
  --attribute-definitions AttributeName=fixture_id,AttributeType=N \
  --key-schema AttributeName=fixture_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

awslocal dynamodb update-time-to-live \
  --table-name liveedge-live-match-state \
  --time-to-live-specification "Enabled=true,AttributeName=expires_at"

awslocal sqs create-queue --queue-name liveedge-live-events-dlq
DLQ_ARN=$(awslocal sqs get-queue-attributes \
  --queue-url http://localhost:4566/000000000000/liveedge-live-events-dlq \
  --attribute-names QueueArn --query 'Attributes.QueueArn' --output text)

awslocal sqs create-queue --queue-name liveedge-live-events \
  --attributes "RedrivePolicy={\"deadLetterTargetArn\":\"$DLQ_ARN\",\"maxReceiveCount\":\"5\"}"
