// Package awsclients constructs the DynamoDB and SQS clients shared by the
// live poller (internal/live) and the event consumer (cmd/live-consumer),
// with support for pointing at LocalStack in local dev
// (docker-compose.yml's localstack service) instead of real AWS.
package awsclients

import (
	"context"
	"fmt"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/sqs"
)

// NewDynamoDB builds a DynamoDB client for region, using endpointURL as the
// base endpoint if non-empty (LocalStack), otherwise the real AWS endpoint.
func NewDynamoDB(ctx context.Context, region, endpointURL string) (*dynamodb.Client, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}
	return dynamodb.NewFromConfig(cfg, func(o *dynamodb.Options) {
		if endpointURL != "" {
			o.BaseEndpoint = &endpointURL
		}
	}), nil
}

// NewSQS builds an SQS client for region, using endpointURL as the base
// endpoint if non-empty (LocalStack), otherwise the real AWS endpoint.
func NewSQS(ctx context.Context, region, endpointURL string) (*sqs.Client, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("loading AWS config: %w", err)
	}
	return sqs.NewFromConfig(cfg, func(o *sqs.Options) {
		if endpointURL != "" {
			o.BaseEndpoint = &endpointURL
		}
	}), nil
}
