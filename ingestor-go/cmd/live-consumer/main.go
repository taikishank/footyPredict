// Command live-consumer long-polls the live-events SQS queue and logs each
// event. It's a placeholder consumer (PROJECT_SPEC.md Phase 3): nothing
// downstream needs these events yet, so this just proves the queue works
// end-to-end - swap in real consumption logic (e.g. driving the dashboard's
// live event feed) behind the same loop once something needs it.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/aws/aws-sdk-go-v2/service/sqs"
	"github.com/aws/aws-sdk-go-v2/service/sqs/types"

	"github.com/taikishank/liveedge/ingestor-go/internal/awsclients"
	"github.com/taikishank/liveedge/ingestor-go/internal/config"
)

// eventMessage mirrors the JSON body internal/live/sqs_publisher.go sends -
// duplicated rather than shared so this consumer only depends on the wire
// contract, not the producer's internals.
type eventMessage struct {
	FixtureID     int64  `json:"fixture_id"`
	EventID       int64  `json:"event_id"`
	TypeID        int64  `json:"type_id"`
	ParticipantID int64  `json:"participant_id"`
	PlayerName    string `json:"player_name"`
	Minute        int    `json:"minute"`
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	cfg, err := config.Load()
	if err != nil {
		logger.Error("loading config", "error", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	client, err := awsclients.NewSQS(ctx, cfg.AWSRegion, cfg.SQSEndpointURL)
	if err != nil {
		logger.Error("configuring sqs client", "error", err)
		os.Exit(1)
	}

	logger.Info("live-consumer starting", "queue_url", cfg.LiveEventsQueueURL)

	for {
		select {
		case <-ctx.Done():
			logger.Info("live-consumer stopped")
			return
		default:
		}

		out, err := client.ReceiveMessage(ctx, &sqs.ReceiveMessageInput{
			QueueUrl:            &cfg.LiveEventsQueueURL,
			MaxNumberOfMessages: 10,
			WaitTimeSeconds:     20, // long poll - avoids busy-looping on an empty queue
		})
		if err != nil {
			if ctx.Err() != nil {
				continue // shutting down mid-call; let the select above return
			}
			logger.Error("receiving messages", "error", err)
			continue
		}

		for _, msg := range out.Messages {
			handleMessage(ctx, client, cfg.LiveEventsQueueURL, msg, logger)
		}
	}
}

func handleMessage(ctx context.Context, client *sqs.Client, queueURL string, msg types.Message, logger *slog.Logger) {
	var evt eventMessage
	if msg.Body == nil {
		logger.Warn("received message with empty body", "message_id", derefOr(msg.MessageId, ""))
		return
	}
	if err := json.Unmarshal([]byte(*msg.Body), &evt); err != nil {
		logger.Error("decoding event message", "error", err, "message_id", derefOr(msg.MessageId, ""))
		return // leave it for the redrive policy to eventually DLQ
	}

	logger.Info("live event",
		"fixture_id", evt.FixtureID,
		"event_id", evt.EventID,
		"type_id", evt.TypeID,
		"participant_id", evt.ParticipantID,
		"player", evt.PlayerName,
		"minute", evt.Minute,
	)

	if _, err := client.DeleteMessage(ctx, &sqs.DeleteMessageInput{
		QueueUrl:      &queueURL,
		ReceiptHandle: msg.ReceiptHandle,
	}); err != nil {
		logger.Error("deleting processed message", "error", err, "message_id", derefOr(msg.MessageId, ""))
	}
}

func derefOr(s *string, fallback string) string {
	if s == nil {
		return fallback
	}
	return *s
}
