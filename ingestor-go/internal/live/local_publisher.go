package live

import (
	"context"
	"log/slog"
)

// LocalPublisher just logs events. It stands in for an SQS-backed
// EventPublisher until real AWS infra (§2 of PROJECT_SPEC.md) is
// provisioned - swap it out for an SQS client behind the same interface
// once that's ready.
type LocalPublisher struct {
	logger *slog.Logger
}

func NewLocalPublisher(logger *slog.Logger) *LocalPublisher {
	return &LocalPublisher{logger: logger}
}

func (p *LocalPublisher) PublishEvents(ctx context.Context, events []Event) error {
	for _, e := range events {
		p.logger.Info("live event",
			"fixture_id", e.FixtureID,
			"event_id", e.EventID,
			"type_id", e.TypeID,
			"participant_id", e.ParticipantID,
			"player", e.PlayerName,
			"minute", e.Minute,
		)
	}
	return nil
}
