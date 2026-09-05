package live

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/sqs"
	"github.com/aws/aws-sdk-go-v2/service/sqs/types"
)

// sqsBatchLimit is SQS's hard cap on SendMessageBatch entries per call.
const sqsBatchLimit = 10

// maxSendAttempts bounds how many times a group of failed entries is
// resent before giving up on the ones that are still failing.
const maxSendAttempts = 3

// retryBackoffBase is the base delay used between retry attempts. Attempt
// n (1-indexed, counting the first retry as attempt 2) sleeps for
// retryBackoffBase * 2^(n-2), i.e. 100ms, 200ms, 400ms, ...
const retryBackoffBase = 100 * time.Millisecond

// eventMessage is the JSON body published for each event - the log-only
// consumer (cmd/live-consumer) and any future real consumer both decode
// this shape.
type eventMessage struct {
	FixtureID     int64  `json:"fixture_id"`
	EventID       int64  `json:"event_id"`
	TypeID        int64  `json:"type_id"`
	ParticipantID int64  `json:"participant_id"`
	PlayerName    string `json:"player_name"`
	Minute        int    `json:"minute"`
}

// sqsBatchSender is the subset of *sqs.Client's API this package depends
// on. It exists so tests can supply a fake instead of a real SQS client;
// *sqs.Client satisfies it structurally.
type sqsBatchSender interface {
	SendMessageBatch(ctx context.Context, params *sqs.SendMessageBatchInput, optFns ...func(*sqs.Options)) (*sqs.SendMessageBatchOutput, error)
}

// SQSPublisher implements EventPublisher against the queue
// infra/modules/sqs-live-events provisions, replacing LocalPublisher.
type SQSPublisher struct {
	client   sqsBatchSender
	queueURL string
}

func NewSQSPublisher(client sqsBatchSender, queueURL string) *SQSPublisher {
	return &SQSPublisher{client: client, queueURL: queueURL}
}

// PublishEvents sends events in batches of up to sqsBatchLimit. Entries SQS
// reports as failed are retried (with backoff) up to maxSendAttempts total
// attempts, unless SQS flags the failure as a sender fault - those are not
// retried since resending the same malformed message can't succeed. It
// reports the first per-message send failure that's still outstanding after
// retries are exhausted, if any, after attempting every batch.
func (p *SQSPublisher) PublishEvents(ctx context.Context, events []Event) error {
	var firstErr error

	for start := 0; start < len(events); start += sqsBatchLimit {
		end := min(start+sqsBatchLimit, len(events))
		batch := events[start:end]

		entries := make([]types.SendMessageBatchRequestEntry, 0, len(batch))
		for i, e := range batch {
			body, err := json.Marshal(eventMessage{
				FixtureID:     e.FixtureID,
				EventID:       e.EventID,
				TypeID:        e.TypeID,
				ParticipantID: e.ParticipantID,
				PlayerName:    e.PlayerName,
				Minute:        e.Minute,
			})
			if err != nil {
				return fmt.Errorf("marshaling event %d for fixture %d: %w", e.EventID, e.FixtureID, err)
			}
			entries = append(entries, types.SendMessageBatchRequestEntry{
				// SQS only requires this to be unique within the batch
				// request, so the entry's index in it is enough.
				Id:          strAddr(strconv.Itoa(i)),
				MessageBody: strAddr(string(body)),
			})
		}

		failed, err := p.sendWithRetry(ctx, entries)
		if err != nil {
			return fmt.Errorf("sending event batch: %w", err)
		}
		if len(failed) > 0 && firstErr == nil {
			f := failed[0]
			firstErr = fmt.Errorf("sqs rejected message %s after exhausting retries: %s (%s)", deref(f.Id), deref(f.Message), deref(f.Code))
		}
	}

	return firstErr
}

// sendWithRetry sends entries via SendMessageBatch, retrying only the
// entries that come back in the output's Failed list (and only those not
// marked SenderFault) up to maxSendAttempts total attempts, backing off
// between attempts. It returns the entries that were still failing once
// retries were exhausted (or abandoned because they were sender faults).
func (p *SQSPublisher) sendWithRetry(ctx context.Context, entries []types.SendMessageBatchRequestEntry) ([]types.BatchResultErrorEntry, error) {
	pending := entries
	// permanent accumulates failures we've given up on across attempts -
	// sender faults (never retried) plus whatever is still failing once
	// retries are exhausted.
	var permanent []types.BatchResultErrorEntry
	var lastRetryableFailed []types.BatchResultErrorEntry

	for attempt := 1; attempt <= maxSendAttempts && len(pending) > 0; attempt++ {
		out, err := p.client.SendMessageBatch(ctx, &sqs.SendMessageBatchInput{
			QueueUrl: &p.queueURL,
			Entries:  pending,
		})
		if err != nil {
			return nil, err
		}

		if len(out.Failed) == 0 {
			return permanent, nil
		}

		byID := make(map[string]types.SendMessageBatchRequestEntry, len(pending))
		for _, e := range pending {
			byID[deref(e.Id)] = e
		}

		var retryable []types.SendMessageBatchRequestEntry
		lastRetryableFailed = nil
		for _, f := range out.Failed {
			if f.SenderFault {
				// Not retryable: the message itself is bad, so retrying
				// won't help.
				permanent = append(permanent, f)
				continue
			}
			if entry, ok := byID[deref(f.Id)]; ok {
				retryable = append(retryable, entry)
			}
			lastRetryableFailed = append(lastRetryableFailed, f)
		}

		pending = retryable

		if len(pending) == 0 {
			break
		}

		if attempt < maxSendAttempts {
			backoff := retryBackoffBase * time.Duration(1<<(attempt-1))
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	return append(permanent, lastRetryableFailed...), nil
}

func strAddr(s string) *string { return &s }

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
