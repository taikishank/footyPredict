package live

import (
	"context"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/sqs"
	"github.com/aws/aws-sdk-go-v2/service/sqs/types"
)

// fakeSQSSender is a minimal sqsBatchSender that lets tests script
// per-call behavior via a slice of response functions, one per expected
// SendMessageBatch call.
type fakeSQSSender struct {
	responses []func(entries []types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput
	calls     int
}

func (f *fakeSQSSender) SendMessageBatch(ctx context.Context, in *sqs.SendMessageBatchInput, _ ...func(*sqs.Options)) (*sqs.SendMessageBatchOutput, error) {
	i := f.calls
	f.calls++
	if i >= len(f.responses) {
		// Default: everything succeeds.
		return &sqs.SendMessageBatchOutput{}, nil
	}
	return f.responses[i](in.Entries), nil
}

func testEvents(n int) []Event {
	events := make([]Event, n)
	for i := 0; i < n; i++ {
		events[i] = Event{FixtureID: 1, EventID: int64(i), TypeID: 14, ParticipantID: 10, PlayerName: "P", Minute: 10}
	}
	return events
}

func TestSQSPublisher_PublishEvents_AllSucceedFirstTry(t *testing.T) {
	sender := &fakeSQSSender{
		responses: []func([]types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput{
			func(entries []types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput {
				return &sqs.SendMessageBatchOutput{}
			},
		},
	}
	p := NewSQSPublisher(sender, "queue-url")

	if err := p.PublishEvents(context.Background(), testEvents(3)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sender.calls != 1 {
		t.Fatalf("got %d SendMessageBatch calls, want 1 (no retries)", sender.calls)
	}
}

func TestSQSPublisher_PublishEvents_RetriesTransientFailureAndSucceeds(t *testing.T) {
	sender := &fakeSQSSender{
		responses: []func([]types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput{
			func(entries []types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput {
				return &sqs.SendMessageBatchOutput{
					Failed: []types.BatchResultErrorEntry{
						{Id: strAddr("1"), Message: strAddr("throttled"), Code: strAddr("Throttling"), SenderFault: false},
					},
				}
			},
			func(entries []types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput {
				if len(entries) != 1 || deref(entries[0].Id) != "1" {
					t.Fatalf("retry should only resend the failed entry, got %+v", entries)
				}
				return &sqs.SendMessageBatchOutput{}
			},
		},
	}
	p := NewSQSPublisher(sender, "queue-url")

	if err := p.PublishEvents(context.Background(), testEvents(3)); err != nil {
		t.Fatalf("unexpected error after successful retry: %v", err)
	}
	if sender.calls != 2 {
		t.Fatalf("got %d SendMessageBatch calls, want 2 (initial + 1 retry)", sender.calls)
	}
}

func TestSQSPublisher_PublishEvents_SenderFaultNotRetried(t *testing.T) {
	sender := &fakeSQSSender{
		responses: []func([]types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput{
			func(entries []types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput {
				return &sqs.SendMessageBatchOutput{
					Failed: []types.BatchResultErrorEntry{
						{Id: strAddr("0"), Message: strAddr("malformed"), Code: strAddr("InvalidMessageContents"), SenderFault: true},
					},
				}
			},
		},
	}
	p := NewSQSPublisher(sender, "queue-url")

	err := p.PublishEvents(context.Background(), testEvents(1))
	if err == nil {
		t.Fatal("expected error for sender fault, got nil")
	}
	if !strings.Contains(err.Error(), "malformed") {
		t.Fatalf("expected error to reflect sender fault, got: %v", err)
	}
	if sender.calls != 1 {
		t.Fatalf("got %d SendMessageBatch calls, want 1 (sender fault must not be retried)", sender.calls)
	}
}

func TestSQSPublisher_PublishEvents_FailsEveryRetryReturnsError(t *testing.T) {
	alwaysFails := func(entries []types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput {
		return &sqs.SendMessageBatchOutput{
			Failed: []types.BatchResultErrorEntry{
				{Id: strAddr("0"), Message: strAddr("service unavailable"), Code: strAddr("ServiceUnavailable"), SenderFault: false},
			},
		}
	}
	sender := &fakeSQSSender{
		responses: []func([]types.SendMessageBatchRequestEntry) *sqs.SendMessageBatchOutput{
			alwaysFails, alwaysFails, alwaysFails,
		},
	}
	p := NewSQSPublisher(sender, "queue-url")

	err := p.PublishEvents(context.Background(), testEvents(1))
	if err == nil {
		t.Fatal("expected error after exhausting retries, got nil")
	}
	if !strings.Contains(err.Error(), "exhausting retries") {
		t.Fatalf("expected error to mention retries were exhausted, got: %v", err)
	}
	if sender.calls != maxSendAttempts {
		t.Fatalf("got %d SendMessageBatch calls, want %d (maxSendAttempts)", sender.calls, maxSendAttempts)
	}
}
