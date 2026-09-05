package live

import (
	"context"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
)

// liveStateTTL is how long a written row survives before DynamoDB's TTL
// sweep reclaims it (see infra/modules/dynamodb-live-state). Comfortably
// longer than any match plus stoppage time, so a live fixture's state never
// expires mid-poll; short enough that a poller that stops touching a
// fixture (match finished, crash) doesn't leave rows around indefinitely.
const liveStateTTL = 6 * time.Hour

// dynamoStateItem mirrors the postgres.go live_match_state row shape this
// replaces (PROJECT_SPEC.md Phase 3).
type dynamoStateItem struct {
	FixtureID int64  `dynamodbav:"fixture_id"`
	State     string `dynamodbav:"state"`
	HomeGoals int    `dynamodbav:"home_goals"`
	AwayGoals int    `dynamodbav:"away_goals"`
	UpdatedAt int64  `dynamodbav:"updated_at"` // unix seconds
	ExpiresAt int64  `dynamodbav:"expires_at"` // unix seconds, DynamoDB TTL attribute
}

// DynamoStateStore implements StateStore against the DynamoDB table
// infra/modules/dynamodb-live-state provisions, replacing the Postgres
// live_match_state table.
type DynamoStateStore struct {
	client    *dynamodb.Client
	tableName string
}

func NewDynamoStateStore(client *dynamodb.Client, tableName string) *DynamoStateStore {
	return &DynamoStateStore{client: client, tableName: tableName}
}

// UpsertLiveState overwrites each fixture's item with its latest polled
// state, one PutItem per fixture. Live poll cycles are small (concurrently
// live fixtures, not the whole fixture list), so this favors simplicity
// over BatchWriteItem's partial-failure handling.
func (s *DynamoStateStore) UpsertLiveState(ctx context.Context, states []MatchState) error {
	now := time.Now()
	for _, state := range states {
		item, err := attributevalue.MarshalMap(dynamoStateItem{
			FixtureID: state.FixtureID,
			State:     state.State,
			HomeGoals: state.HomeGoals,
			AwayGoals: state.AwayGoals,
			UpdatedAt: now.Unix(),
			ExpiresAt: now.Add(liveStateTTL).Unix(),
		})
		if err != nil {
			return fmt.Errorf("marshaling live state for fixture %d: %w", state.FixtureID, err)
		}

		_, err = s.client.PutItem(ctx, &dynamodb.PutItemInput{
			TableName: &s.tableName,
			Item:      item,
		})
		if err != nil {
			return fmt.Errorf("putting live state for fixture %d: %w", state.FixtureID, err)
		}
	}
	return nil
}
