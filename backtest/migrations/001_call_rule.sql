-- Extracted verbatim from ingestor-go/internal/store/postgres.go, which
-- remains the source of truth - the Go Migrate() runs this same text on
-- ingestor startup. This copy exists so the column can be added without
-- booting the ingestor (which also polls fixtures and odds). Idempotent:
-- ADD COLUMN IF NOT EXISTS plus a guarded primary-key widening, so running
-- it twice, or after the ingestor has already applied it, is a no-op.
--
-- Apply:
--   psql "$POSTGRES_URL" -v ON_ERROR_STOP=1 -f backtest/migrations/001_call_rule.sql

BEGIN;

-- Which selection rule turned the snapshot's probabilities into this bet
-- (backtest/selection.py). model_version and call_rule are different axes:
-- the first says which artifact produced model_home/draw/away, the second
-- what was done with them. Every row written before the split was following
-- max_edge, which is what DEFAULT 'max_edge' labels them; the default is
-- then dropped so a writer that forgets the column fails loudly rather than
-- silently mislabelling a belief-arm row as an edge-arm one - the same
-- reason model_version drops its default above.
--
-- The key widens again: one fixture is legitimately graded once per model
-- version PER RULE, and the two rows deliberately share an edges snapshot,
-- so the snapshot foreign key must NOT widen with it.
ALTER TABLE prediction_calls ADD COLUMN IF NOT EXISTS call_rule TEXT NOT NULL DEFAULT 'max_edge';
ALTER TABLE prediction_calls ALTER COLUMN call_rule DROP DEFAULT;
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
		WHERE i.indrelid = 'prediction_calls'::regclass AND i.indisprimary
		  AND a.attname = 'call_rule'
	) THEN
		ALTER TABLE prediction_calls DROP CONSTRAINT prediction_calls_pkey;
		ALTER TABLE prediction_calls ADD PRIMARY KEY (fixture_id, model_version, call_rule);
	END IF;
END $$;

-- Same column and same widened key on the DC arm, for the same reason. The
-- DC arm selects its own side, so it has its own rule of the same name
-- rather than inheriting the singles pick (see backtest/selection.py).
ALTER TABLE double_chance_calls ADD COLUMN IF NOT EXISTS call_rule TEXT NOT NULL DEFAULT 'max_edge';
ALTER TABLE double_chance_calls ALTER COLUMN call_rule DROP DEFAULT;
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
		WHERE i.indrelid = 'double_chance_calls'::regclass AND i.indisprimary
		  AND a.attname = 'call_rule'
	) THEN
		ALTER TABLE double_chance_calls DROP CONSTRAINT double_chance_calls_pkey;
		ALTER TABLE double_chance_calls ADD PRIMARY KEY (fixture_id, model_version, call_rule);
	END IF;
END $$;

COMMIT;
