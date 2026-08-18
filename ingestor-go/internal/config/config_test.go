package config

import "testing"

// TestLivePollInterval_StaysWithinHourlyCeiling guards the actual
// constraint that matters: a ticker fired at this interval must never
// produce more than maxLivePollsPerHour ticks within any rolling hour.
func TestLivePollInterval_StaysWithinHourlyCeiling(t *testing.T) {
	interval := livePollInterval()

	ticksPerHour := int(60*60) / int(interval.Seconds())
	if ticksPerHour > maxLivePollsPerHour {
		t.Fatalf("interval %s allows %d ticks/hour, want <= %d", interval, ticksPerHour, maxLivePollsPerHour)
	}
}
