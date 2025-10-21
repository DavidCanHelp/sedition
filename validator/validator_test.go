package validator

import (
	"math/big"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/contribution"
)

// TestNewValidator tests validator creation
func TestNewValidator(t *testing.T) {
	address := "0x1234567890abcdef"
	stake := big.NewInt(1000000)
	reputation := 5.0

	v := NewValidator(address, stake, reputation)

	if v.Address != address {
		t.Errorf("Address mismatch: got %s, want %s", v.Address, address)
	}

	if v.TokenStake.Cmp(stake) != 0 {
		t.Errorf("Stake mismatch: got %s, want %s", v.TokenStake.String(), stake.String())
	}

	if v.ReputationScore != reputation {
		t.Errorf("Reputation mismatch: got %f, want %f", v.ReputationScore, reputation)
	}

	if !v.IsActive {
		t.Error("New validator should be active")
	}

	if len(v.RecentContribs) != 0 {
		t.Error("New validator should have no contributions")
	}

	if len(v.SlashingHistory) != 0 {
		t.Error("New validator should have no slashing history")
	}
}

// TestAddContribution tests adding contributions
func TestAddContribution(t *testing.T) {
	v := NewValidator("validator1", big.NewInt(1000), 5.0)

	contrib := contribution.Contribution{
		ID:           "contrib1",
		Timestamp:    time.Now(),
		QualityScore: 8.5,
	}

	v.AddContribution(contrib)

	if len(v.RecentContribs) != 1 {
		t.Errorf("Expected 1 contribution, got %d", len(v.RecentContribs))
	}

	if v.RecentContribs[0].ID != contrib.ID {
		t.Error("Contribution ID mismatch")
	}

	if !v.LastActivityTime.Equal(contrib.Timestamp) {
		t.Error("Last activity time not updated")
	}

	// Add multiple contributions
	for i := 0; i < 5; i++ {
		c := contribution.Contribution{
			ID:           string(rune('A' + i)),
			Timestamp:    time.Now().Add(time.Duration(i) * time.Minute),
			QualityScore: float64(i) + 1.0,
		}
		v.AddContribution(c)
	}

	if len(v.RecentContribs) != 6 {
		t.Errorf("Expected 6 contributions, got %d", len(v.RecentContribs))
	}
}

// TestApplySlashing tests slashing functionality
func TestApplySlashing(t *testing.T) {
	initialStake := big.NewInt(1000000)
	v := NewValidator("validator1", initialStake, 5.0)

	slashAmount := big.NewInt(100000)
	reason := MaliciousCode
	evidence := "Attempted to inject malicious code"

	v.ApplySlashing(slashAmount, reason, evidence)

	// Check stake reduced
	expectedStake := big.NewInt(900000) // 1000000 - 100000
	if v.TokenStake.Cmp(expectedStake) != 0 {
		t.Errorf("Stake after slashing: got %s, want %s", v.TokenStake.String(), expectedStake.String())
	}

	// Check slashing history
	if len(v.SlashingHistory) != 1 {
		t.Errorf("Expected 1 slashing event, got %d", len(v.SlashingHistory))
	}

	event := v.SlashingHistory[0]
	if event.Reason != reason {
		t.Errorf("Slashing reason mismatch: got %v, want %v", event.Reason, reason)
	}

	if event.AmountSlashed.Cmp(slashAmount) != 0 {
		t.Error("Slashed amount mismatch")
	}

	if event.Evidence != evidence {
		t.Errorf("Evidence mismatch: got %s, want %s", event.Evidence, evidence)
	}

	// Apply multiple slashing events
	v.ApplySlashing(big.NewInt(50000), DoubleProposal, "Double proposal detected")
	v.ApplySlashing(big.NewInt(25000), FalseContribution, "False contribution claim")

	if len(v.SlashingHistory) != 3 {
		t.Errorf("Expected 3 slashing events, got %d", len(v.SlashingHistory))
	}
}

// TestSlashingReasonString tests slashing reason string representation
func TestSlashingReasonString(t *testing.T) {
	tests := []struct {
		reason   SlashingReason
		expected string
	}{
		{MaliciousCode, "MaliciousCode"},
		{FalseContribution, "FalseContribution"},
		{DoubleProposal, "DoubleProposal"},
		{NetworkAttack, "NetworkAttack"},
		{QualityViolation, "QualityViolation"},
		{SlashingReason(999), "Unknown"},
	}

	for _, tt := range tests {
		if got := tt.reason.String(); got != tt.expected {
			t.Errorf("SlashingReason(%v).String() = %s, want %s", tt.reason, got, tt.expected)
		}
	}
}

// TestGetRecentContributionQuality tests quality calculation
func TestGetRecentContributionQuality(t *testing.T) {
	v := NewValidator("validator1", big.NewInt(1000), 5.0)

	// Test with no contributions
	since := time.Now().Add(-24 * time.Hour)
	quality := v.GetRecentContributionQuality(since)
	if quality != 0.0 {
		t.Errorf("Expected 0.0 quality with no contributions, got %f", quality)
	}

	// Add contributions with different timestamps
	now := time.Now()
	contributions := []struct {
		id      string
		time    time.Time
		quality float64
	}{
		{"c1", now.Add(-48 * time.Hour), 7.0}, // Old
		{"c2", now.Add(-12 * time.Hour), 8.0}, // Recent
		{"c3", now.Add(-6 * time.Hour), 9.0},  // Recent
		{"c4", now.Add(-1 * time.Hour), 10.0}, // Recent
	}

	for _, c := range contributions {
		v.AddContribution(contribution.Contribution{
			ID:           c.id,
			Timestamp:    c.time,
			QualityScore: c.quality,
		})
	}

	// Calculate quality for last 24 hours
	since = now.Add(-24 * time.Hour)
	avgQuality := v.GetRecentContributionQuality(since)

	// Should average 8.0, 9.0, and 10.0 = 9.0
	expected := (8.0 + 9.0 + 10.0) / 3.0
	if avgQuality != expected {
		t.Errorf("Average quality: got %f, want %f", avgQuality, expected)
	}

	// Calculate quality for last 72 hours (should include all)
	since = now.Add(-72 * time.Hour)
	avgQuality = v.GetRecentContributionQuality(since)
	expected = (7.0 + 8.0 + 9.0 + 10.0) / 4.0
	if avgQuality != expected {
		t.Errorf("Average quality (72h): got %f, want %f", avgQuality, expected)
	}

	// Test with cutoff that excludes all
	since = now.Add(1 * time.Hour)
	avgQuality = v.GetRecentContributionQuality(since)
	if avgQuality != 0.0 {
		t.Errorf("Expected 0.0 with future cutoff, got %f", avgQuality)
	}
}

// TestCleanupOldContributions tests contribution cleanup
func TestCleanupOldContributions(t *testing.T) {
	v := NewValidator("validator1", big.NewInt(1000), 5.0)

	now := time.Now()

	// Add contributions at different times
	contributions := []struct {
		id   string
		time time.Time
	}{
		{"old1", now.Add(-72 * time.Hour)},
		{"old2", now.Add(-48 * time.Hour)},
		{"recent1", now.Add(-12 * time.Hour)},
		{"recent2", now.Add(-6 * time.Hour)},
		{"recent3", now.Add(-1 * time.Hour)},
	}

	for _, c := range contributions {
		v.AddContribution(contribution.Contribution{
			ID:           c.id,
			Timestamp:    c.time,
			QualityScore: 5.0,
		})
	}

	if len(v.RecentContribs) != 5 {
		t.Errorf("Expected 5 contributions before cleanup, got %d", len(v.RecentContribs))
	}

	// Cleanup contributions older than 24 hours
	cutoff := now.Add(-24 * time.Hour)
	v.CleanupOldContributions(cutoff)

	// Should have 3 recent contributions left
	if len(v.RecentContribs) != 3 {
		t.Errorf("Expected 3 contributions after cleanup, got %d", len(v.RecentContribs))
	}

	// Verify the remaining contributions are the recent ones
	expectedIDs := map[string]bool{"recent1": true, "recent2": true, "recent3": true}
	for _, c := range v.RecentContribs {
		if !expectedIDs[c.ID] {
			t.Errorf("Unexpected contribution ID after cleanup: %s", c.ID)
		}
	}

	// Cleanup all
	v.CleanupOldContributions(now.Add(1 * time.Hour))
	if len(v.RecentContribs) != 0 {
		t.Errorf("Expected 0 contributions after full cleanup, got %d", len(v.RecentContribs))
	}
}

// TestCreateValidator tests the exported CreateValidator function
func TestCreateValidator(t *testing.T) {
	address := "0xValidator123"
	pubKey := "pubkey123"
	stake := big.NewInt(5000000)

	v := CreateValidator(address, pubKey, stake)

	if v.Address != address {
		t.Errorf("Address mismatch: got %s, want %s", v.Address, address)
	}

	if v.PublicKey != pubKey {
		t.Errorf("PublicKey mismatch: got %s, want %s", v.PublicKey, pubKey)
	}

	if v.TokenStake.Cmp(stake) != 0 {
		t.Error("Stake mismatch")
	}

	// Should have default reputation of 5.0
	if v.ReputationScore != 5.0 {
		t.Errorf("Default reputation should be 5.0, got %f", v.ReputationScore)
	}

	if !v.IsActive {
		t.Error("New validator should be active")
	}
}

// Benchmark tests
func BenchmarkNewValidator(b *testing.B) {
	stake := big.NewInt(1000000)
	for i := 0; i < b.N; i++ {
		NewValidator("validator1", stake, 5.0)
	}
}

func BenchmarkAddContribution(b *testing.B) {
	v := NewValidator("validator1", big.NewInt(1000000), 5.0)
	contrib := contribution.Contribution{
		ID:           "contrib1",
		Timestamp:    time.Now(),
		QualityScore: 8.5,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.AddContribution(contrib)
	}
}

func BenchmarkGetRecentContributionQuality(b *testing.B) {
	v := NewValidator("validator1", big.NewInt(1000000), 5.0)

	// Add some contributions
	now := time.Now()
	for i := 0; i < 100; i++ {
		v.AddContribution(contribution.Contribution{
			ID:           string(rune(i)),
			Timestamp:    now.Add(time.Duration(-i) * time.Hour),
			QualityScore: float64(i % 10),
		})
	}

	since := now.Add(-24 * time.Hour)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.GetRecentContributionQuality(since)
	}
}
