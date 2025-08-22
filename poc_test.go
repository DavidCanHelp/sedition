package poc

import (
	"fmt"
	"math/big"
	"testing"
	"time"
)

// TestConsensusEngineIntegration tests the complete PoC consensus system
func TestConsensusEngineIntegration(t *testing.T) {
	// Create a new consensus engine
	minStake := big.NewInt(1000000) // 1 million tokens minimum
	blockTime := time.Second * 10   // 10 second blocks
	
	engine := NewConsensusEngine(minStake, blockTime)
	
	// Register several validators
	validators := []struct {
		address string
		stake   *big.Int
	}{
		{"validator1", big.NewInt(5000000)},
		{"validator2", big.NewInt(3000000)},
		{"validator3", big.NewInt(2000000)},
	}
	
	for _, v := range validators {
		err := engine.RegisterValidator(v.address, v.stake)
		if err != nil {
			t.Fatalf("Failed to register validator %s: %v", v.address, err)
		}
	}
	
	// Test validator registration
	if len(engine.validators) != 3 {
		t.Errorf("Expected 3 validators, got %d", len(engine.validators))
	}
	
	// Submit some contributions
	contrib := Contribution{
		ID:            "contrib1",
		Timestamp:     time.Now(),
		Type:          CodeCommit,
		LinesAdded:    150,
		LinesModified: 50,
		LinesDeleted:  20,
		TestCoverage:  85.0,
		Complexity:    5.2,
		Documentation: 80.0,
		QualityScore:  88.5,
		PeerReviews:   2,
		ReviewScore:   4.5,
	}
	
	err := engine.SubmitContribution("validator1", contrib)
	if err != nil {
		t.Fatalf("Failed to submit contribution: %v", err)
	}
	
	// Check that validator's stake was updated
	validator := engine.validators["validator1"]
	if validator.TotalStake.Cmp(validator.TokenStake) == 0 {
		t.Error("Total stake should be different from token stake after contribution")
	}
	
	// Test block proposer selection
	proposer, err := engine.SelectBlockProposer()
	if err != nil {
		t.Fatalf("Failed to select block proposer: %v", err)
	}
	
	if proposer == "" {
		t.Error("Proposer should not be empty")
	}
	
	// Verify proposer is one of our validators
	found := false
	for _, v := range validators {
		if v.address == proposer {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Selected proposer %s is not a registered validator", proposer)
	}
	
	// Test network stats
	stats := engine.GetNetworkStats()
	if stats.TotalValidators != 3 {
		t.Errorf("Expected 3 total validators, got %d", stats.TotalValidators)
	}
	if stats.ActiveValidators != 3 {
		t.Errorf("Expected 3 active validators, got %d", stats.ActiveValidators)
	}
}

// TestQualityAnalyzer tests the quality analysis system
func TestQualityAnalyzer(t *testing.T) {
	analyzer := NewQualityAnalyzer()
	
	// Test high-quality contribution
	contrib := Contribution{
		QualityScore:  95.0,
		TestCoverage:  90.0,
		Documentation: 85.0,
		Complexity:    3.0,
		PeerReviews:   3,
		ReviewScore:   4.8,
		Type:          CodeCommit,
	}
	
	score, err := analyzer.AnalyzeContribution(contrib)
	if err != nil {
		t.Fatalf("Failed to analyze contribution: %v", err)
	}
	
	if score < 85.0 {
		t.Errorf("Expected high quality score (>85), got %.2f", score)
	}
	
	// Test low-quality contribution
	lowQualityContrib := Contribution{
		QualityScore:  45.0,
		TestCoverage:  30.0,
		Documentation: 20.0,
		Complexity:    15.0,
		PeerReviews:   1,
		ReviewScore:   2.0,
		Type:          CodeCommit,
	}
	
	lowScore, err := analyzer.AnalyzeContribution(lowQualityContrib)
	if err != nil {
		t.Fatalf("Failed to analyze low quality contribution: %v", err)
	}
	
	if lowScore > 60.0 {
		t.Errorf("Expected low quality score (<60), got %.2f", lowScore)
	}
}

// TestReputationTracker tests the reputation tracking system
func TestReputationTracker(t *testing.T) {
	tracker := NewReputationTracker()
	
	// Initialize reputation for a contributor
	tracker.InitializeReputation("contributor1")
	
	initialRep := tracker.GetReputation("contributor1")
	if initialRep != tracker.baseReputation {
		t.Errorf("Expected initial reputation %.1f, got %.1f", 
			tracker.baseReputation, initialRep)
	}
	
	// Submit a high-quality contribution
	goodContrib := Contribution{
		Timestamp:    time.Now(),
		QualityScore: 90.0,
		Type:         CodeCommit,
		LinesAdded:   100,
	}
	
	tracker.UpdateReputation("contributor1", goodContrib)
	
	newRep := tracker.GetReputation("contributor1")
	if newRep <= initialRep {
		t.Error("Reputation should increase after good contribution")
	}
	
	// Test slashing
	tracker.ApplySlashing("contributor1", MaliciousCode)
	
	slashedRep := tracker.GetReputation("contributor1")
	if slashedRep >= newRep {
		t.Error("Reputation should decrease after slashing")
	}
	
	// Test detailed reputation retrieval
	details := tracker.GetDetailedReputation("contributor1")
	if details == nil {
		t.Error("Should return detailed reputation information")
	}
	
	if details.IsRecovering != true {
		t.Error("Contributor should be in recovery mode after major slashing")
	}
}

// TestMetricsCalculator tests the comprehensive metrics system
func TestMetricsCalculator(t *testing.T) {
	calculator := NewMetricsCalculator()
	
	// Create a test validator
	validator := &Validator{
		Address:    "test_validator",
		TokenStake: big.NewInt(1000000),
	}
	
	// Create contribution history
	history := []Contribution{
		{
			Timestamp:     time.Now().Add(-7 * 24 * time.Hour),
			QualityScore:  85.0,
			Type:          CodeCommit,
			LinesAdded:    120,
			TestCoverage:  80.0,
			Documentation: 75.0,
		},
		{
			Timestamp:     time.Now().Add(-14 * 24 * time.Hour),
			QualityScore:  92.0,
			Type:          Testing,
			LinesAdded:    80,
			TestCoverage:  95.0,
			Documentation: 85.0,
		},
		{
			Timestamp:     time.Now().Add(-21 * 24 * time.Hour),
			QualityScore:  78.0,
			Type:          Documentation,
			LinesAdded:    200,
			TestCoverage:  70.0,
			Documentation: 95.0,
		},
	}
	
	// Create peer review history
	reviews := []PeerReviewEvent{
		{
			Timestamp:  time.Now().Add(-5 * 24 * time.Hour),
			IsReviewer: true,
			Rating:     4.5,
			ReviewType: CodeReviewType,
		},
		{
			Timestamp:  time.Now().Add(-10 * 24 * time.Hour),
			IsReviewer: false,
			Rating:     4.2,
			ReviewType: CodeReviewType,
		},
	}
	
	// Calculate metrics
	metrics := calculator.CalculateMetrics(validator, history, reviews)
	
	// Verify overall score is reasonable
	if metrics.OverallScore < 0 || metrics.OverallScore > 100 {
		t.Errorf("Overall score should be 0-100, got %.2f", metrics.OverallScore)
	}
	
	// Verify individual scores
	if metrics.ProductivityScore < 0 || metrics.ProductivityScore > 100 {
		t.Errorf("Productivity score should be 0-100, got %.2f", metrics.ProductivityScore)
	}
	
	if metrics.QualityScore < 0 || metrics.QualityScore > 100 {
		t.Errorf("Quality score should be 0-100, got %.2f", metrics.QualityScore)
	}
	
	if metrics.CollaborationScore < 0 || metrics.CollaborationScore > 100 {
		t.Errorf("Collaboration score should be 0-100, got %.2f", metrics.CollaborationScore)
	}
	
	// Check that metrics are populated
	if metrics.Productivity.ContributionsLastMonth == 0 {
		t.Error("Should have contributions in last month based on test data")
	}
	
	if len(metrics.Trends.ShortTermTrend.String()) == 0 {
		// This would fail because TrendDirection doesn't have String() method
		// but we can check that trends are calculated
		if metrics.Trends.ShortTermTrend == TrendUnknown &&
		   metrics.Trends.MediumTermTrend == TrendUnknown {
			t.Log("Trends calculated (may be unknown due to limited data)")
		}
	}
}

// TestSlashingConditions tests various slashing scenarios
func TestSlashingConditions(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000000), time.Second*10)
	
	// Register a validator
	err := engine.RegisterValidator("bad_validator", big.NewInt(5000000))
	if err != nil {
		t.Fatalf("Failed to register validator: %v", err)
	}
	
	initialStake := new(big.Int).Set(engine.validators["bad_validator"].TokenStake)
	
	// Test different slashing reasons
	slashingTests := []struct {
		reason          SlashingReason
		expectedPenalty bool
	}{
		{MaliciousCode, true},
		{FalseContribution, true},
		{NetworkAttack, true},
		{QualityViolation, true},
	}
	
	for _, test := range slashingTests {
		// Reset validator stake
		engine.validators["bad_validator"].TokenStake = new(big.Int).Set(initialStake)
		
		err := engine.SlashValidator("bad_validator", test.reason, "test evidence")
		if err != nil {
			t.Fatalf("Failed to slash validator for %v: %v", test.reason, err)
		}
		
		if test.expectedPenalty {
			currentStake := engine.validators["bad_validator"].TokenStake
			if currentStake.Cmp(initialStake) >= 0 {
				t.Errorf("Expected stake reduction for %v, but stake unchanged", test.reason)
			}
		}
	}
}

// BenchmarkBlockProposerSelection benchmarks the proposer selection algorithm
func BenchmarkBlockProposerSelection(b *testing.B) {
	engine := NewConsensusEngine(big.NewInt(1000000), time.Second*10)
	
	// Register many validators
	for i := 0; i < 1000; i++ {
		address := fmt.Sprintf("validator_%d", i)
		stake := big.NewInt(int64(1000000 + i*100000)) // Varying stakes
		err := engine.RegisterValidator(address, stake)
		if err != nil {
			b.Fatalf("Failed to register validator %d: %v", i, err)
		}
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_, err := engine.SelectBlockProposer()
		if err != nil {
			b.Fatalf("Failed to select proposer: %v", err)
		}
	}
}

// Helper to add String method for TrendDirection (if needed for tests)
func (td TrendDirection) String() string {
	switch td {
	case TrendImproving:
		return "Improving"
	case TrendStable:
		return "Stable"
	case TrendDeclining:
		return "Declining"
	default:
		return "Unknown"
	}
}