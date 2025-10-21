package poc

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/contribution"
	"github.com/davidcanhelp/sedition/validator"
)

// TestConsensusEngineIntegration tests the complete PoC consensus system
func TestConsensusEngineIntegration(t *testing.T) {
	// Create a new consensus engine
	minStake := big.NewInt(1000000) // 1 million tokens minimum
	blockTime := time.Second * 10   // 10 second blocks

	cfg := config.DefaultConsensusConfig()
	cfg.MinStakeRequired = minStake
	cfg.BlockTime = blockTime
	engine := consensus.NewEngine(cfg)

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

	// Test validator registration - validators is private, use network stats instead
	stats := engine.GetNetworkStats()
	if stats.TotalValidators != 3 {
		t.Errorf("Expected 3 validators, got %d", stats.TotalValidators)
	}

	// Submit some contributions
	contrib := contribution.Contribution{
		ID:            "contrib1",
		Timestamp:     time.Now(),
		Type:          contribution.CodeCommit,
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
	validatorObj, ok := engine.GetValidator("validator1")
	if !ok {
		t.Fatal("Failed to get validator1")
	}
	if validatorObj.TotalStake.Cmp(validatorObj.TokenStake) == 0 {
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
	netStats := engine.GetNetworkStats()
	if netStats.TotalValidators != 3 {
		t.Errorf("Expected 3 total validators, got %d", netStats.TotalValidators)
	}
	if netStats.ActiveValidators != 3 {
		t.Errorf("Expected 3 active validators, got %d", netStats.ActiveValidators)
	}
}

// TestQualityAnalyzer tests the quality analysis system
func TestQualityAnalyzer(t *testing.T) {
	analyzer := contribution.NewQualityAnalyzer()

	// Test high-quality contribution
	contrib := contribution.Contribution{
		ID:            "test_high",
		Timestamp:     time.Now(),
		QualityScore:  95.0,
		TestCoverage:  90.0,
		Documentation: 85.0,
		Complexity:    3.0,
		PeerReviews:   3,
		ReviewScore:   4.8,
		Type:          contribution.CodeCommit,
		LinesAdded:    100,
	}

	score, err := analyzer.AnalyzeContribution(contrib)
	if err != nil {
		t.Fatalf("Failed to analyze contribution: %v", err)
	}

	if score < 85.0 {
		t.Errorf("Expected high quality score (>85), got %.2f", score)
	}

	// Test low-quality contribution
	lowQualityContrib := contribution.Contribution{
		ID:            "test_low",
		Timestamp:     time.Now(),
		QualityScore:  45.0,
		TestCoverage:  30.0,
		Documentation: 20.0,
		Complexity:    15.0,
		PeerReviews:   1,
		ReviewScore:   2.0,
		Type:          contribution.CodeCommit,
		LinesAdded:    50,
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
	tracker := validator.NewReputationTracker()

	// Get reputation for a contributor (automatically initialized)
	initialRep := tracker.GetReputation("contributor1")
	baseRep := 5.0 // Default base reputation
	if initialRep != baseRep {
		t.Errorf("Expected initial reputation %.1f, got %.1f",
			baseRep, initialRep)
	}

	// Submit a high-quality contribution
	goodContrib := contribution.Contribution{
		ID:           "good1",
		Timestamp:    time.Now(),
		QualityScore: 90.0,
		Type:         contribution.CodeCommit,
		LinesAdded:   100,
		TestCoverage: 85.0,
		Documentation: 80.0,
		Complexity:   5.0,
	}

	tracker.UpdateReputation("contributor1", goodContrib)

	newRep := tracker.GetReputation("contributor1")
	if newRep <= initialRep {
		t.Error("Reputation should increase after good contribution")
	}

	// Test slashing
	tracker.ApplySlashing("contributor1", validator.MaliciousCode)

	slashedRep := tracker.GetReputation("contributor1")
	if slashedRep >= newRep {
		t.Error("Reputation should decrease after slashing")
	}

	// Test reputation retrieval after slashing
	finalRep := tracker.GetReputation("contributor1")
	if finalRep >= newRep {
		t.Error("Final reputation should be less than pre-slashing reputation")
	}

	t.Log("Reputation tracker tests passed")
}

// TestMetricsCalculator tests the comprehensive metrics system
func TestMetricsCalculator(t *testing.T) {
	calculator := contribution.NewMetricsCalculator()

	// Create contribution history
	history := []contribution.Contribution{
		{
			ID:            "contrib1",
			Timestamp:     time.Now().Add(-7 * 24 * time.Hour),
			QualityScore:  85.0,
			Type:          contribution.CodeCommit,
			LinesAdded:    120,
			LinesModified: 30,
			TestCoverage:  80.0,
			Documentation: 75.0,
			Complexity:    5.0,
			PeerReviews:   2,
			ReviewScore:   4.5,
		},
		{
			ID:            "contrib2",
			Timestamp:     time.Now().Add(-14 * 24 * time.Hour),
			QualityScore:  92.0,
			Type:          contribution.Testing,
			LinesAdded:    80,
			LinesModified: 10,
			TestCoverage:  95.0,
			Documentation: 85.0,
			Complexity:    3.0,
			PeerReviews:   3,
			ReviewScore:   4.8,
		},
		{
			ID:            "contrib3",
			Timestamp:     time.Now().Add(-21 * 24 * time.Hour),
			QualityScore:  78.0,
			Type:          contribution.Documentation,
			LinesAdded:    200,
			LinesModified: 50,
			TestCoverage:  70.0,
			Documentation: 95.0,
			Complexity:    2.0,
			PeerReviews:   1,
			ReviewScore:   4.0,
		},
	}

	// Calculate metrics
	metrics := calculator.CalculateMetrics(history)

	// Verify metrics are populated
	if metrics.TotalContributions != 3 {
		t.Errorf("Expected 3 contributions, got %d", metrics.TotalContributions)
	}

	if metrics.AverageQuality < 0 || metrics.AverageQuality > 100 {
		t.Errorf("Average quality should be 0-100, got %.2f", metrics.AverageQuality)
	}

	if metrics.TotalLinesChanged == 0 {
		t.Error("Total lines changed should not be zero")
	}

	if len(metrics.TypeDistribution) == 0 {
		t.Error("Type distribution should be populated")
	}

	t.Logf("Metrics calculated successfully: %d contributions, avg quality: %.2f",
		metrics.TotalContributions, metrics.AverageQuality)
}

// TestSlashingConditions tests various slashing scenarios
func TestSlashingConditions(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	cfg.MinStakeRequired = big.NewInt(1000000)
	cfg.BlockTime = time.Second * 10
	engine := consensus.NewEngine(cfg)

	// Register a validator
	err := engine.RegisterValidator("bad_validator", big.NewInt(5000000))
	if err != nil {
		t.Fatalf("Failed to register validator: %v", err)
	}

	badValidator, ok := engine.GetValidator("bad_validator")
	if !ok {
		t.Fatal("Failed to get bad_validator")
	}
	initialStake := new(big.Int).Set(badValidator.TokenStake)

	// Test different slashing reasons
	slashingTests := []struct {
		reason          validator.SlashingReason
		expectedPenalty bool
	}{
		{validator.MaliciousCode, true},
		{validator.FalseContribution, true},
		{validator.NetworkAttack, true},
		{validator.QualityViolation, true},
	}

	for _, test := range slashingTests {
		err := engine.SlashValidator("bad_validator", test.reason, "test evidence")
		if err != nil {
			t.Fatalf("Failed to slash validator for %v: %v", test.reason, err)
		}

		if test.expectedPenalty {
			currentValidator, _ := engine.GetValidator("bad_validator")
			if currentValidator.TokenStake.Cmp(initialStake) >= 0 {
				t.Errorf("Expected stake reduction for %v, but stake unchanged", test.reason)
			}
		}

		// Re-register validator for next test
		engine.RegisterValidator("bad_validator", big.NewInt(5000000))
	}
}

// BenchmarkBlockProposerSelection benchmarks the proposer selection algorithm
func BenchmarkBlockProposerSelection(b *testing.B) {
	cfg := config.DefaultConsensusConfig()
	cfg.MinStakeRequired = big.NewInt(1000000)
	cfg.BlockTime = time.Second * 10
	engine := consensus.NewEngine(cfg)

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

// Note: TrendDirection type removed as it's not implemented yet
// Helper function commented out until trend analysis is implemented
