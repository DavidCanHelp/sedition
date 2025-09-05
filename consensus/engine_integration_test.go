package consensus_test

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

func TestConsensusEngineIntegration(t *testing.T) {
	// Create engine with custom config
	cfg := &config.ConsensusConfig{
		BlockTime:               1 * time.Second,
		EpochLength:             10,
		MinStakeRequired:        big.NewInt(100),
		SlashingRate:            0.1,
		InitialReputation:       5.0,
		ReputationDecayRate:     0.01,
		MinReputationMultiplier: 0.5,
		MaxReputationMultiplier: 2.0,
		ContributionWindow:      24 * time.Hour,
		MinContributionBonus:    0.8,
		MaxContributionBonus:    1.5,
		QualityThreshold:        75.0,
		InactivityPenalty:       0.8,
		NoRecentActivityPenalty: 0.9,
		ProposerHistorySize:     10,
		MaxProposerFrequency:    2,
		ProposerFrequencyWindow: 3,
	}

	engine := consensus.NewEngine(cfg)

	// Test validator registration
	t.Run("ValidatorRegistration", func(t *testing.T) {
		err := engine.RegisterValidator("validator1", big.NewInt(200))
		if err != nil {
			t.Fatalf("Failed to register validator: %v", err)
		}

		err = engine.RegisterValidator("validator2", big.NewInt(300))
		if err != nil {
			t.Fatalf("Failed to register validator: %v", err)
		}

		// Try to register with insufficient stake
		err = engine.RegisterValidator("validator3", big.NewInt(50))
		if err == nil {
			t.Error("Expected error for insufficient stake")
		}
	})

	// Test contribution submission
	t.Run("ContributionSubmission", func(t *testing.T) {
		contrib := contribution.Contribution{
			ID:            "contrib1",
			Timestamp:     time.Now(),
			Type:          contribution.CodeCommit,
			LinesAdded:    100,
			LinesModified: 50,
			TestCoverage:  80.0,
			Complexity:    10.0,
			Documentation: 70.0,
		}

		err := engine.SubmitContribution("validator1", contrib)
		if err != nil {
			t.Fatalf("Failed to submit contribution: %v", err)
		}

		// Try to submit for non-existent validator
		err = engine.SubmitContribution("nonexistent", contrib)
		if err == nil {
			t.Error("Expected error for non-existent validator")
		}
	})

	// Test block proposer selection
	t.Run("ProposerSelection", func(t *testing.T) {
		proposer, err := engine.SelectBlockProposer()
		if err != nil {
			t.Fatalf("Failed to select proposer: %v", err)
		}

		if proposer != "validator1" && proposer != "validator2" {
			t.Errorf("Unexpected proposer: %s", proposer)
		}
	})

	// Test slashing
	t.Run("ValidatorSlashing", func(t *testing.T) {
		err := engine.SlashValidator("validator1", validator.MaliciousCode, "test evidence")
		if err != nil {
			t.Fatalf("Failed to slash validator: %v", err)
		}

		// Get stats to verify slashing
		stats, err := engine.GetValidatorStats("validator1")
		if err != nil {
			t.Fatalf("Failed to get validator stats: %v", err)
		}

		if stats.SlashingCount != 1 {
			t.Errorf("Expected 1 slashing event, got %d", stats.SlashingCount)
		}

		// Slashed stake should be 180 (200 - 10%)
		expectedStake := big.NewInt(180)
		if stats.TokenStake.Cmp(expectedStake) != 0 {
			t.Errorf("Expected stake %v after slashing, got %v", expectedStake, stats.TokenStake)
		}
	})

	// Test epoch update
	t.Run("EpochUpdate", func(t *testing.T) {
		initialStats := engine.GetNetworkStats()
		initialEpoch := initialStats.CurrentEpoch

		engine.UpdateEpoch()

		updatedStats := engine.GetNetworkStats()
		if updatedStats.CurrentEpoch != initialEpoch+1 {
			t.Errorf("Expected epoch %d, got %d", initialEpoch+1, updatedStats.CurrentEpoch)
		}
	})

	// Test network statistics
	t.Run("NetworkStatistics", func(t *testing.T) {
		stats := engine.GetNetworkStats()

		if stats.TotalValidators != 2 {
			t.Errorf("Expected 2 validators, got %d", stats.TotalValidators)
		}

		if stats.ActiveValidators != 2 {
			t.Errorf("Expected 2 active validators, got %d", stats.ActiveValidators)
		}

		expectedTotalStake := big.NewInt(480) // 180 + 300
		if stats.TotalStaked.Cmp(expectedTotalStake) != 0 {
			t.Errorf("Expected total stake %v, got %v", expectedTotalStake, stats.TotalStaked)
		}
	})
}

func TestProposerFairness(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	cfg.MinStakeRequired = big.NewInt(100)
	engine := consensus.NewEngine(cfg)

	// Register multiple validators with equal stake
	for i := 1; i <= 5; i++ {
		addr := fmt.Sprintf("validator%d", i)
		err := engine.RegisterValidator(addr, big.NewInt(1000))
		if err != nil {
			t.Fatalf("Failed to register validator %s: %v", addr, err)
		}
	}

	// Select proposers many times and check distribution
	proposerCount := make(map[string]int)
	iterations := 1000

	for i := 0; i < iterations; i++ {
		proposer, err := engine.SelectBlockProposer()
		if err != nil {
			t.Fatalf("Failed to select proposer: %v", err)
		}
		proposerCount[proposer]++
	}

	// Check that all validators were selected
	for i := 1; i <= 5; i++ {
		addr := fmt.Sprintf("validator%d", i)
		if count, exists := proposerCount[addr]; !exists || count == 0 {
			t.Errorf("Validator %s was never selected", addr)
		}
	}

	// Check for reasonable distribution (each should get roughly 20%)
	expectedCount := iterations / 5
	tolerance := expectedCount / 4 // 25% tolerance

	for addr, count := range proposerCount {
		if count < expectedCount-tolerance || count > expectedCount+tolerance {
			t.Logf("Warning: Validator %s selected %d times (expected ~%d)",
				addr, count, expectedCount)
		}
	}
}

func TestContributionQualityImpact(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	cfg.MinStakeRequired = big.NewInt(100)
	engine := consensus.NewEngine(cfg)

	// Register two validators with same stake
	engine.RegisterValidator("highQuality", big.NewInt(1000))
	engine.RegisterValidator("lowQuality", big.NewInt(1000))

	// Submit high quality contributions for first validator
	for i := 0; i < 5; i++ {
		contrib := contribution.Contribution{
			ID:            fmt.Sprintf("high%d", i),
			Timestamp:     time.Now(),
			Type:          contribution.CodeCommit,
			LinesAdded:    100,
			TestCoverage:  90.0,
			Documentation: 85.0,
		}
		engine.SubmitContribution("highQuality", contrib)
	}

	// Submit low quality contributions for second validator
	for i := 0; i < 5; i++ {
		contrib := contribution.Contribution{
			ID:            fmt.Sprintf("low%d", i),
			Timestamp:     time.Now(),
			Type:          contribution.CodeCommit,
			LinesAdded:    100,
			TestCoverage:  20.0,
			Documentation: 10.0,
		}
		engine.SubmitContribution("lowQuality", contrib)
	}

	// Check validator stats
	highStats, _ := engine.GetValidatorStats("highQuality")
	lowStats, _ := engine.GetValidatorStats("lowQuality")

	// High quality validator should have better reputation
	if highStats.ReputationScore <= lowStats.ReputationScore {
		t.Errorf("High quality validator should have better reputation: high=%f, low=%f",
			highStats.ReputationScore, lowStats.ReputationScore)
	}

	// High quality validator should have higher total stake
	if highStats.TotalStake.Cmp(lowStats.TotalStake) <= 0 {
		t.Errorf("High quality validator should have higher total stake")
	}
}

func BenchmarkProposerSelection(b *testing.B) {
	cfg := config.DefaultConsensusConfig()
	engine := consensus.NewEngine(cfg)

	// Register 100 validators
	for i := 0; i < 100; i++ {
		addr := fmt.Sprintf("validator%d", i)
		stake := big.NewInt(int64(1000 + i*10))
		engine.RegisterValidator(addr, stake)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := engine.SelectBlockProposer()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkContributionSubmission(b *testing.B) {
	cfg := config.DefaultConsensusConfig()
	engine := consensus.NewEngine(cfg)

	engine.RegisterValidator("validator1", big.NewInt(1000))

	contrib := contribution.Contribution{
		ID:           "contrib",
		Timestamp:    time.Now(),
		Type:         contribution.CodeCommit,
		LinesAdded:   100,
		TestCoverage: 80.0,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := engine.SubmitContribution("validator1", contrib)
		if err != nil {
			b.Fatal(err)
		}
	}
}
