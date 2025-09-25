package consensus

import (
	"fmt"
	"math/big"
	"sync"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/contribution"
	"github.com/davidcanhelp/sedition/validator"
)

func TestNewEngine_Initialization(t *testing.T) {
	tests := []struct {
		name   string
		config *config.ConsensusConfig
	}{
		{
			name:   "with nil config uses defaults",
			config: nil,
		},
		{
			name: "with custom config",
			config: &config.ConsensusConfig{
				MinStakeRequired:    big.NewInt(5000),
				BlockTime:           5 * time.Second,
				EpochLength:         50,
				ProposerHistorySize: 20,
				InitialReputation:   5.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewEngine(tt.config)

			if engine == nil {
				t.Fatal("engine should not be nil")
			}
			if engine.validators == nil {
				t.Error("validators map should be initialized")
			}
			if engine.config == nil {
				t.Error("config should not be nil")
			}
			if engine.qualityAnalyzer == nil {
				t.Error("quality analyzer should be initialized")
			}
			if engine.reputationTracker == nil {
				t.Error("reputation tracker should be initialized")
			}
			if engine.metricsCalculator == nil {
				t.Error("metrics calculator should be initialized")
			}
			if engine.currentEpoch != 0 {
				t.Errorf("initial epoch should be 0, got %d", engine.currentEpoch)
			}
		})
	}
}

func TestEngine_RegisterValidator_Success(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	cfg.MinStakeRequired = big.NewInt(1000)
	engine := NewEngine(cfg)

	tests := []struct {
		name    string
		address string
		stake   *big.Int
	}{
		{
			name:    "register with minimum stake",
			address: "validator1",
			stake:   big.NewInt(1000),
		},
		{
			name:    "register with above minimum stake",
			address: "validator2",
			stake:   big.NewInt(5000),
		},
		{
			name:    "register with large stake",
			address: "validator3",
			stake:   big.NewInt(1000000),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := engine.RegisterValidator(tt.address, tt.stake)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify validator was registered
			v, exists := engine.validators[tt.address]
			if !exists {
				t.Fatal("validator should exist after registration")
			}
			if v.Address != tt.address {
				t.Errorf("address mismatch: expected %s, got %s", tt.address, v.Address)
			}
			if v.TokenStake.Cmp(tt.stake) != 0 {
				t.Errorf("stake mismatch: expected %v, got %v", tt.stake, v.TokenStake)
			}
			if !v.IsActive {
				t.Error("validator should be active after registration")
			}
			if v.ReputationScore != cfg.InitialReputation {
				t.Errorf("reputation should be %f, got %f", cfg.InitialReputation, v.ReputationScore)
			}
		})
	}
}

func TestEngine_RegisterValidator_InsufficientStake(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	cfg.MinStakeRequired = big.NewInt(1000)
	engine := NewEngine(cfg)

	tests := []struct {
		name    string
		address string
		stake   *big.Int
	}{
		{
			name:    "below minimum stake",
			address: "validator1",
			stake:   big.NewInt(999),
		},
		{
			name:    "zero stake",
			address: "validator2",
			stake:   big.NewInt(0),
		},
		{
			name:    "negative stake",
			address: "validator3",
			stake:   big.NewInt(-1000),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := engine.RegisterValidator(tt.address, tt.stake)
			if err == nil {
				t.Fatal("expected error for insufficient stake")
			}

			// Verify validator was NOT registered
			_, exists := engine.validators[tt.address]
			if exists {
				t.Error("validator should not exist after failed registration")
			}
		})
	}
}

func TestEngine_SubmitContribution(t *testing.T) {
	engine := NewEngine(nil)

	// Register test validators
	engine.RegisterValidator("validator1", big.NewInt(10000))
	engine.RegisterValidator("validator2", big.NewInt(20000))

	tests := []struct {
		name          string
		validatorAddr string
		contribution  contribution.Contribution
		expectError   bool
	}{
		{
			name:          "valid code contribution",
			validatorAddr: "validator1",
			contribution: contribution.Contribution{
				ID:            "contrib1",
				Type:          contribution.CodeCommit,
				Timestamp:     time.Now(),
				LinesAdded:    150,
				LinesModified: 50,
				TestCoverage:  85.0,
				Complexity:    6.5,
			},
			expectError: false,
		},
		{
			name:          "valid documentation contribution",
			validatorAddr: "validator2",
			contribution: contribution.Contribution{
				ID:            "contrib2",
				Type:          contribution.Documentation,
				Timestamp:     time.Now(),
				Documentation: 95.0,
			},
			expectError: false,
		},
		{
			name:          "contribution to non-existent validator",
			validatorAddr: "nonexistent",
			contribution: contribution.Contribution{
				ID:        "contrib3",
				Type:      contribution.CodeCommit,
				Timestamp: time.Now(),
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := engine.SubmitContribution(tt.validatorAddr, tt.contribution)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}

				// Verify contribution was added
				v := engine.validators[tt.validatorAddr]
				found := false
				for _, c := range v.RecentContribs {
					if c.ID == tt.contribution.ID {
						found = true
						// Verify quality score was calculated
						if c.QualityScore == 0 {
							t.Error("quality score should be calculated")
						}
						break
					}
				}
				if !found {
					t.Error("contribution not found in validator's recent contributions")
				}
			}
		})
	}
}

func TestEngine_SelectBlockProposer(t *testing.T) {
	engine := NewEngine(nil)

	// Test with no validators
	_, err := engine.SelectBlockProposer()
	if err == nil {
		t.Error("expected error when no validators registered")
	}

	// Register validators with different stakes
	validators := []struct {
		address string
		stake   *big.Int
	}{
		{"validator1", big.NewInt(10000)},
		{"validator2", big.NewInt(20000)},
		{"validator3", big.NewInt(30000)},
		{"validator4", big.NewInt(40000)},
	}

	for _, v := range validators {
		engine.RegisterValidator(v.address, v.stake)
	}

	// Add contributions to affect selection weights
	engine.SubmitContribution("validator2", contribution.Contribution{
		ID:           "high-quality",
		Type:         contribution.CodeCommit,
		Timestamp:    time.Now(),
		LinesAdded:   1000,
		TestCoverage: 95.0,
		Documentation: 90.0,
	})

	// Test selection distribution
	selectionCount := make(map[string]int)
	iterations := 1000

	for i := 0; i < iterations; i++ {
		proposer, err := engine.SelectBlockProposer()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		selectionCount[proposer]++
	}

	// Verify all validators were selected at least once
	for _, v := range validators {
		count := selectionCount[v.address]
		if count == 0 {
			t.Errorf("validator %s was never selected", v.address)
		}
		t.Logf("Validator %s selected %d times (%.1f%%)",
			v.address, count, float64(count)/float64(iterations)*100)
	}

	// Verify higher stake validators are selected more frequently
	if selectionCount["validator4"] < selectionCount["validator1"] {
		t.Error("validator with 4x stake should be selected more frequently")
	}
}

func TestEngine_ConcurrentOperations(t *testing.T) {
	engine := NewEngine(nil)

	// Number of concurrent operations
	numValidators := 50
	numContributions := 10

	// Register validators concurrently
	var wg sync.WaitGroup
	wg.Add(numValidators)

	for i := 0; i < numValidators; i++ {
		go func(id int) {
			defer wg.Done()
			addr := fmt.Sprintf("validator%d", id)
			stake := big.NewInt(int64(10000 + id*1000))
			err := engine.RegisterValidator(addr, stake)
			if err != nil {
				t.Errorf("failed to register validator %s: %v", addr, err)
			}
		}(i)
	}

	wg.Wait()

	// Verify all validators registered
	if len(engine.validators) != numValidators {
		t.Errorf("expected %d validators, got %d", numValidators, len(engine.validators))
	}

	// Submit contributions concurrently
	wg.Add(numValidators * numContributions)

	for i := 0; i < numValidators; i++ {
		for j := 0; j < numContributions; j++ {
			go func(valId, contribId int) {
				defer wg.Done()
				addr := fmt.Sprintf("validator%d", valId)
				contrib := contribution.Contribution{
					ID:         fmt.Sprintf("contrib_%d_%d", valId, contribId),
					Type:       contribution.CodeCommit,
					Timestamp:  time.Now(),
					LinesAdded: valId + contribId,
				}
				engine.SubmitContribution(addr, contrib)
			}(i, j)
		}
	}

	wg.Wait()

	// Verify contributions were added
	totalContribs := 0
	for _, v := range engine.validators {
		totalContribs += len(v.RecentContribs)
	}

	expectedContribs := numValidators * numContributions
	if totalContribs != expectedContribs {
		t.Errorf("expected %d total contributions, got %d", expectedContribs, totalContribs)
	}
}

func TestEngine_SlashValidator(t *testing.T) {
	engine := NewEngine(nil)

	// Register test validator
	addr := "validator1"
	initialStake := big.NewInt(10000)
	engine.RegisterValidator(addr, initialStake)

	v := engine.validators[addr]
	initialReputation := v.ReputationScore

	// Test slashing
	err := engine.SlashValidator(addr, validator.MaliciousCode, "detected malicious behavior")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify reputation was reduced
	if v.ReputationScore >= initialReputation {
		t.Error("reputation should be reduced after slashing")
	}

	// Verify slashing history
	if len(v.SlashingHistory) != 1 {
		t.Errorf("expected 1 slashing event, got %d", len(v.SlashingHistory))
	}

	if v.SlashingHistory[0].Evidence != "detected malicious behavior" {
		t.Error("slashing evidence not recorded correctly")
	}

	// Test slashing non-existent validator
	err = engine.SlashValidator("nonexistent", validator.FalseContribution, "test")
	if err == nil {
		t.Error("expected error when slashing non-existent validator")
	}
}

// Benchmark tests
func BenchmarkEngine_RegisterValidator(b *testing.B) {
	engine := NewEngine(nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addr := fmt.Sprintf("validator%d", i)
		stake := big.NewInt(int64(10000 + i))
		engine.RegisterValidator(addr, stake)
	}
}

func BenchmarkEngine_SelectBlockProposer(b *testing.B) {
	engine := NewEngine(nil)

	// Pre-register 100 validators
	for i := 0; i < 100; i++ {
		addr := fmt.Sprintf("validator%d", i)
		stake := big.NewInt(int64(10000 + i*1000))
		engine.RegisterValidator(addr, stake)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.SelectBlockProposer()
	}
}

func BenchmarkEngine_SubmitContribution(b *testing.B) {
	engine := NewEngine(nil)
	engine.RegisterValidator("validator1", big.NewInt(10000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contrib := contribution.Contribution{
			ID:         fmt.Sprintf("contrib%d", i),
			Type:       contribution.CodeCommit,
			Timestamp:  time.Now(),
			LinesAdded: i,
		}
		engine.SubmitContribution("validator1", contrib)
	}
}