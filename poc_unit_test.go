package poc

import (
	"math/big"
	"testing"
	"time"
)

func TestNewConsensusEngine(t *testing.T) {
	minStake := big.NewInt(1000)
	blockTime := 5 * time.Second

	engine := NewConsensusEngine(minStake, blockTime)

	if engine == nil {
		t.Fatal("Expected non-nil consensus engine")
	}

	if engine.validators == nil {
		t.Error("Expected validators map to be initialized")
	}

	if engine.minStakeRequired.Cmp(minStake) != 0 {
		t.Errorf("Expected minStakeRequired to be %v, got %v", minStake, engine.minStakeRequired)
	}

	if engine.blockTime != blockTime {
		t.Errorf("Expected blockTime to be %v, got %v", blockTime, engine.blockTime)
	}

	if engine.epochLength != 100 {
		t.Errorf("Expected epochLength to be 100, got %d", engine.epochLength)
	}

	if engine.slashingRate != 0.1 {
		t.Errorf("Expected slashingRate to be 0.1, got %f", engine.slashingRate)
	}
}

func TestRegisterValidator(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	tests := []struct {
		name          string
		address       string
		stake         *big.Int
		expectError   bool
		errorContains string
	}{
		{
			name:        "Valid registration",
			address:     "validator1",
			stake:       big.NewInt(2000),
			expectError: false,
		},
		{
			name:          "Insufficient stake",
			address:       "validator2",
			stake:         big.NewInt(500),
			expectError:   true,
			errorContains: "insufficient stake",
		},
		{
			name:        "Minimum stake exactly",
			address:     "validator3",
			stake:       big.NewInt(1000),
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := engine.RegisterValidator(tt.address, tt.stake)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got nil")
				} else if tt.errorContains != "" && err.Error() != tt.errorContains+" amount" {
					t.Errorf("Expected error containing '%s', got '%s'", tt.errorContains, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}

				validator, exists := engine.validators[tt.address]
				if !exists {
					t.Error("Expected validator to be registered")
				}

				if validator.TokenStake.Cmp(tt.stake) != 0 {
					t.Errorf("Expected stake %v, got %v", tt.stake, validator.TokenStake)
				}

				if validator.ReputationScore != 5.0 {
					t.Errorf("Expected initial reputation 5.0, got %f", validator.ReputationScore)
				}

				if !validator.IsActive {
					t.Error("Expected validator to be active")
				}
			}
		})
	}
}

func TestCalculateContributionBonus(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	tests := []struct {
		name        string
		validator   *Validator
		expectedMin float64
		expectedMax float64
		description string
	}{
		{
			name: "No contributions",
			validator: &Validator{
				RecentContribs: []Contribution{},
			},
			expectedMin: 0.8,
			expectedMax: 0.8,
			description: "Should apply inactivity penalty",
		},
		{
			name: "Old contributions only",
			validator: &Validator{
				RecentContribs: []Contribution{
					{
						Timestamp:    time.Now().Add(-10 * 24 * time.Hour),
						QualityScore: 80.0,
					},
				},
			},
			expectedMin: 0.9,
			expectedMax: 0.9,
			description: "Should apply small penalty for no recent contributions",
		},
		{
			name: "Recent high quality contributions",
			validator: &Validator{
				RecentContribs: []Contribution{
					{
						Timestamp:    time.Now().Add(-2 * 24 * time.Hour),
						QualityScore: 90.0,
					},
					{
						Timestamp:    time.Now().Add(-1 * 24 * time.Hour),
						QualityScore: 85.0,
					},
				},
			},
			expectedMin: 1.0,
			expectedMax: 1.5,
			description: "Should apply bonus for quality contributions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bonus := engine.calculateContributionBonus(tt.validator)

			if bonus < tt.expectedMin || bonus > tt.expectedMax {
				t.Errorf("%s: Expected bonus between %f and %f, got %f",
					tt.description, tt.expectedMin, tt.expectedMax, bonus)
			}
		})
	}
}

func TestIsProposerTooFrequent(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	tests := []struct {
		name            string
		proposerHistory []string
		checkAddress    string
		expected        bool
	}{
		{
			name:            "Empty history",
			proposerHistory: []string{},
			checkAddress:    "validator1",
			expected:        false,
		},
		{
			name:            "Short history",
			proposerHistory: []string{"validator1", "validator2"},
			checkAddress:    "validator1",
			expected:        false,
		},
		{
			name:            "Not too frequent",
			proposerHistory: []string{"validator1", "validator2", "validator3", "validator4", "validator5"},
			checkAddress:    "validator1",
			expected:        false,
		},
		{
			name:            "Too frequent - twice in last 3",
			proposerHistory: []string{"validator1", "validator2", "validator1", "validator2", "validator1"},
			checkAddress:    "validator1",
			expected:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine.proposerHistory = tt.proposerHistory
			result := engine.isProposerTooFrequent(tt.checkAddress)

			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestRecordProposer(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	// Test adding to empty history
	engine.recordProposer("validator1")
	if len(engine.proposerHistory) != 1 {
		t.Errorf("Expected history length 1, got %d", len(engine.proposerHistory))
	}
	if engine.proposerHistory[0] != "validator1" {
		t.Errorf("Expected first proposer to be validator1, got %s", engine.proposerHistory[0])
	}

	// Test history limit
	for i := 2; i <= 25; i++ {
		engine.recordProposer("validator" + string(rune(i)))
	}

	if len(engine.proposerHistory) != 20 {
		t.Errorf("Expected history to be limited to 20, got %d", len(engine.proposerHistory))
	}
}

func TestGetActiveValidators(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	// Register validators with different states
	engine.validators["active1"] = &Validator{
		Address:    "active1",
		TokenStake: big.NewInt(2000),
		TotalStake: big.NewInt(2000),
		IsActive:   true,
	}

	engine.validators["inactive"] = &Validator{
		Address:    "inactive",
		TokenStake: big.NewInt(2000),
		TotalStake: big.NewInt(2000),
		IsActive:   false,
	}

	engine.validators["insufficient"] = &Validator{
		Address:    "insufficient",
		TokenStake: big.NewInt(500),
		TotalStake: big.NewInt(500),
		IsActive:   true,
	}

	engine.validators["zero_total"] = &Validator{
		Address:    "zero_total",
		TokenStake: big.NewInt(2000),
		TotalStake: big.NewInt(0),
		IsActive:   true,
	}

	active := engine.getActiveValidators()

	if len(active) != 1 {
		t.Errorf("Expected 1 active validator, got %d", len(active))
	}

	if _, exists := active["active1"]; !exists {
		t.Error("Expected active1 to be in active validators")
	}
}

func TestSlashValidator(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	// Register a validator
	initialStake := big.NewInt(10000)
	engine.RegisterValidator("validator1", initialStake)

	validator := engine.validators["validator1"]
	initialRep := validator.ReputationScore

	// Slash the validator
	err := engine.SlashValidator("validator1", MaliciousCode, "test evidence")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Check stake was reduced by 10%
	expectedStake := big.NewInt(9000)
	if validator.TokenStake.Cmp(expectedStake) != 0 {
		t.Errorf("Expected stake %v after slashing, got %v", expectedStake, validator.TokenStake)
	}

	// Check slashing history
	if len(validator.SlashingHistory) != 1 {
		t.Errorf("Expected 1 slashing event, got %d", len(validator.SlashingHistory))
	}

	if validator.SlashingHistory[0].Reason != MaliciousCode {
		t.Error("Slashing reason mismatch")
	}

	// Reputation should be reduced
	if validator.ReputationScore >= initialRep {
		t.Error("Expected reputation to decrease after slashing")
	}

	// Test slashing non-existent validator
	err = engine.SlashValidator("nonexistent", MaliciousCode, "test")
	if err == nil {
		t.Error("Expected error for non-existent validator")
	}

	// Test deactivation on low stake
	engine.validators["validator1"].TokenStake = big.NewInt(1100)
	err = engine.SlashValidator("validator1", MaliciousCode, "test evidence 2")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if validator.IsActive {
		t.Error("Expected validator to be deactivated due to insufficient stake")
	}
}

func TestUpdateEpoch(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	// Register validator with contributions
	engine.RegisterValidator("validator1", big.NewInt(2000))
	validator := engine.validators["validator1"]

	// Add old and recent contributions
	validator.RecentContribs = []Contribution{
		{
			Timestamp:    time.Now().Add(-time.Duration(engine.epochLength+10) * engine.blockTime),
			QualityScore: 50.0,
		},
		{
			Timestamp:    time.Now().Add(-time.Duration(engine.epochLength-10) * engine.blockTime),
			QualityScore: 80.0,
		},
		{
			Timestamp:    time.Now(),
			QualityScore: 90.0,
		},
	}

	initialEpoch := engine.currentEpoch

	engine.UpdateEpoch()

	// Check epoch incremented
	if engine.currentEpoch != initialEpoch+1 {
		t.Errorf("Expected epoch %d, got %d", initialEpoch+1, engine.currentEpoch)
	}

	// Check old contributions removed
	if len(validator.RecentContribs) != 2 {
		t.Errorf("Expected 2 recent contributions after cleanup, got %d", len(validator.RecentContribs))
	}

	// Verify only recent contributions remain
	for _, contrib := range validator.RecentContribs {
		if contrib.QualityScore == 50.0 {
			t.Error("Old contribution should have been removed")
		}
	}
}

func TestGetValidatorStats(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	// Test non-existent validator
	_, err := engine.GetValidatorStats("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent validator")
	}

	// Register validator
	stake := big.NewInt(2000)
	engine.RegisterValidator("validator1", stake)
	validator := engine.validators["validator1"]

	// Add contributions
	validator.RecentContribs = []Contribution{
		{QualityScore: 80.0},
		{QualityScore: 90.0},
	}

	// Add slashing event
	validator.SlashingHistory = []SlashingEvent{{}}

	stats, err := engine.GetValidatorStats("validator1")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if stats.Address != "validator1" {
		t.Errorf("Expected address validator1, got %s", stats.Address)
	}

	if stats.TokenStake.Cmp(stake) != 0 {
		t.Errorf("Expected stake %v, got %v", stake, stats.TokenStake)
	}

	if stats.ContributionCount != 2 {
		t.Errorf("Expected 2 contributions, got %d", stats.ContributionCount)
	}

	expectedAvg := 85.0
	if stats.AverageQuality != expectedAvg {
		t.Errorf("Expected average quality %f, got %f", expectedAvg, stats.AverageQuality)
	}

	if stats.SlashingCount != 1 {
		t.Errorf("Expected 1 slashing event, got %d", stats.SlashingCount)
	}
}

func TestGetNetworkStats(t *testing.T) {
	engine := NewConsensusEngine(big.NewInt(1000), 5*time.Second)

	// Register multiple validators
	engine.RegisterValidator("validator1", big.NewInt(2000))
	engine.RegisterValidator("validator2", big.NewInt(3000))

	// Deactivate one
	engine.validators["validator2"].IsActive = false

	stats := engine.GetNetworkStats()

	if stats.TotalValidators != 2 {
		t.Errorf("Expected 2 total validators, got %d", stats.TotalValidators)
	}

	if stats.ActiveValidators != 1 {
		t.Errorf("Expected 1 active validator, got %d", stats.ActiveValidators)
	}

	expectedStake := big.NewInt(2000)
	if stats.TotalStaked.Cmp(expectedStake) != 0 {
		t.Errorf("Expected total stake %v, got %v", expectedStake, stats.TotalStaked)
	}

	if stats.CurrentEpoch != 0 {
		t.Errorf("Expected epoch 0, got %d", stats.CurrentEpoch)
	}
}
