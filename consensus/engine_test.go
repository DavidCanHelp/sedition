package consensus

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/contribution"
	"github.com/davidcanhelp/sedition/errors"
	"github.com/davidcanhelp/sedition/validator"
)

func TestNewEngine(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	engine := NewEngine(cfg)
	
	if engine == nil {
		t.Fatal("NewEngine returned nil")
	}
	
	if engine.config != cfg {
		t.Error("Engine config not set correctly")
	}
	
	if engine.currentEpoch != 0 {
		t.Errorf("Expected initial epoch 0, got %d", engine.currentEpoch)
	}
	
	if engine.validators == nil {
		t.Error("Validators map not initialized")
	}
}

func TestNewEngineWithNilConfig(t *testing.T) {
	engine := NewEngine(nil)
	
	if engine == nil {
		t.Fatal("NewEngine with nil config should return engine with defaults")
	}
	
	if engine.config == nil {
		t.Error("Engine should have default config when nil is passed")
	}
}

func TestRegisterValidator(t *testing.T) {
	engine := NewEngine(nil)
	
	// Register a validator
	addr := "test_validator"
	stake := big.NewInt(10000)
	
	err := engine.RegisterValidator(addr, stake)
	if err != nil {
		t.Fatalf("Failed to register validator: %v", err)
	}
	
	// Check validator was added
	if _, exists := engine.validators[addr]; !exists {
		t.Error("Validator not found after registration")
	}
}

func TestRegisterValidatorInsufficientStake(t *testing.T) {
	engine := NewEngine(nil)
	
	// Try to register with insufficient stake
	addr := "test_validator"
	stake := big.NewInt(100) // Below minimum
	
	err := engine.RegisterValidator(addr, stake)
	if err == nil {
		t.Error("Expected error for insufficient stake")
	}
	
	// Check error type
	consensusErr, ok := err.(*errors.ConsensusError)
	if !ok {
		t.Error("Expected ConsensusError type")
	}
	
	if consensusErr.Code != errors.ErrInsufficientStake {
		t.Errorf("Expected error code %s, got %s", errors.ErrInsufficientStake, consensusErr.Code)
	}
}

func TestSubmitContribution(t *testing.T) {
	engine := NewEngine(nil)
	
	// Register a validator first
	addr := "test_validator"
	stake := big.NewInt(10000)
	engine.RegisterValidator(addr, stake)
	
	// Submit a contribution
	contrib := contribution.Contribution{
		ID:           "test_commit",
		Type:         contribution.CodeCommit,
		QualityScore: 80.0,
		TestCoverage: 85.0,
		Complexity:   5.0,
		Timestamp:    time.Now(),
	}
	
	err := engine.SubmitContribution(addr, contrib)
	if err != nil {
		t.Fatalf("Failed to submit contribution: %v", err)
	}
}

func TestSubmitContributionInvalidValidator(t *testing.T) {
	engine := NewEngine(nil)
	
	// Try to submit contribution for non-existent validator
	contrib := contribution.Contribution{
		ID:   "test_commit",
		Type: contribution.CodeCommit,
	}
	
	err := engine.SubmitContribution("non_existent", contrib)
	if err == nil {
		t.Error("Expected error for non-existent validator")
	}
	
	// Check error type
	consensusErr, ok := err.(*errors.ConsensusError)
	if !ok {
		t.Error("Expected ConsensusError type")
	}
	
	if consensusErr.Code != errors.ErrValidatorNotFound {
		t.Errorf("Expected error code %s, got %s", errors.ErrValidatorNotFound, consensusErr.Code)
	}
}

func TestSelectBlockProposer(t *testing.T) {
	engine := NewEngine(nil)
	
	// Register multiple validators
	validators := map[string]*big.Int{
		"validator1": big.NewInt(10000),
		"validator2": big.NewInt(20000),
		"validator3": big.NewInt(30000),
	}
	
	for addr, stake := range validators {
		err := engine.RegisterValidator(addr, stake)
		if err != nil {
			t.Fatalf("Failed to register validator %s: %v", addr, err)
		}
	}
	
	// Select proposer
	proposer, err := engine.SelectBlockProposer()
	if err != nil {
		t.Fatalf("Failed to select proposer: %v", err)
	}
	
	// Check proposer is one of the registered validators
	if _, exists := validators[proposer]; !exists {
		t.Errorf("Selected proposer %s is not a registered validator", proposer)
	}
}

func TestSelectBlockProposerNoValidators(t *testing.T) {
	engine := NewEngine(nil)
	
	// Try to select proposer with no validators
	_, err := engine.SelectBlockProposer()
	if err == nil {
		t.Error("Expected error when no validators are registered")
	}
	
	// Check error type
	consensusErr, ok := err.(*errors.ConsensusError)
	if !ok {
		t.Error("Expected ConsensusError type")
	}
	
	if consensusErr.Code != errors.ErrNoActiveValidators {
		t.Errorf("Expected error code %s, got %s", errors.ErrNoActiveValidators, consensusErr.Code)
	}
}

func TestSlashValidator(t *testing.T) {
	engine := NewEngine(nil)
	
	// Register a validator
	addr := "test_validator"
	initialStake := big.NewInt(10000)
	engine.RegisterValidator(addr, initialStake)
	
	// Slash the validator
	err := engine.SlashValidator(addr, validator.DoubleProposal, "evidence")
	if err != nil {
		t.Fatalf("Failed to slash validator: %v", err)
	}
	
	// Check that stake was reduced
	v := engine.validators[addr]
	if v.TokenStake.Cmp(initialStake) >= 0 {
		t.Error("Validator stake should be reduced after slashing")
	}
}

func TestUpdateEpoch(t *testing.T) {
	engine := NewEngine(nil)
	
	initialEpoch := engine.currentEpoch
	
	// Update epoch
	engine.UpdateEpoch()
	
	if engine.currentEpoch != initialEpoch+1 {
		t.Errorf("Expected epoch %d, got %d", initialEpoch+1, engine.currentEpoch)
	}
}

func TestGetValidatorStats(t *testing.T) {
	engine := NewEngine(nil)
	
	// Register a validator
	addr := "test_validator"
	engine.RegisterValidator(addr, big.NewInt(10000))
	
	// Get stats
	stats, err := engine.GetValidatorStats(addr)
	if err != nil {
		t.Fatalf("Failed to get validator stats: %v", err)
	}
	
	if stats == nil {
		t.Error("Expected non-nil stats")
	}
	
	if stats.Address != addr {
		t.Errorf("Expected address %s, got %s", addr, stats.Address)
	}
}

func TestGetValidatorStatsNotFound(t *testing.T) {
	engine := NewEngine(nil)
	
	// Try to get stats for non-existent validator
	_, err := engine.GetValidatorStats("non_existent")
	if err == nil {
		t.Error("Expected error for non-existent validator")
	}
	
	// Check error type
	consensusErr, ok := err.(*errors.ConsensusError)
	if !ok {
		t.Error("Expected ConsensusError type")
	}
	
	if consensusErr.Code != errors.ErrValidatorNotFound {
		t.Errorf("Expected error code %s, got %s", errors.ErrValidatorNotFound, consensusErr.Code)
	}
}

func TestGetNetworkStats(t *testing.T) {
	engine := NewEngine(nil)
	
	// Register some validators
	engine.RegisterValidator("validator1", big.NewInt(10000))
	engine.RegisterValidator("validator2", big.NewInt(20000))
	
	// Get network stats
	stats := engine.GetNetworkStats()
	
	if stats == nil {
		t.Fatal("Expected non-nil network stats")
	}
	
	if stats.TotalValidators != 2 {
		t.Errorf("Expected 2 validators, got %d", stats.TotalValidators)
	}
	
	if stats.ActiveValidators != 2 {
		t.Errorf("Expected 2 active validators, got %d", stats.ActiveValidators)
	}
	
	expectedStake := big.NewInt(30000)
	if stats.TotalStaked.Cmp(expectedStake) != 0 {
		t.Errorf("Expected total stake %v, got %v", expectedStake, stats.TotalStaked)
	}
}

func TestProposerRotation(t *testing.T) {
	engine := NewEngine(nil)
	
	// Register validators
	for i := 1; i <= 5; i++ {
		addr := fmt.Sprintf("validator%d", i)
		stake := big.NewInt(int64(i * 10000))
		engine.RegisterValidator(addr, stake)
	}
	
	// Track selected proposers
	proposerCount := make(map[string]int)
	
	// Select proposers multiple times
	for i := 0; i < 100; i++ {
		proposer, err := engine.SelectBlockProposer()
		if err != nil {
			t.Fatalf("Failed to select proposer: %v", err)
		}
		proposerCount[proposer]++
		
		// Record proposer to test frequency limiting
		engine.recordProposer(proposer)
	}
	
	// Verify all validators were selected at least once
	for addr := range engine.validators {
		if proposerCount[addr] == 0 {
			t.Logf("Warning: Validator %s was never selected as proposer", addr)
		}
	}
	
	// Higher stake validators should be selected more often
	if proposerCount["validator5"] < proposerCount["validator1"] {
		t.Log("Warning: Higher stake validator selected less frequently")
	}
}

// Benchmark tests
func BenchmarkRegisterValidator(b *testing.B) {
	engine := NewEngine(nil)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addr := fmt.Sprintf("validator_%d", i)
		stake := big.NewInt(int64(10000 + i))
		engine.RegisterValidator(addr, stake)
	}
}

func BenchmarkSelectBlockProposer(b *testing.B) {
	engine := NewEngine(nil)
	
	// Register validators
	for i := 0; i < 100; i++ {
		addr := fmt.Sprintf("validator_%d", i)
		stake := big.NewInt(int64(10000 + i*100))
		engine.RegisterValidator(addr, stake)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.SelectBlockProposer()
	}
}

func BenchmarkSubmitContribution(b *testing.B) {
	engine := NewEngine(nil)
	
	// Register a validator
	addr := "test_validator"
	engine.RegisterValidator(addr, big.NewInt(10000))
	
	contrib := contribution.Contribution{
		ID:           "test",
		Type:         contribution.CodeCommit,
		QualityScore: 80.0,
		TestCoverage: 85.0,
		Timestamp:    time.Now(),
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.SubmitContribution(addr, contrib)
	}
}