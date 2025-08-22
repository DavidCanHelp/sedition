package poc

import (
	"math/big"
	"testing"
	"time"
)

func TestSimpleConsensus(t *testing.T) {
	// Create basic consensus engine
	minStake := big.NewInt(1000000)
	blockTime := time.Second
	engine := NewConsensusEngine(minStake, blockTime)
	
	// Register validators
	validators := []string{"alice", "bob", "carol", "dave"}
	stakes := []int64{5000000, 3000000, 4000000, 2000000}
	
	for i, validator := range validators {
		err := engine.RegisterValidator(validator, big.NewInt(stakes[i]))
		if err != nil {
			t.Fatalf("Failed to register validator %s: %v", validator, err)
		}
	}
	
	// Test leader selection
	proposer, err := engine.SelectBlockProposer()
	if err != nil {
		t.Fatalf("Failed to select block proposer: %v", err)
	}
	
	// Verify proposer is one of our validators
	found := false
	for _, v := range validators {
		if proposer == v {
			found = true
			break
		}
	}
	
	if !found {
		t.Errorf("Selected proposer %s not in validator set", proposer)
	}
	
	t.Logf("✅ Simple consensus test passed - selected proposer: %s", proposer)
}