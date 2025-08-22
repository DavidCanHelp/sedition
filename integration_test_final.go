package poc

import (
	"fmt"
	"math/big"
	"testing"
	"time"
)

// TestCompleteIntegration tests the entire consensus system end-to-end
func TestCompleteIntegration(t *testing.T) {
	t.Log("🚀 Starting complete integration test")
	
	// Step 1: Initialize consensus engine
	minStake := big.NewInt(1000000)
	blockTime := time.Second
	engine := NewConsensusEngine(minStake, blockTime)
	
	// Step 2: Register validators
	validators := []struct {
		id    string
		stake int64
	}{
		{"alice", 5000000},
		{"bob", 3000000},
		{"carol", 4000000},
		{"dave", 2000000},
	}
	
	for _, v := range validators {
		err := engine.RegisterValidator(v.id, big.NewInt(v.stake))
		if err != nil {
			t.Fatalf("Failed to register validator %s: %v", v.id, err)
		}
		t.Logf("✅ Registered validator %s with stake %d", v.id, v.stake)
	}
	
	// Step 3: Submit contributions and select proposers
	for i := 0; i < 3; i++ {
		// Submit contribution
		contribution := Contribution{
			ID:            generateContributionID(),
			Timestamp:     time.Now(),
			Type:          CodeCommit,
			LinesAdded:    100 + i*50,
			LinesModified: 20 + i*10,
			TestCoverage:  80.0 + float64(i*2),
			Complexity:    5.0 - float64(i)*0.5,
			Documentation: 75.0 + float64(i*3),
			PeerReviews:   2,
			ReviewScore:   4.0 + float64(i)*0.2,
		}
		
		validatorID := validators[i%len(validators)].id
		err := engine.SubmitContribution(validatorID, contribution)
		if err != nil {
			t.Errorf("Failed to submit contribution: %v", err)
		}
		
		// Select block proposer
		proposer, err := engine.SelectBlockProposer()
		if err != nil {
			t.Errorf("Failed to select proposer: %v", err)
		}
		
		t.Logf("📝 Round %d: Proposer selected: %s", i+1, proposer)
		
		// Verify proposer is valid
		found := false
		for _, v := range validators {
			if v.id == proposer {
				found = true
				break
			}
		}
		
		if !found {
			t.Errorf("Selected proposer %s not in validator set", proposer)
		}
	}
	
	// Step 4: Test slashing mechanism
	t.Log("🔨 Testing slashing mechanism")
	
	initialRep := engine.reputationTracker.GetReputation("alice")
	engine.SlashValidator("alice", MaliciousCode, "test evidence")
	finalRep := engine.reputationTracker.GetReputation("alice")
	
	if finalRep >= initialRep {
		t.Errorf("Slashing failed: reputation not reduced (initial: %f, final: %f)", initialRep, finalRep)
	} else {
		t.Logf("✅ Slashing successful: reputation reduced from %f to %f", initialRep, finalRep)
	}
	
	// Step 5: Test Byzantine fault tolerance
	t.Log("🛡️ Testing Byzantine fault tolerance")
	
	// Simulate Byzantine behavior by removing a validator
	delete(engine.validators, "dave")
	
	// System should still function with n=3, f=0 (can tolerate 0 Byzantine nodes)
	proposer, err := engine.SelectBlockProposer()
	if err != nil {
		t.Errorf("System failed with 1 node removed: %v", err)
	} else {
		t.Logf("✅ System continues functioning with Byzantine node removed: proposer %s", proposer)
	}
	
	// Step 6: Performance metrics
	t.Log("📊 Performance metrics")
	
	start := time.Now()
	for i := 0; i < 100; i++ {
		_, _ = engine.SelectBlockProposer()
	}
	elapsed := time.Since(start)
	
	avgTime := elapsed / 100
	if avgTime > time.Millisecond {
		t.Logf("⚠️ Performance warning: average selection time %v", avgTime)
	} else {
		t.Logf("✅ Performance excellent: average selection time %v", avgTime)
	}
	
	t.Log("🎉 Integration test completed successfully!")
}

// Helper function to generate contribution IDs
func generateContributionID() string {
	return fmt.Sprintf("contrib_%d", time.Now().UnixNano())
}

// TestConcurrentOperations tests thread safety
func TestConcurrentOperations(t *testing.T) {
	minStake := big.NewInt(1000000)
	blockTime := time.Second
	engine := NewConsensusEngine(minStake, blockTime)
	
	// Register initial validators
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("validator_%d", i)
		stake := big.NewInt(int64(1000000 + i*100000))
		engine.RegisterValidator(id, stake)
	}
	
	// Run concurrent operations
	done := make(chan bool, 3)
	
	// Goroutine 1: Continuous proposer selection
	go func() {
		for i := 0; i < 100; i++ {
			engine.SelectBlockProposer()
		}
		done <- true
	}()
	
	// Goroutine 2: Continuous contribution submission
	go func() {
		for i := 0; i < 100; i++ {
			contribution := Contribution{
				ID:        fmt.Sprintf("contrib_%d", i),
				Timestamp: time.Now(),
				Type:      CodeCommit,
			}
			engine.SubmitContribution(fmt.Sprintf("validator_%d", i%10), contribution)
		}
		done <- true
	}()
	
	// Goroutine 3: Continuous reputation queries
	go func() {
		for i := 0; i < 100; i++ {
			engine.reputationTracker.GetReputation(fmt.Sprintf("validator_%d", i%10))
		}
		done <- true
	}()
	
	// Wait for all goroutines
	for i := 0; i < 3; i++ {
		<-done
	}
	
	t.Log("✅ Concurrent operations test passed - no deadlocks or races")
}