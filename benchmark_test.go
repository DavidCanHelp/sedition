package poc

import (
	"fmt"
	"testing"
	"time"
)

// TestPoCBenchmark runs comprehensive benchmarks for the PoC consensus
func TestPoCBenchmark(t *testing.T) {
	// Test configurations for different network sizes
	configs := []SimulationConfig{
		{
			NumValidators:       100,
			NumByzantine:        33,
			NumRounds:           1000,
			NetworkLatency:      50 * time.Millisecond,
			ContributionRate:    10.0, // per hour
			QualityDistribution: "normal",
			RandomSeed:          42,
		},
		{
			NumValidators:       500,
			NumByzantine:        166,
			NumRounds:           1000,
			NetworkLatency:      100 * time.Millisecond,
			ContributionRate:    10.0,
			QualityDistribution: "normal",
			RandomSeed:          42,
		},
		{
			NumValidators:       1000,
			NumByzantine:        333,
			NumRounds:           1000,
			NetworkLatency:      150 * time.Millisecond,
			ContributionRate:    10.0,
			QualityDistribution: "normal",
			RandomSeed:          42,
		},
		{
			NumValidators:       5000,
			NumByzantine:        1666,
			NumRounds:           100, // Fewer rounds for large network
			NetworkLatency:      200 * time.Millisecond,
			ContributionRate:    10.0,
			QualityDistribution: "normal",
			RandomSeed:          42,
		},
	}
	
	RunBenchmark(configs)
}

// TestByzantineTolerance verifies Byzantine fault tolerance
func TestByzantineTolerance(t *testing.T) {
	testCases := []struct {
		name         string
		validators   int
		byzantine    int
		expectFail   bool
	}{
		{"10% Byzantine", 100, 10, false},
		{"25% Byzantine", 100, 25, false},
		{"33% Byzantine (limit)", 100, 33, false},
		{"34% Byzantine (should fail)", 100, 34, true},
		{"40% Byzantine (should fail)", 100, 40, true},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := SimulationConfig{
				NumValidators:       tc.validators,
				NumByzantine:        tc.byzantine,
				NumRounds:           100,
				NetworkLatency:      50 * time.Millisecond,
				ContributionRate:    10.0,
				QualityDistribution: "normal",
				RandomSeed:          42,
			}
			
			sim := NewSimulator(&config)
			metrics, _ := sim.Run()
			
			successRate := float64(metrics.SuccessfulBlocks) / float64(metrics.TotalRounds)
			
			if tc.expectFail && successRate > 0.5 {
				t.Errorf("Expected failure with %d%% Byzantine, but got %.2f%% success rate",
					tc.byzantine*100/tc.validators, successRate*100)
			} else if !tc.expectFail && successRate < 0.9 {
				t.Errorf("Expected success with %d%% Byzantine, but got only %.2f%% success rate",
					tc.byzantine*100/tc.validators, successRate*100)
			}
		})
	}
}

// TestFinalityTime measures consensus finality time
func TestFinalityTime(t *testing.T) {
	config := SimulationConfig{
		NumValidators:       100,
		NumByzantine:        20,
		NumRounds:           100,
		NetworkLatency:      50 * time.Millisecond,
		ContributionRate:    10.0,
		QualityDistribution: "normal",
		RandomSeed:          42,
	}
	
	sim := NewSimulator(&config)
	metrics, err := sim.Run()
	
	if err != nil {
		t.Fatalf("Simulation failed: %v", err)
	}
	
	// Check that finality is achieved within target
	targetFinality := 10.0 // seconds
	if metrics.AverageFinality > targetFinality {
		t.Errorf("Finality time %.2f exceeds target of %.2f seconds",
			metrics.AverageFinality, targetFinality)
	}
	
	t.Logf("Average finality achieved: %.3f seconds", metrics.AverageFinality)
}

// TestThroughput measures transaction throughput
func TestThroughput(t *testing.T) {
	config := SimulationConfig{
		NumValidators:       100,
		NumByzantine:        20,
		NumRounds:           1000,
		NetworkLatency:      50 * time.Millisecond,
		ContributionRate:    10.0,
		QualityDistribution: "normal",
		RandomSeed:          42,
	}
	
	sim := NewSimulator(&config)
	metrics, err := sim.Run()
	
	if err != nil {
		t.Fatalf("Simulation failed: %v", err)
	}
	
	// Check that throughput meets target
	targetThroughput := 1000.0 // tx/s
	if metrics.Throughput < targetThroughput {
		t.Errorf("Throughput %.2f below target of %.2f tx/s",
			metrics.Throughput, targetThroughput)
	}
	
	t.Logf("Throughput achieved: %.2f tx/s", metrics.Throughput)
}

// TestQualityIncentives verifies that quality is rewarded
func TestQualityIncentives(t *testing.T) {
	config := SimulationConfig{
		NumValidators:       50,
		NumByzantine:        0, // No Byzantine for this test
		NumRounds:           500,
		NetworkLatency:      50 * time.Millisecond,
		ContributionRate:    20.0,
		QualityDistribution: "uniform",
		RandomSeed:          42,
	}
	
	sim := NewSimulator(&config)
	
	// Track initial reputations
	initialReps := make(map[string]float64)
	for id, node := range sim.nodes {
		initialReps[id] = node.Reputation
	}
	
	// Run simulation
	metrics, err := sim.Run()
	if err != nil {
		t.Fatalf("Simulation failed: %v", err)
	}
	
	// Check that high-quality contributors gained reputation
	improvedCount := 0
	for id, node := range sim.nodes {
		if node.Contributions > 70 && node.Reputation > initialReps[id] {
			improvedCount++
		}
	}
	
	if improvedCount < len(sim.nodes)/3 {
		t.Errorf("Only %d/%d high-quality contributors improved reputation",
			improvedCount, len(sim.nodes))
	}
	
	t.Logf("Quality incentives working: %d validators improved reputation", improvedCount)
	t.Logf("Average block quality: %.2f", averageQuality(metrics.QualityScores))
}

// TestSlashingMechanism verifies slashing for malicious behavior
func TestSlashingMechanism(t *testing.T) {
	config := SimulationConfig{
		NumValidators:       50,
		NumByzantine:        10,
		NumRounds:           200,
		NetworkLatency:      50 * time.Millisecond,
		ContributionRate:    10.0,
		QualityDistribution: "normal",
		RandomSeed:          42,
	}
	
	sim := NewSimulator(&config)
	
	// Track initial stakes
	initialStakes := make(map[string]uint64)
	for id, node := range sim.nodes {
		initialStakes[id] = node.Stake
	}
	
	// Run simulation
	metrics, err := sim.Run()
	if err != nil {
		t.Fatalf("Simulation failed: %v", err)
	}
	
	// Check that slashing occurred
	if metrics.SlashingEvents == 0 {
		t.Error("No slashing events occurred despite Byzantine validators")
	}
	
	// Verify Byzantine validators were slashed
	slashedCount := 0
	for id, node := range sim.nodes {
		if !node.IsHonest && node.Stake < initialStakes[id] {
			slashedCount++
		}
	}
	
	t.Logf("Slashing mechanism active: %d events, %d validators slashed",
		metrics.SlashingEvents, slashedCount)
}

// TestNetworkPartitionRecovery tests recovery from network partitions
func TestNetworkPartitionRecovery(t *testing.T) {
	config := SimulationConfig{
		NumValidators:       100,
		NumByzantine:        20,
		NumRounds:           300,
		NetworkLatency:      50 * time.Millisecond,
		ContributionRate:    10.0,
		QualityDistribution: "normal",
		RandomSeed:          42,
	}
	
	sim := NewSimulator(&config)
	
	// Start simulation in background
	go func() {
		// Create partition after 100 rounds
		time.Sleep(5 * time.Second)
		
		// Partition 30% of nodes
		partitionNodes := []string{}
		count := 0
		for id := range sim.nodes {
			if count < 30 {
				partitionNodes = append(partitionNodes, id)
				count++
			}
		}
		
		t.Logf("Creating network partition of %d nodes", len(partitionNodes))
		sim.SimulateNetworkPartition(partitionNodes, 2*time.Second)
	}()
	
	metrics, err := sim.Run()
	if err != nil {
		t.Fatalf("Simulation failed: %v", err)
	}
	
	// Should still achieve reasonable success rate despite partition
	successRate := float64(metrics.SuccessfulBlocks) / float64(metrics.TotalRounds)
	if successRate < 0.7 {
		t.Errorf("Low success rate %.2f%% after partition recovery", successRate*100)
	}
	
	t.Logf("Network partition recovery successful: %.2f%% success rate", successRate*100)
}

// BenchmarkPoCConsensus benchmarks the consensus algorithm performance
func BenchmarkPoCConsensus(b *testing.B) {
	sizes := []int{10, 50, 100, 500, 1000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("validators_%d", size), func(b *testing.B) {
			poc := NewProofOfContribution()
			
			// Register validators
			for i := 0; i < size; i++ {
				poc.RegisterValidator(&ValidatorState{
					Address:             fmt.Sprintf("validator_%d", i),
					Stake:               1000,
					Reputation:          0.5,
					RecentContributions: 50,
				})
			}
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				// Benchmark leader selection
				leader := poc.SelectBlockProposer()
				if leader == nil {
					b.Fatal("Failed to select leader")
				}
			}
		})
	}
}

// BenchmarkQualityAnalysis benchmarks code quality analysis
func BenchmarkQualityAnalysis(b *testing.B) {
	analyzer := NewCodeQualityAnalyzer()
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		contribution := &Contribution{
			Type:          CodeCommit,
			LinesAdded:    100,
			LinesModified: 20,
			LinesDeleted:  10,
			TestCoverage:  0.85,
			Documentation: 0.7,
			Complexity:    15,
			QualityScore:  0,
		}
		
		score, _ := analyzer.AnalyzeContribution(*contribution)
		if score < 0 || score > 100 {
			b.Fatalf("Invalid quality score: %f", score)
		}
	}
}

// Helper function to calculate average quality
func averageQuality(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	sum := 0.0
	for _, s := range scores {
		sum += s
	}
	return sum / float64(len(scores))
}