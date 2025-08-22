package poc

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// SimulationConfig defines parameters for consensus simulation
type SimulationConfig struct {
	NumValidators      int
	NumByzantine       int
	NumRounds          int
	NetworkLatency     time.Duration
	ContributionRate   float64 // Contributions per hour
	QualityDistribution string  // "uniform", "normal", "power"
	RandomSeed         int64
}

// SimulationMetrics tracks performance metrics
type SimulationMetrics struct {
	TotalRounds          int64
	SuccessfulBlocks     int64
	FailedRounds         int64
	AverageFinality      float64
	Throughput           float64
	ByzantineAttempts    int64
	ByzantineSuccesses   int64
	QualityScores        []float64
	ConsensusTime        []time.Duration
	NetworkUtilization   float64
	ReputationEvolution  map[string][]float64
	SlashingEvents       int64
	ReorganizationDepth  int
	mu                   sync.RWMutex
}

// NetworkSimulator simulates network conditions
type NetworkSimulator struct {
	latency      time.Duration
	packetLoss   float64
	jitter       time.Duration
	partitions   map[string]bool
	mu           sync.RWMutex
}

// SimulationNode represents a validator in simulation
type SimulationNode struct {
	ID           string
	IsHonest     bool
	Stake        uint64
	Reputation   float64
	Contributions float64
	Online       bool
	MessageQueue chan Message
	State        *ValidatorState
}

// Message represents a network message
type Message struct {
	From      string
	To        string
	Type      string
	Content   interface{}
	Timestamp time.Time
}

// Simulator runs PoC consensus simulations
type Simulator struct {
	config   *SimulationConfig
	nodes    map[string]*SimulationNode
	network  *NetworkSimulator
	metrics  *SimulationMetrics
	engine   *ProofOfContribution
	round    int64
	stopped  bool
	mu       sync.RWMutex
	wg       sync.WaitGroup
}

// NewSimulator creates a new consensus simulator
func NewSimulator(config *SimulationConfig) *Simulator {
	rand.Seed(config.RandomSeed)
	
	sim := &Simulator{
		config:  config,
		nodes:   make(map[string]*SimulationNode),
		network: &NetworkSimulator{
			latency:    config.NetworkLatency,
			packetLoss: 0.01, // 1% packet loss
			jitter:     config.NetworkLatency / 10,
			partitions: make(map[string]bool),
		},
		metrics: &SimulationMetrics{
			ReputationEvolution: make(map[string][]float64),
		},
	}
	
	// Create consensus engine first
	sim.engine = NewProofOfContribution()
	
	// Initialize nodes after engine is created
	sim.initializeNodes()
	
	return sim
}

// initializeNodes creates validator nodes for simulation
func (s *Simulator) initializeNodes() {
	byzantineCount := 0
	
	for i := 0; i < s.config.NumValidators; i++ {
		nodeID := fmt.Sprintf("validator_%d", i)
		
		// Determine if Byzantine
		isByzantine := byzantineCount < s.config.NumByzantine
		if isByzantine {
			byzantineCount++
		}
		
		// Create node with random initial state
		node := &SimulationNode{
			ID:           nodeID,
			IsHonest:     !isByzantine,
			Stake:        uint64(1000 + rand.Intn(9000)), // 1000-10000 tokens
			Reputation:   0.5 + rand.Float64()*0.3,       // 0.5-0.8 initial
			Contributions: s.generateQualityScore(),
			Online:       true,
			MessageQueue: make(chan Message, 1000),
			State: &ValidatorState{
				Address:           nodeID,
				Stake:             0, // Will be set from node.Stake
				Reputation:        0, // Will be set from node.Reputation
				RecentContributions: 0, // Will be set from node.Contributions
			},
		}
		
		// Sync state
		node.State.Stake = node.Stake
		node.State.Reputation = node.Reputation
		node.State.RecentContributions = node.Contributions
		
		s.nodes[nodeID] = node
		
		// Register with consensus engine
		s.engine.RegisterValidator(node.State)
	}
}

// generateQualityScore generates a quality score based on distribution
func (s *Simulator) generateQualityScore() float64 {
	switch s.config.QualityDistribution {
	case "normal":
		// Normal distribution centered at 60 with stddev 15
		score := rand.NormFloat64()*15 + 60
		return math.Max(0, math.Min(100, score))
		
	case "power":
		// Power law distribution (few high quality, many low quality)
		score := math.Pow(rand.Float64(), 2) * 100
		return math.Max(0, math.Min(100, score))
		
	default: // uniform
		return rand.Float64() * 100
	}
}

// Run starts the simulation
func (s *Simulator) Run() (*SimulationMetrics, error) {
	fmt.Printf("Starting PoC simulation with %d validators (%d Byzantine)\n", 
		s.config.NumValidators, s.config.NumByzantine)
	
	startTime := time.Now()
	
	// Run consensus rounds
	for round := 0; round < s.config.NumRounds && !s.stopped; round++ {
		roundStart := time.Now()
		
		if err := s.runRound(round); err != nil {
			s.metrics.FailedRounds++
			fmt.Printf("Round %d failed: %v\n", round, err)
		} else {
			s.metrics.SuccessfulBlocks++
		}
		
		s.metrics.TotalRounds++
		s.metrics.ConsensusTime = append(s.metrics.ConsensusTime, time.Since(roundStart))
		
		// Update metrics
		s.updateMetrics()
		
		// Simulate network delays
		time.Sleep(s.network.latency)
		
		// Periodic status update
		if round%100 == 0 && round > 0 {
			s.printStatus(round)
		}
	}
	
	// Calculate final metrics
	s.finalizeMetrics(time.Since(startTime))
	
	return s.metrics, nil
}

// runRound executes a single consensus round
func (s *Simulator) runRound(round int) error {
	s.mu.Lock()
	s.round = int64(round)
	s.mu.Unlock()
	
	// Phase 1: Leader election
	leader := s.engine.SelectBlockProposer()
	
	// Phase 2: Block proposal
	var proposal *Block
	if node, exists := s.nodes[leader.Address]; exists {
		if node.IsHonest {
			proposal = s.createHonestProposal(node, round)
		} else {
			proposal = s.createByzantineProposal(node, round)
			atomic.AddInt64(&s.metrics.ByzantineAttempts, 1)
		}
	}
	
	// Phase 3: Voting
	votes := s.collectVotes(proposal)
	
	// Phase 4: Finalization
	if s.hasSupermajority(votes, proposal) {
		if err := s.finalizeBlock(proposal); err != nil {
			return err
		}
		
		// Check if Byzantine succeeded
		if proposer, exists := s.nodes[leader.Address]; exists && !proposer.IsHonest {
			atomic.AddInt64(&s.metrics.ByzantineSuccesses, 1)
		}
	} else {
		return fmt.Errorf("no supermajority reached")
	}
	
	// Phase 5: State updates
	s.updateValidatorStates(proposal)
	
	return nil
}

// createHonestProposal creates a valid block proposal
func (s *Simulator) createHonestProposal(node *SimulationNode, round int) *Block {
	quality := s.generateQualityScore()
	
	return &Block{
		Height:    uint64(round),
		Proposer:  node.ID,
		Timestamp: time.Now(),
		Quality:   quality,
		Commits: []Commit{
			{
				Hash:    fmt.Sprintf("commit_%d_%s", round, node.ID),
				Author:  node.ID,
				Message: fmt.Sprintf("Round %d commit", round),
				Quality: quality,
			},
		},
	}
}

// createByzantineProposal creates a potentially malicious proposal
func (s *Simulator) createByzantineProposal(node *SimulationNode, round int) *Block {
	// Byzantine validators might propose low-quality blocks
	quality := rand.Float64() * 30 // Low quality
	
	return &Block{
		Height:    uint64(round),
		Proposer:  node.ID,
		Timestamp: time.Now(),
		Quality:   quality,
		Commits: []Commit{
			{
				Hash:    fmt.Sprintf("byzantine_%d_%s", round, node.ID),
				Author:  node.ID,
				Message: "Malicious commit",
				Quality: quality,
			},
		},
	}
}

// collectVotes simulates the voting phase
func (s *Simulator) collectVotes(proposal *Block) map[string]bool {
	votes := make(map[string]bool)
	
	for _, node := range s.nodes {
		// Simulate network partitions
		if !node.Online {
			continue
		}
		
		// Simulate voting decision
		if node.IsHonest {
			// Honest nodes vote for valid proposals
			if proposal.Quality >= 40 { // Minimum quality threshold
				votes[node.ID] = true
			}
		} else {
			// Byzantine nodes might vote for anything
			votes[node.ID] = rand.Float64() < 0.7
		}
	}
	
	return votes
}

// hasSupermajority checks if proposal has 2/3+ stake
func (s *Simulator) hasSupermajority(votes map[string]bool, proposal *Block) bool {
	totalStake := uint64(0)
	votedStake := uint64(0)
	
	for nodeID, node := range s.nodes {
		stake := s.engine.CalculateTotalStake(node.State)
		totalStake += stake
		
		if votes[nodeID] {
			votedStake += stake
		}
	}
	
	return votedStake > (totalStake * 2 / 3)
}

// finalizeBlock commits the block to the chain
func (s *Simulator) finalizeBlock(block *Block) error {
	// Record quality score
	s.metrics.mu.Lock()
	s.metrics.QualityScores = append(s.metrics.QualityScores, block.Quality)
	s.metrics.mu.Unlock()
	
	// Process through consensus engine
	return s.engine.ProcessBlock(block)
}

// updateValidatorStates updates validator states after a round
func (s *Simulator) updateValidatorStates(block *Block) {
	for nodeID, node := range s.nodes {
		// Update reputation based on participation
		if block.Proposer == nodeID {
			node.Reputation = math.Min(1.0, node.Reputation*1.1)
		} else {
			node.Reputation = node.Reputation * 0.99 // Slight decay
		}
		
		// Track reputation evolution
		s.metrics.mu.Lock()
		s.metrics.ReputationEvolution[nodeID] = append(
			s.metrics.ReputationEvolution[nodeID], 
			node.Reputation,
		)
		s.metrics.mu.Unlock()
		
		// Simulate contributions
		if rand.Float64() < s.config.ContributionRate/3600 {
			node.Contributions = s.generateQualityScore()
		}
		
		// Check for slashing conditions
		if !node.IsHonest && block.Quality < 30 && block.Proposer == nodeID {
			node.Stake = node.Stake / 2 // 50% slash
			atomic.AddInt64(&s.metrics.SlashingEvents, 1)
		}
		
		// Sync state with engine
		node.State.Stake = node.Stake
		node.State.Reputation = node.Reputation
		node.State.RecentContributions = node.Contributions
	}
}

// updateMetrics updates running metrics
func (s *Simulator) updateMetrics() {
	s.metrics.mu.Lock()
	defer s.metrics.mu.Unlock()
	
	// Calculate average finality
	if len(s.metrics.ConsensusTime) > 0 {
		total := time.Duration(0)
		for _, t := range s.metrics.ConsensusTime {
			total += t
		}
		s.metrics.AverageFinality = float64(total) / float64(len(s.metrics.ConsensusTime)) / float64(time.Second)
	}
	
	// Calculate throughput
	if s.metrics.TotalRounds > 0 {
		avgTime := s.metrics.AverageFinality
		if avgTime > 0 {
			s.metrics.Throughput = 1.0 / avgTime * 100 // Assume 100 txs per block
		}
	}
}

// printStatus prints current simulation status
func (s *Simulator) printStatus(round int) {
	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()
	
	successRate := float64(s.metrics.SuccessfulBlocks) / float64(s.metrics.TotalRounds) * 100
	byzSuccessRate := float64(0)
	if s.metrics.ByzantineAttempts > 0 {
		byzSuccessRate = float64(s.metrics.ByzantineSuccesses) / float64(s.metrics.ByzantineAttempts) * 100
	}
	
	fmt.Printf("\n=== Round %d Status ===\n", round)
	fmt.Printf("Success Rate: %.2f%%\n", successRate)
	fmt.Printf("Average Finality: %.3f seconds\n", s.metrics.AverageFinality)
	fmt.Printf("Throughput: %.2f tx/s\n", s.metrics.Throughput)
	fmt.Printf("Byzantine Success Rate: %.2f%%\n", byzSuccessRate)
	fmt.Printf("Slashing Events: %d\n", s.metrics.SlashingEvents)
	
	// Show top validators by reputation
	s.printTopValidators(5)
}

// printTopValidators shows validators with highest reputation
func (s *Simulator) printTopValidators(n int) {
	type validatorRep struct {
		id  string
		rep float64
	}
	
	var validators []validatorRep
	for id, node := range s.nodes {
		validators = append(validators, validatorRep{id, node.Reputation})
	}
	
	// Sort by reputation
	for i := 0; i < len(validators)-1; i++ {
		for j := i + 1; j < len(validators); j++ {
			if validators[j].rep > validators[i].rep {
				validators[i], validators[j] = validators[j], validators[i]
			}
		}
	}
	
	fmt.Print("\nTop Validators by Reputation:\n")
	for i := 0; i < n && i < len(validators); i++ {
		fmt.Printf("  %d. %s: %.3f\n", i+1, validators[i].id, validators[i].rep)
	}
}

// finalizeMetrics calculates final simulation metrics
func (s *Simulator) finalizeMetrics(duration time.Duration) {
	s.metrics.mu.Lock()
	defer s.metrics.mu.Unlock()
	
	// Calculate quality statistics
	if len(s.metrics.QualityScores) > 0 {
		sum := float64(0)
		for _, q := range s.metrics.QualityScores {
			sum += q
		}
		avgQuality := sum / float64(len(s.metrics.QualityScores))
		
		fmt.Printf("\n=== Final Simulation Results ===\n")
		fmt.Printf("Total Duration: %v\n", duration)
		fmt.Printf("Total Rounds: %d\n", s.metrics.TotalRounds)
		fmt.Printf("Successful Blocks: %d (%.2f%%)\n", 
			s.metrics.SuccessfulBlocks,
			float64(s.metrics.SuccessfulBlocks)/float64(s.metrics.TotalRounds)*100)
		fmt.Printf("Average Block Quality: %.2f\n", avgQuality)
		fmt.Printf("Average Finality: %.3f seconds\n", s.metrics.AverageFinality)
		fmt.Printf("Average Throughput: %.2f tx/s\n", s.metrics.Throughput)
		fmt.Printf("Byzantine Attempts: %d\n", s.metrics.ByzantineAttempts)
		fmt.Printf("Byzantine Successes: %d\n", s.metrics.ByzantineSuccesses)
		fmt.Printf("Slashing Events: %d\n", s.metrics.SlashingEvents)
	}
}

// Stop gracefully stops the simulation
func (s *Simulator) Stop() {
	s.mu.Lock()
	s.stopped = true
	s.mu.Unlock()
	
	s.wg.Wait()
}

// SimulateNetworkPartition simulates a network partition
func (s *Simulator) SimulateNetworkPartition(nodes []string, duration time.Duration) {
	s.network.mu.Lock()
	for _, node := range nodes {
		s.network.partitions[node] = true
	}
	s.network.mu.Unlock()
	
	// Restore after duration
	time.AfterFunc(duration, func() {
		s.network.mu.Lock()
		for _, node := range nodes {
			delete(s.network.partitions, node)
		}
		s.network.mu.Unlock()
	})
}

// RunBenchmark runs performance benchmarks
func RunBenchmark(configs []SimulationConfig) {
	fmt.Print("\n=== Running PoC Consensus Benchmarks ===\n\n")
	
	results := make([]struct {
		Validators int
		Throughput float64
		Finality   float64
		Byzantine  int
	}, 0)
	
	for _, config := range configs {
		fmt.Printf("Testing with %d validators, %d Byzantine...\n", 
			config.NumValidators, config.NumByzantine)
		
		sim := NewSimulator(&config)
		metrics, err := sim.Run()
		
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		
		results = append(results, struct {
			Validators int
			Throughput float64
			Finality   float64
			Byzantine  int
		}{
			Validators: config.NumValidators,
			Throughput: metrics.Throughput,
			Finality:   metrics.AverageFinality,
			Byzantine:  config.NumByzantine,
		})
	}
	
	// Print results table
	fmt.Print("\n=== Benchmark Results ===\n")
	fmt.Println("Validators | Byzantine | Throughput (tx/s) | Finality (s)")
	fmt.Println("-----------|-----------|-------------------|-------------")
	for _, r := range results {
		fmt.Printf("%10d | %9d | %17.2f | %12.3f\n", 
			r.Validators, r.Byzantine, r.Throughput, r.Finality)
	}
}