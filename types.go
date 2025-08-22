package poc

import (
	"errors"
	"math"
	"math/big"
	"time"
)

// Block represents a block in the blockchain
type Block struct {
	Height    uint64
	Proposer  string
	Timestamp time.Time
	Quality   float64
	Commits   []Commit
	Votes     map[string]bool
	Hash      string
}

// Commit represents a code commit in a block
type Commit struct {
	Hash      string
	Author    string
	Message   string
	Quality   float64
	Timestamp time.Time
	Files     []string
}

// ValidatorState represents the simplified state of a validator for simulation
type ValidatorState struct {
	Address             string
	Stake               uint64
	Reputation          float64
	RecentContributions float64
	LastActive          time.Time
	SlashingHistory     []SlashingEvent
}

// NetworkState represents the overall network state
type NetworkState struct {
	Height          uint64
	TotalStake      uint64
	ActiveValidators int
	QualityThreshold float64
	SlashingRate    float64
}

// ProofOfContribution is the main consensus struct
type ProofOfContribution struct {
	engine *ConsensusEngine
}

// NewProofOfContribution creates a new PoC consensus instance
func NewProofOfContribution() *ProofOfContribution {
	minStake := new(big.Int)
	minStake.SetString("1000000000000000000", 10) // 1 token minimum
	
	engine := &ConsensusEngine{
		validators:        make(map[string]*Validator),
		qualityAnalyzer:   NewQualityAnalyzer(),
		reputationTracker: NewReputationTracker(),
		metricsCalculator: NewMetricsCalculator(),
		minStakeRequired:  minStake,
		blockTime:         12 * time.Second,
		epochLength:       100,
		slashingRate:      0.1, // 10% slashing
		lastBlockTime:     time.Now(),
		proposerHistory:   make([]string, 0),
	}
	
	return &ProofOfContribution{engine: engine}
}

// RegisterValidator adds a validator to the consensus
func (poc *ProofOfContribution) RegisterValidator(state *ValidatorState) {
	stake := new(big.Int).SetUint64(state.Stake)
	
	validator := &Validator{
		Address:          state.Address,
		TokenStake:       stake,
		ReputationScore:  state.Reputation * 10, // Convert to 0-10 scale
		RecentContribs:   []Contribution{},
		TotalStake:       stake,
		LastActivityTime: state.LastActive,
		SlashingHistory:  state.SlashingHistory,
		IsActive:         true,
	}
	
	poc.engine.validators[state.Address] = validator
}

// SelectBlockProposer selects the next block proposer
func (poc *ProofOfContribution) SelectBlockProposer() *Validator {
	leader, _ := poc.engine.SelectBlockProposer()
	if leader == "" {
		return nil
	}
	return poc.engine.validators[leader]
}

// CalculateTotalStake calculates the total stake for a validator
func (poc *ProofOfContribution) CalculateTotalStake(state *ValidatorState) uint64 {
	validator, exists := poc.engine.validators[state.Address]
	if !exists {
		return 0
	}
	
	// Calculate stake manually
	repMultiplier := validator.ReputationScore / 10.0
	if repMultiplier < 0.1 {
		repMultiplier = 0.1
	}
	
	contribBonus := 1.0 + (state.RecentContributions / 200.0)
	
	baseStake := new(big.Float).SetInt(validator.TokenStake)
	totalStake := new(big.Float).Mul(baseStake, big.NewFloat(repMultiplier))
	totalStake = new(big.Float).Mul(totalStake, big.NewFloat(contribBonus))
	
	result, _ := totalStake.Uint64()
	return result
}

// ProcessBlock processes a new block
func (poc *ProofOfContribution) ProcessBlock(block *Block) error {
	// Validate block quality
	if block.Quality < 30 {
		return errors.New("block quality too low")
	}
	
	// Update validator metrics based on block
	if proposer, exists := poc.engine.validators[block.Proposer]; exists {
		// Reward good behavior
		proposer.ReputationScore = math.Min(10, proposer.ReputationScore*1.01)
	}
	
	return nil
}

// NewCodeQualityAnalyzer creates a new quality analyzer (for tests)
func NewCodeQualityAnalyzer() *QualityAnalyzer {
	return NewQualityAnalyzer()
}