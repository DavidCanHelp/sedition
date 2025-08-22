// Package poc implements the Proof of Contribution (PoC) consensus algorithm
// specifically designed for collaborative software development.
//
// The PoC consensus mechanism combines:
// - Token stake (economic incentive)
// - Reputation score (historical performance)
// - Recent contributions (activity and quality)
//
// Mathematical Foundation:
// Total Stake = TokenStake × ReputationMultiplier × ContributionBonus
// Selection Probability ∝ Total Stake / Sum(All Stakes)
package poc

import (
	"crypto/rand"
	"errors"
	"math"
	"math/big"
	"time"
)

// ConsensusEngine represents the main PoC consensus engine
type ConsensusEngine struct {
	validators        map[string]*Validator
	qualityAnalyzer   *QualityAnalyzer
	reputationTracker *ReputationTracker
	metricsCalculator *MetricsCalculator
	
	// Consensus parameters
	minStakeRequired  *big.Int
	blockTime         time.Duration
	epochLength       int64
	slashingRate      float64 // Percentage of stake to slash for malicious behavior
	
	// Current state
	currentEpoch      int64
	lastBlockTime     time.Time
	proposerHistory   []string // Track recent proposers for fairness
}

// calculateTotalStake calculates the total stake for a validator
func (ce *ConsensusEngine) calculateTotalStake(v *Validator) {
	// Calculate reputation multiplier (0.1 to 1.0 based on 0-10 score)
	repMultiplier := v.ReputationScore / 10.0
	if repMultiplier < 0.1 {
		repMultiplier = 0.1
	}
	
	// Calculate contribution bonus (1.0 to 2.0 based on recent quality)
	contribBonus := 1.0
	if len(v.RecentContribs) > 0 {
		totalQuality := 0.0
		for _, c := range v.RecentContribs {
			totalQuality += c.QualityScore
		}
		avgQuality := totalQuality / float64(len(v.RecentContribs))
		contribBonus = 1.0 + (avgQuality / 100.0)
	}
	
	// Calculate total stake
	baseStake := new(big.Float).SetInt(v.TokenStake)
	totalStake := new(big.Float).Mul(baseStake, big.NewFloat(repMultiplier))
	totalStake = new(big.Float).Mul(totalStake, big.NewFloat(contribBonus))
	
	// Convert back to big.Int
	v.TotalStake, _ = totalStake.Int(nil)
}

// Validator represents a network validator/contributor
type Validator struct {
	Address           string
	TokenStake        *big.Int        // Raw token amount staked
	ReputationScore   float64         // Reputation score (0.0 to 10.0)
	RecentContribs    []Contribution  // Recent contributions for this epoch
	TotalStake        *big.Int        // Calculated total stake including all factors
	LastActivityTime  time.Time       // Last contribution timestamp
	SlashingHistory   []SlashingEvent // History of slashing events
	IsActive          bool            // Whether validator is currently active
}

// Contribution represents a single code contribution
type Contribution struct {
	ID              string
	Timestamp       time.Time
	Type            ContributionType
	LinesAdded      int
	LinesModified   int
	LinesDeleted    int
	TestCoverage    float64    // Percentage of test coverage added
	Complexity      float64    // Cyclomatic complexity score
	Documentation   float64    // Documentation coverage score
	QualityScore    float64    // Overall quality score (0-100)
	PeerReviews     int        // Number of peer reviews received
	ReviewScore     float64    // Average review score
}

// ContributionType defines the type of contribution
type ContributionType int

const (
	CodeCommit ContributionType = iota
	PullRequest
	IssueResolution
	Documentation
	Testing
	CodeReview
	Security
)

// SlashingEvent represents a slashing incident
type SlashingEvent struct {
	Timestamp   time.Time
	Reason      SlashingReason
	AmountSlashed *big.Int
	Evidence    string
}

// SlashingReason defines reasons for slashing
type SlashingReason int

const (
	MaliciousCode SlashingReason = iota
	FalseContribution
	DoubleProposal
	NetworkAttack
	QualityViolation
)

// NewConsensusEngine creates a new PoC consensus engine
func NewConsensusEngine(minStake *big.Int, blockTime time.Duration) *ConsensusEngine {
	return &ConsensusEngine{
		validators:        make(map[string]*Validator),
		qualityAnalyzer:   NewQualityAnalyzer(),
		reputationTracker: NewReputationTracker(),
		metricsCalculator: NewMetricsCalculator(),
		minStakeRequired:  minStake,
		blockTime:         blockTime,
		epochLength:       100,  // 100 blocks per epoch
		slashingRate:      0.1,  // 10% slashing rate
		currentEpoch:      0,
		lastBlockTime:     time.Now(),
		proposerHistory:   make([]string, 0, 20), // Track last 20 proposers
	}
}

// RegisterValidator registers a new validator in the network
func (ce *ConsensusEngine) RegisterValidator(address string, tokenStake *big.Int) error {
	if tokenStake.Cmp(ce.minStakeRequired) < 0 {
		return errors.New("insufficient stake amount")
	}
	
	validator := &Validator{
		Address:          address,
		TokenStake:       new(big.Int).Set(tokenStake),
		ReputationScore:  5.0, // Start with neutral reputation
		RecentContribs:   make([]Contribution, 0),
		TotalStake:       new(big.Int),
		LastActivityTime: time.Now(),
		SlashingHistory:  make([]SlashingEvent, 0),
		IsActive:         true,
	}
	
	ce.validators[address] = validator
	ce.calculateValidatorStake(validator)
	
	return nil
}

// SubmitContribution processes a new contribution from a validator
func (ce *ConsensusEngine) SubmitContribution(validatorAddr string, contrib Contribution) error {
	validator, exists := ce.validators[validatorAddr]
	if !exists {
		return errors.New("validator not found")
	}
	
	// Analyze contribution quality
	qualityScore, err := ce.qualityAnalyzer.AnalyzeContribution(contrib)
	if err != nil {
		return err
	}
	contrib.QualityScore = qualityScore
	
	// Update validator's contributions
	validator.RecentContribs = append(validator.RecentContribs, contrib)
	validator.LastActivityTime = time.Now()
	
	// Update reputation based on contribution
	ce.reputationTracker.UpdateReputation(validatorAddr, contrib)
	validator.ReputationScore = ce.reputationTracker.GetReputation(validatorAddr)
	
	// Recalculate total stake
	ce.calculateValidatorStake(validator)
	
	return nil
}

// calculateValidatorStake calculates the total stake for a validator
// Formula: TotalStake = TokenStake × ReputationMultiplier × ContributionBonus
func (ce *ConsensusEngine) calculateValidatorStake(validator *Validator) {
	// Base stake from tokens
	baseStake := new(big.Float).SetInt(validator.TokenStake)
	
	// Reputation multiplier (0.5x to 2.0x based on reputation score 0-10)
	reputationMultiplier := math.Max(0.5, math.Min(2.0, validator.ReputationScore/5.0))
	
	// Contribution bonus based on recent activity
	contributionBonus := ce.calculateContributionBonus(validator)
	
	// Calculate final stake
	finalStake := new(big.Float).Mul(baseStake, big.NewFloat(reputationMultiplier))
	finalStake = finalStake.Mul(finalStake, big.NewFloat(contributionBonus))
	
	// Convert back to big.Int
	validator.TotalStake, _ = finalStake.Int(nil)
}

// calculateContributionBonus calculates bonus multiplier based on recent contributions
func (ce *ConsensusEngine) calculateContributionBonus(validator *Validator) float64 {
	if len(validator.RecentContribs) == 0 {
		return 0.8 // Penalty for inactivity
	}
	
	// Calculate average quality of recent contributions
	totalQuality := 0.0
	recentContributions := 0
	cutoffTime := time.Now().Add(-7 * 24 * time.Hour) // Last 7 days
	
	for _, contrib := range validator.RecentContribs {
		if contrib.Timestamp.After(cutoffTime) {
			totalQuality += contrib.QualityScore
			recentContributions++
		}
	}
	
	if recentContributions == 0 {
		return 0.9 // Small penalty for no recent contributions
	}
	
	avgQuality := totalQuality / float64(recentContributions)
	
	// Bonus calculation: 0.8x to 1.5x based on quality and frequency
	qualityBonus := math.Max(0.8, math.Min(1.3, avgQuality/75.0)) // Quality score 0-100
	frequencyBonus := math.Max(1.0, math.Min(1.2, float64(recentContributions)/5.0))
	
	return qualityBonus * frequencyBonus
}

// SelectBlockProposer selects the next block proposer using weighted random selection
func (ce *ConsensusEngine) SelectBlockProposer() (string, error) {
	activeValidators := ce.getActiveValidators()
	if len(activeValidators) == 0 {
		return "", errors.New("no active validators")
	}
	
	// Calculate total stake for probability distribution
	totalStake := new(big.Int)
	stakes := make(map[string]*big.Int)
	
	for addr, validator := range activeValidators {
		stakes[addr] = validator.TotalStake
		totalStake = totalStake.Add(totalStake, validator.TotalStake)
	}
	
	// Generate random number for selection
	randNum, err := rand.Int(rand.Reader, totalStake)
	if err != nil {
		return "", err
	}
	
	// Select proposer based on weighted probability
	currentSum := new(big.Int)
	for addr, stake := range stakes {
		currentSum = currentSum.Add(currentSum, stake)
		if randNum.Cmp(currentSum) < 0 {
			// Add fairness check - avoid same proposer too frequently
			if ce.isProposerTooFrequent(addr) {
				continue // Try next validator
			}
			
			ce.recordProposer(addr)
			return addr, nil
		}
	}
	
	// Fallback - should not happen with proper math
	for addr := range activeValidators {
		return addr, nil
	}
	
	return "", errors.New("failed to select proposer")
}

// isProposerTooFrequent checks if a validator has been selected too frequently
func (ce *ConsensusEngine) isProposerTooFrequent(addr string) bool {
	if len(ce.proposerHistory) < 5 {
		return false // Not enough history
	}
	
	// Check if proposer was selected in last 3 blocks
	recentCount := 0
	for i := len(ce.proposerHistory) - 3; i < len(ce.proposerHistory); i++ {
		if ce.proposerHistory[i] == addr {
			recentCount++
		}
	}
	
	return recentCount >= 2 // Max 2 times in last 3 blocks
}

// recordProposer records the selected proposer for fairness tracking
func (ce *ConsensusEngine) recordProposer(addr string) {
	ce.proposerHistory = append(ce.proposerHistory, addr)
	
	// Keep only last 20 proposers
	if len(ce.proposerHistory) > 20 {
		ce.proposerHistory = ce.proposerHistory[1:]
	}
}

// getActiveValidators returns all active validators with sufficient stake
func (ce *ConsensusEngine) getActiveValidators() map[string]*Validator {
	active := make(map[string]*Validator)
	
	for addr, validator := range ce.validators {
		if validator.IsActive && 
		   validator.TokenStake.Cmp(ce.minStakeRequired) >= 0 &&
		   validator.TotalStake.Cmp(big.NewInt(0)) > 0 {
			active[addr] = validator
		}
	}
	
	return active
}

// SlashValidator slashes a validator for malicious behavior
func (ce *ConsensusEngine) SlashValidator(addr string, reason SlashingReason, evidence string) error {
	validator, exists := ce.validators[addr]
	if !exists {
		return errors.New("validator not found")
	}
	
	// Calculate slashing amount
	slashAmount := new(big.Float).SetInt(validator.TokenStake)
	slashAmount = slashAmount.Mul(slashAmount, big.NewFloat(ce.slashingRate))
	slashAmountInt, _ := slashAmount.Int(nil)
	
	// Apply slashing
	validator.TokenStake = validator.TokenStake.Sub(validator.TokenStake, slashAmountInt)
	
	// Record slashing event
	slashingEvent := SlashingEvent{
		Timestamp:     time.Now(),
		Reason:        reason,
		AmountSlashed: slashAmountInt,
		Evidence:      evidence,
	}
	validator.SlashingHistory = append(validator.SlashingHistory, slashingEvent)
	
	// Update reputation (significant penalty)
	ce.reputationTracker.ApplySlashing(addr, reason)
	validator.ReputationScore = ce.reputationTracker.GetReputation(addr)
	
	// Recalculate stake
	ce.calculateValidatorStake(validator)
	
	// Deactivate if stake too low
	if validator.TokenStake.Cmp(ce.minStakeRequired) < 0 {
		validator.IsActive = false
	}
	
	return nil
}

// UpdateEpoch advances to the next epoch and performs cleanup
func (ce *ConsensusEngine) UpdateEpoch() {
	ce.currentEpoch++
	
	// Clean up old contributions and update reputation decay
	for addr, validator := range ce.validators {
		// Remove contributions older than epoch
		cutoffTime := time.Now().Add(-time.Duration(ce.epochLength) * ce.blockTime)
		newContribs := make([]Contribution, 0)
		
		for _, contrib := range validator.RecentContribs {
			if contrib.Timestamp.After(cutoffTime) {
				newContribs = append(newContribs, contrib)
			}
		}
		
		validator.RecentContribs = newContribs
		
		// Apply reputation decay
		ce.reputationTracker.ApplyDecay(addr)
		validator.ReputationScore = ce.reputationTracker.GetReputation(addr)
		
		// Recalculate stake
		ce.calculateValidatorStake(validator)
	}
}

// GetValidatorStats returns statistics for a validator
func (ce *ConsensusEngine) GetValidatorStats(addr string) (*ValidatorStats, error) {
	validator, exists := ce.validators[addr]
	if !exists {
		return nil, errors.New("validator not found")
	}
	
	stats := &ValidatorStats{
		Address:           validator.Address,
		TokenStake:        new(big.Int).Set(validator.TokenStake),
		ReputationScore:   validator.ReputationScore,
		TotalStake:        new(big.Int).Set(validator.TotalStake),
		ContributionCount: len(validator.RecentContribs),
		IsActive:          validator.IsActive,
		LastActivity:      validator.LastActivityTime,
		SlashingCount:     len(validator.SlashingHistory),
	}
	
	// Calculate average quality
	if len(validator.RecentContribs) > 0 {
		totalQuality := 0.0
		for _, contrib := range validator.RecentContribs {
			totalQuality += contrib.QualityScore
		}
		stats.AverageQuality = totalQuality / float64(len(validator.RecentContribs))
	}
	
	return stats, nil
}

// ValidatorStats holds statistics for a validator
type ValidatorStats struct {
	Address           string
	TokenStake        *big.Int
	ReputationScore   float64
	TotalStake        *big.Int
	ContributionCount int
	AverageQuality    float64
	IsActive          bool
	LastActivity      time.Time
	SlashingCount     int
}

// GetNetworkStats returns overall network statistics
func (ce *ConsensusEngine) GetNetworkStats() *NetworkStats {
	stats := &NetworkStats{
		TotalValidators: len(ce.validators),
		CurrentEpoch:    ce.currentEpoch,
		LastBlockTime:   ce.lastBlockTime,
	}
	
	activeCount := 0
	totalStake := new(big.Int)
	
	for _, validator := range ce.validators {
		if validator.IsActive {
			activeCount++
			totalStake = totalStake.Add(totalStake, validator.TokenStake)
		}
	}
	
	stats.ActiveValidators = activeCount
	stats.TotalStaked = totalStake
	
	return stats
}

// NetworkStats holds network-wide statistics
type NetworkStats struct {
	TotalValidators   int
	ActiveValidators  int
	TotalStaked      *big.Int
	CurrentEpoch     int64
	LastBlockTime    time.Time
}