// Package consensus implements the Proof of Contribution consensus mechanism
package consensus

import (
	"crypto/rand"
	"math"
	"math/big"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/contribution"
	"github.com/davidcanhelp/sedition/errors"
	"github.com/davidcanhelp/sedition/validator"
)

// Engine represents the main PoC consensus engine
type Engine struct {
	validators        map[string]*validator.Validator
	qualityAnalyzer   *contribution.QualityAnalyzer
	reputationTracker *validator.ReputationTracker
	metricsCalculator *contribution.MetricsCalculator
	config            *config.ConsensusConfig

	// Current state
	currentEpoch    int64
	lastBlockTime   time.Time
	proposerHistory []string
}

// NewEngine creates a new PoC consensus engine with configuration
func NewEngine(cfg *config.ConsensusConfig) *Engine {
	if cfg == nil {
		cfg = config.DefaultConsensusConfig()
	}

	return &Engine{
		validators:        make(map[string]*validator.Validator),
		qualityAnalyzer:   contribution.NewQualityAnalyzer(),
		reputationTracker: validator.NewReputationTracker(),
		metricsCalculator: contribution.NewMetricsCalculator(),
		config:            cfg,
		currentEpoch:      0,
		lastBlockTime:     time.Now(),
		proposerHistory:   make([]string, 0, cfg.ProposerHistorySize),
	}
}

// RegisterValidator registers a new validator in the network
func (e *Engine) RegisterValidator(address string, tokenStake *big.Int) error {
	if tokenStake.Cmp(e.config.MinStakeRequired) < 0 {
		return errors.NewConsensusError(
			errors.ErrInsufficientStake,
			"stake amount below minimum requirement",
		).WithDetails("required", e.config.MinStakeRequired).
			WithDetails("provided", tokenStake)
	}

	v := &validator.Validator{
		Address:          address,
		TokenStake:       new(big.Int).Set(tokenStake),
		ReputationScore:  e.config.InitialReputation,
		RecentContribs:   make([]contribution.Contribution, 0),
		TotalStake:       new(big.Int),
		LastActivityTime: time.Now(),
		SlashingHistory:  make([]validator.SlashingEvent, 0),
		IsActive:         true,
	}

	e.validators[address] = v
	e.calculateValidatorStake(v)

	return nil
}

// SubmitContribution processes a new contribution from a validator
func (e *Engine) SubmitContribution(validatorAddr string, contrib contribution.Contribution) error {
	v, exists := e.validators[validatorAddr]
	if !exists {
		return errors.NewConsensusError(
			errors.ErrValidatorNotFound,
			"validator not registered",
		).WithDetails("address", validatorAddr)
	}

	// Analyze contribution quality
	qualityScore, err := e.qualityAnalyzer.AnalyzeContribution(contrib)
	if err != nil {
		return errors.NewConsensusError(
			errors.ErrInvalidContribution,
			"failed to analyze contribution quality",
		).WithDetails("error", err.Error())
	}
	contrib.QualityScore = qualityScore

	// Update validator's contributions
	v.RecentContribs = append(v.RecentContribs, contrib)
	v.LastActivityTime = time.Now()

	// Update reputation based on contribution
	e.reputationTracker.UpdateReputation(validatorAddr, contrib)
	v.ReputationScore = e.reputationTracker.GetReputation(validatorAddr)

	// Recalculate total stake
	e.calculateValidatorStake(v)

	return nil
}

// calculateValidatorStake calculates the total stake for a validator
func (e *Engine) calculateValidatorStake(v *validator.Validator) {
	// Base stake from tokens
	baseStake := new(big.Float).SetInt(v.TokenStake)

	// Reputation multiplier based on config
	repRatio := v.ReputationScore / 10.0
	reputationMultiplier := e.config.MinReputationMultiplier +
		(e.config.MaxReputationMultiplier-e.config.MinReputationMultiplier)*repRatio

	// Contribution bonus based on recent activity
	contributionBonus := e.calculateContributionBonus(v)

	// Calculate final stake
	finalStake := new(big.Float).Mul(baseStake, big.NewFloat(reputationMultiplier))
	finalStake = finalStake.Mul(finalStake, big.NewFloat(contributionBonus))

	// Convert back to big.Int
	v.TotalStake, _ = finalStake.Int(nil)
}

// calculateContributionBonus calculates bonus multiplier based on recent contributions
func (e *Engine) calculateContributionBonus(v *validator.Validator) float64 {
	if len(v.RecentContribs) == 0 {
		return e.config.InactivityPenalty
	}

	// Calculate average quality of recent contributions
	totalQuality := 0.0
	recentContributions := 0
	cutoffTime := time.Now().Add(-e.config.ContributionWindow)

	for _, contrib := range v.RecentContribs {
		if contrib.Timestamp.After(cutoffTime) {
			totalQuality += contrib.QualityScore
			recentContributions++
		}
	}

	if recentContributions == 0 {
		return e.config.NoRecentActivityPenalty
	}

	avgQuality := totalQuality / float64(recentContributions)

	// Calculate bonus based on quality threshold
	qualityRatio := avgQuality / e.config.QualityThreshold
	qualityBonus := e.config.MinContributionBonus +
		(e.config.MaxContributionBonus-e.config.MinContributionBonus)*math.Min(1.0, qualityRatio)

	// Frequency bonus
	frequencyBonus := math.Min(1.2, 1.0+float64(recentContributions)*0.04)

	return qualityBonus * frequencyBonus
}

// SelectBlockProposer selects the next block proposer using weighted random selection
func (e *Engine) SelectBlockProposer() (string, error) {
	activeValidators := e.getActiveValidators()
	if len(activeValidators) == 0 {
		return "", errors.NewConsensusError(
			errors.ErrNoActiveValidators,
			"no validators available for selection",
		)
	}

	// Calculate total stake for probability distribution
	totalStake := new(big.Int)
	stakes := make(map[string]*big.Int)

	for addr, v := range activeValidators {
		stakes[addr] = v.TotalStake
		totalStake = totalStake.Add(totalStake, v.TotalStake)
	}

	// Generate random number for selection
	randNum, err := rand.Int(rand.Reader, totalStake)
	if err != nil {
		return "", errors.NewConsensusError(
			errors.ErrProposerSelection,
			"failed to generate random number",
		).WithDetails("error", err.Error())
	}

	// Select proposer based on weighted probability
	currentSum := new(big.Int)
	for addr, stake := range stakes {
		currentSum = currentSum.Add(currentSum, stake)
		if randNum.Cmp(currentSum) < 0 {
			// Add fairness check
			if e.isProposerTooFrequent(addr) {
				continue
			}

			e.recordProposer(addr)
			return addr, nil
		}
	}

	// Fallback - should not happen with proper math
	for addr := range activeValidators {
		return addr, nil
	}

	return "", errors.NewConsensusError(
		errors.ErrProposerSelection,
		"failed to select proposer",
	)
}

// isProposerTooFrequent checks if a validator has been selected too frequently
func (e *Engine) isProposerTooFrequent(addr string) bool {
	historyLen := len(e.proposerHistory)
	if historyLen < e.config.ProposerFrequencyWindow {
		return false
	}

	// Check recent selections
	recentCount := 0
	startIdx := historyLen - e.config.ProposerFrequencyWindow
	for i := startIdx; i < historyLen; i++ {
		if e.proposerHistory[i] == addr {
			recentCount++
		}
	}

	return recentCount >= e.config.MaxProposerFrequency
}

// recordProposer records the selected proposer for fairness tracking
func (e *Engine) recordProposer(addr string) {
	e.proposerHistory = append(e.proposerHistory, addr)

	// Keep only configured history size
	if len(e.proposerHistory) > e.config.ProposerHistorySize {
		e.proposerHistory = e.proposerHistory[1:]
	}
}

// getActiveValidators returns all active validators with sufficient stake
func (e *Engine) getActiveValidators() map[string]*validator.Validator {
	active := make(map[string]*validator.Validator)

	for addr, v := range e.validators {
		if v.IsActive &&
			v.TokenStake.Cmp(e.config.MinStakeRequired) >= 0 &&
			v.TotalStake.Cmp(big.NewInt(0)) > 0 {
			active[addr] = v
		}
	}

	return active
}

// SlashValidator slashes a validator for malicious behavior
func (e *Engine) SlashValidator(addr string, reason validator.SlashingReason, evidence string) error {
	v, exists := e.validators[addr]
	if !exists {
		return errors.NewConsensusError(
			errors.ErrValidatorNotFound,
			"validator not found for slashing",
		).WithDetails("address", addr)
	}

	// Calculate slashing amount
	slashAmount := new(big.Float).SetInt(v.TokenStake)
	slashAmount = slashAmount.Mul(slashAmount, big.NewFloat(e.config.SlashingRate))
	slashAmountInt, _ := slashAmount.Int(nil)

	// Apply slashing
	v.TokenStake = v.TokenStake.Sub(v.TokenStake, slashAmountInt)

	// Record slashing event
	slashingEvent := validator.SlashingEvent{
		Timestamp:     time.Now(),
		Reason:        reason,
		AmountSlashed: slashAmountInt,
		Evidence:      evidence,
	}
	v.SlashingHistory = append(v.SlashingHistory, slashingEvent)

	// Update reputation (significant penalty)
	e.reputationTracker.ApplySlashing(addr, reason)
	v.ReputationScore = e.reputationTracker.GetReputation(addr)

	// Recalculate stake
	e.calculateValidatorStake(v)

	// Deactivate if stake too low
	if v.TokenStake.Cmp(e.config.MinStakeRequired) < 0 {
		v.IsActive = false
	}

	return nil
}

// UpdateEpoch advances to the next epoch and performs cleanup
func (e *Engine) UpdateEpoch() {
	e.currentEpoch++

	// Clean up old contributions and update reputation decay
	for addr, v := range e.validators {
		// Remove contributions older than epoch
		cutoffTime := time.Now().Add(-time.Duration(e.config.EpochLength) * e.config.BlockTime)
		newContribs := make([]contribution.Contribution, 0)

		for _, contrib := range v.RecentContribs {
			if contrib.Timestamp.After(cutoffTime) {
				newContribs = append(newContribs, contrib)
			}
		}

		v.RecentContribs = newContribs

		// Apply reputation decay
		e.reputationTracker.ApplyDecay(addr)
		v.ReputationScore = e.reputationTracker.GetReputation(addr)

		// Recalculate stake
		e.calculateValidatorStake(v)
	}
}

// GetValidatorStats returns statistics for a validator
func (e *Engine) GetValidatorStats(addr string) (*validator.Stats, error) {
	v, exists := e.validators[addr]
	if !exists {
		return nil, errors.NewConsensusError(
			errors.ErrValidatorNotFound,
			"validator not found",
		).WithDetails("address", addr)
	}

	stats := &validator.Stats{
		Address:           v.Address,
		TokenStake:        new(big.Int).Set(v.TokenStake),
		ReputationScore:   v.ReputationScore,
		TotalStake:        new(big.Int).Set(v.TotalStake),
		ContributionCount: len(v.RecentContribs),
		IsActive:          v.IsActive,
		LastActivity:      v.LastActivityTime,
		SlashingCount:     len(v.SlashingHistory),
	}

	// Calculate average quality
	if len(v.RecentContribs) > 0 {
		totalQuality := 0.0
		for _, contrib := range v.RecentContribs {
			totalQuality += contrib.QualityScore
		}
		stats.AverageQuality = totalQuality / float64(len(v.RecentContribs))
	}

	return stats, nil
}

// GetNetworkStats returns overall network statistics
func (e *Engine) GetNetworkStats() *NetworkStats {
	stats := &NetworkStats{
		TotalValidators: len(e.validators),
		CurrentEpoch:    e.currentEpoch,
		LastBlockTime:   e.lastBlockTime,
	}

	activeCount := 0
	totalStake := new(big.Int)

	for _, v := range e.validators {
		if v.IsActive {
			activeCount++
			totalStake = totalStake.Add(totalStake, v.TokenStake)
		}
	}

	stats.ActiveValidators = activeCount
	stats.TotalStaked = totalStake

	return stats
}

// NetworkStats holds network-wide statistics
type NetworkStats struct {
	TotalValidators  int
	ActiveValidators int
	TotalStaked      *big.Int
	CurrentEpoch     int64
	LastBlockTime    time.Time
}
