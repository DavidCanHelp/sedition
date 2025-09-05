// Package validator includes reputation management
package validator

import (
	"math"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/contribution"
)

// ReputationTracker manages validator reputation scores
type ReputationTracker struct {
	scores           map[string]float64
	history          map[string][]ReputationEvent
	mu               sync.RWMutex
	config           *config.ConsensusConfig
	lastDecayApplied time.Time
}

// ReputationEvent represents a change in reputation
type ReputationEvent struct {
	Timestamp time.Time
	OldScore  float64
	NewScore  float64
	Reason    string
	Delta     float64
}

// NewReputationTracker creates a new reputation tracker
func NewReputationTracker() *ReputationTracker {
	return &ReputationTracker{
		scores:           make(map[string]float64),
		history:          make(map[string][]ReputationEvent),
		config:           config.DefaultConsensusConfig(),
		lastDecayApplied: time.Now(),
	}
}

// NewReputationTrackerWithConfig creates a new reputation tracker with custom config
func NewReputationTrackerWithConfig(cfg *config.ConsensusConfig) *ReputationTracker {
	return &ReputationTracker{
		scores:           make(map[string]float64),
		history:          make(map[string][]ReputationEvent),
		config:           cfg,
		lastDecayApplied: time.Now(),
	}
}

// GetReputation returns the current reputation score for a validator
func (rt *ReputationTracker) GetReputation(address string) float64 {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	score, exists := rt.scores[address]
	if !exists {
		return rt.config.InitialReputation
	}
	return score
}

// SetReputation sets the reputation score for a validator
func (rt *ReputationTracker) SetReputation(address string, score float64) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	oldScore := rt.scores[address]
	// Clamp score between 0 and 10
	newScore := math.Max(0, math.Min(10, score))
	rt.scores[address] = newScore

	// Record event
	event := ReputationEvent{
		Timestamp: time.Now(),
		OldScore:  oldScore,
		NewScore:  newScore,
		Reason:    "manual_set",
		Delta:     newScore - oldScore,
	}

	rt.history[address] = append(rt.history[address], event)
}

// UpdateReputation updates reputation based on a contribution
func (rt *ReputationTracker) UpdateReputation(address string, contrib contribution.Contribution) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	currentScore := rt.scores[address]
	if currentScore == 0 {
		currentScore = rt.config.InitialReputation
	}

	// Calculate reputation change based on contribution quality
	// qualityFactor := contrib.QualityScore / 100.0 // Normalize to 0-1 (reserved for future use)

	// Determine reputation delta
	var delta float64
	switch {
	case contrib.QualityScore >= 90:
		delta = 0.2 // Excellent contribution
	case contrib.QualityScore >= 75:
		delta = 0.1 // Good contribution
	case contrib.QualityScore >= 50:
		delta = 0.05 // Average contribution
	case contrib.QualityScore >= 25:
		delta = -0.05 // Below average
	default:
		delta = -0.1 // Poor contribution
	}

	// Weight by contribution type
	switch contrib.Type {
	case contribution.Security:
		delta *= 1.5 // Security contributions more valuable
	case contribution.CodeReview:
		delta *= 1.2 // Code review valued
	case contribution.Documentation:
		delta *= 0.8 // Documentation less impact on reputation
	}

	// Apply peer review factor if available
	if contrib.PeerReviews > 0 && contrib.ReviewScore > 0 {
		reviewFactor := contrib.ReviewScore / 5.0 // Assuming 5-star system
		delta *= reviewFactor
	}

	newScore := math.Max(0, math.Min(10, currentScore+delta))
	rt.scores[address] = newScore

	// Record event
	event := ReputationEvent{
		Timestamp: time.Now(),
		OldScore:  currentScore,
		NewScore:  newScore,
		Reason:    "contribution",
		Delta:     delta,
	}

	rt.history[address] = append(rt.history[address], event)
}

// ApplySlashing applies reputation penalty for slashing
func (rt *ReputationTracker) ApplySlashing(address string, reason SlashingReason) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	currentScore := rt.scores[address]
	if currentScore == 0 {
		currentScore = rt.config.InitialReputation
	}

	// Determine penalty based on slashing reason
	var penalty float64
	switch reason {
	case MaliciousCode:
		penalty = 3.0 // Severe penalty
	case NetworkAttack:
		penalty = 2.5
	case DoubleProposal:
		penalty = 1.5
	case FalseContribution:
		penalty = 1.0
	case QualityViolation:
		penalty = 0.5
	default:
		penalty = 0.5
	}

	newScore := math.Max(0, currentScore-penalty)
	rt.scores[address] = newScore

	// Record event
	event := ReputationEvent{
		Timestamp: time.Now(),
		OldScore:  currentScore,
		NewScore:  newScore,
		Reason:    "slashing_" + reason.String(),
		Delta:     -penalty,
	}

	rt.history[address] = append(rt.history[address], event)
}

// ApplyDecay applies time-based reputation decay
func (rt *ReputationTracker) ApplyDecay(address string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	currentScore := rt.scores[address]
	if currentScore == 0 {
		return // No decay for uninitialized scores
	}

	// Apply decay towards neutral (5.0)
	neutral := rt.config.InitialReputation
	decayRate := rt.config.ReputationDecayRate

	if currentScore > neutral {
		// Decay towards neutral from above
		newScore := currentScore - (currentScore-neutral)*decayRate
		rt.scores[address] = newScore
	} else if currentScore < neutral {
		// Recover towards neutral from below
		newScore := currentScore + (neutral-currentScore)*decayRate
		rt.scores[address] = newScore
	}

	// Record event if significant change
	if math.Abs(rt.scores[address]-currentScore) > 0.01 {
		event := ReputationEvent{
			Timestamp: time.Now(),
			OldScore:  currentScore,
			NewScore:  rt.scores[address],
			Reason:    "decay",
			Delta:     rt.scores[address] - currentScore,
		}
		rt.history[address] = append(rt.history[address], event)
	}
}

// ApplyGlobalDecay applies decay to all validators
func (rt *ReputationTracker) ApplyGlobalDecay() {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	for address := range rt.scores {
		currentScore := rt.scores[address]
		neutral := rt.config.InitialReputation
		decayRate := rt.config.ReputationDecayRate

		if currentScore > neutral {
			rt.scores[address] = currentScore - (currentScore-neutral)*decayRate
		} else if currentScore < neutral {
			rt.scores[address] = currentScore + (neutral-currentScore)*decayRate
		}
	}

	rt.lastDecayApplied = time.Now()
}

// GetReputationHistory returns the reputation history for a validator
func (rt *ReputationTracker) GetReputationHistory(address string) []ReputationEvent {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	history, exists := rt.history[address]
	if !exists {
		return []ReputationEvent{}
	}

	// Return a copy to prevent external modification
	result := make([]ReputationEvent, len(history))
	copy(result, history)
	return result
}

// GetTopValidators returns the top N validators by reputation
func (rt *ReputationTracker) GetTopValidators(n int) []string {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	// Create slice of addresses with scores
	type validatorScore struct {
		address string
		score   float64
	}

	validators := make([]validatorScore, 0, len(rt.scores))
	for addr, score := range rt.scores {
		validators = append(validators, validatorScore{addr, score})
	}

	// Sort by score (descending)
	for i := 0; i < len(validators)-1; i++ {
		for j := i + 1; j < len(validators); j++ {
			if validators[j].score > validators[i].score {
				validators[i], validators[j] = validators[j], validators[i]
			}
		}
	}

	// Return top N addresses
	result := make([]string, 0, n)
	for i := 0; i < n && i < len(validators); i++ {
		result = append(result, validators[i].address)
	}

	return result
}
