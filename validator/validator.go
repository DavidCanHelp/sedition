// Package validator manages validator state and reputation
package validator

import (
	"math/big"
	"time"

	"github.com/davidcanhelp/sedition/contribution"
)

// Validator represents a network validator/contributor
type Validator struct {
	Address          string
	TokenStake       *big.Int                    // Raw token amount staked
	ReputationScore  float64                     // Reputation score (0.0 to 10.0)
	RecentContribs   []contribution.Contribution // Recent contributions for this epoch
	TotalStake       *big.Int                    // Calculated total stake including all factors
	LastActivityTime time.Time                   // Last contribution timestamp
	SlashingHistory  []SlashingEvent             // History of slashing events
	IsActive         bool                        // Whether validator is currently active
}

// SlashingEvent represents a slashing incident
type SlashingEvent struct {
	Timestamp     time.Time
	Reason        SlashingReason
	AmountSlashed *big.Int
	Evidence      string
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

// String returns the string representation of a slashing reason
func (s SlashingReason) String() string {
	switch s {
	case MaliciousCode:
		return "MaliciousCode"
	case FalseContribution:
		return "FalseContribution"
	case DoubleProposal:
		return "DoubleProposal"
	case NetworkAttack:
		return "NetworkAttack"
	case QualityViolation:
		return "QualityViolation"
	default:
		return "Unknown"
	}
}

// Stats holds statistics for a validator
type Stats struct {
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

// NewValidator creates a new validator with initial values
func NewValidator(address string, stake *big.Int, initialReputation float64) *Validator {
	return &Validator{
		Address:          address,
		TokenStake:       new(big.Int).Set(stake),
		ReputationScore:  initialReputation,
		RecentContribs:   make([]contribution.Contribution, 0),
		TotalStake:       new(big.Int),
		LastActivityTime: time.Now(),
		SlashingHistory:  make([]SlashingEvent, 0),
		IsActive:         true,
	}
}

// AddContribution adds a new contribution to the validator's history
func (v *Validator) AddContribution(contrib contribution.Contribution) {
	v.RecentContribs = append(v.RecentContribs, contrib)
	v.LastActivityTime = contrib.Timestamp
}

// ApplySlashing applies a slashing penalty to the validator
func (v *Validator) ApplySlashing(amount *big.Int, reason SlashingReason, evidence string) {
	v.TokenStake = v.TokenStake.Sub(v.TokenStake, amount)

	event := SlashingEvent{
		Timestamp:     time.Now(),
		Reason:        reason,
		AmountSlashed: amount,
		Evidence:      evidence,
	}

	v.SlashingHistory = append(v.SlashingHistory, event)
}

// GetRecentContributionQuality calculates the average quality of recent contributions
func (v *Validator) GetRecentContributionQuality(since time.Time) float64 {
	if len(v.RecentContribs) == 0 {
		return 0.0
	}

	totalQuality := 0.0
	count := 0

	for _, contrib := range v.RecentContribs {
		if contrib.Timestamp.After(since) {
			totalQuality += contrib.QualityScore
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return totalQuality / float64(count)
}

// CleanupOldContributions removes contributions older than the specified time
func (v *Validator) CleanupOldContributions(cutoff time.Time) {
	newContribs := make([]contribution.Contribution, 0)

	for _, contrib := range v.RecentContribs {
		if contrib.Timestamp.After(cutoff) {
			newContribs = append(newContribs, contrib)
		}
	}

	v.RecentContribs = newContribs
}
