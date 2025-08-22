// Package poc implements the reputation tracking system for the PoC consensus
// This system tracks contributor reputation over time using multiple factors:
// - Contribution quality history
// - Peer review performance
// - Community engagement
// - Long-term consistency
// - Recovery from mistakes
//
// Mathematical Foundation:
// Reputation = BaseReputation × QualityFactor × ConsistencyFactor × PeerFactor × TimeFactor
// With exponential decay over time to ensure recent performance matters most
package poc

import (
	"math"
	"time"
)

// ReputationTracker manages reputation scores for all contributors
type ReputationTracker struct {
	// Reputation records for each contributor
	reputations map[string]*ReputationRecord
	
	// System parameters
	baseReputation      float64 // Starting reputation for new contributors
	maxReputation       float64 // Maximum achievable reputation
	minReputation       float64 // Minimum reputation (prevents permanent exclusion)
	decayRate          float64 // Daily decay rate (0.0 to 1.0)
	recoveryRate       float64 // Rate of recovery from low reputation
	
	// Quality thresholds for reputation adjustments
	excellentThreshold  float64 // Quality score threshold for reputation boost
	poorThreshold      float64 // Quality score threshold for reputation penalty
	
	// Peer review weights
	reviewerWeight     float64 // Weight of being a good reviewer
	revieweeWeight     float64 // Weight of receiving good reviews
}

// ReputationRecord holds detailed reputation data for a contributor
type ReputationRecord struct {
	Address             string
	CurrentReputation   float64
	PeakReputation      float64      // Highest reputation ever achieved
	BaselineReputation  float64      // Moving average baseline
	LastUpdate          time.Time
	
	// Historical tracking
	QualityHistory      []QualityEvent
	PeerReviewHistory   []PeerReviewEvent
	SlashingHistory     []ReputationSlashing
	
	// Performance metrics
	TotalContributions  int
	QualityAverage      float64
	ConsistencyScore    float64   // Measure of consistent performance
	PeerRating         float64   // Average rating from peer reviews
	
	// Recovery tracking
	IsRecovering       bool      // Whether contributor is in recovery mode
	RecoveryStartTime  time.Time // When recovery mode started
	PreSlashingRep     float64   // Reputation before major slashing event
}

// QualityEvent records a contribution quality assessment
type QualityEvent struct {
	Timestamp     time.Time
	QualityScore  float64
	ContribType   ContributionType
	ImpactWeight  float64 // Weight based on contribution size/impact
}

// PeerReviewEvent records peer review activity
type PeerReviewEvent struct {
	Timestamp    time.Time
	IsReviewer   bool    // true if this person was the reviewer
	Rating       float64 // Rating given or received (1-5)
	ReviewType   ReviewType
}

// ReviewType defines the type of peer review
type ReviewType int

const (
	CodeReviewType ReviewType = iota
	DesignReview
	SecurityReview
	PerformanceReview
	DocumentationReview
)

// ReputationSlashing records reputation penalties
type ReputationSlashing struct {
	Timestamp       time.Time
	Reason          SlashingReason
	AmountSlashed   float64
	PreSlashingRep  float64
	PostSlashingRep float64
	RecoveryPlan    string
}

// NewReputationTracker creates a new reputation tracking system
func NewReputationTracker() *ReputationTracker {
	return &ReputationTracker{
		reputations:        make(map[string]*ReputationRecord),
		baseReputation:     5.0,  // Start at neutral reputation (1-10 scale)
		maxReputation:      10.0, // Maximum reputation
		minReputation:      0.5,  // Minimum to prevent total exclusion
		decayRate:         0.005, // 0.5% daily decay
		recoveryRate:      0.02,  // 2% daily recovery boost when in recovery
		excellentThreshold: 85.0, // Quality scores above 85 boost reputation
		poorThreshold:     60.0,  // Quality scores below 60 hurt reputation
		reviewerWeight:    0.3,   // Weight for being a good reviewer
		revieweeWeight:    0.7,   // Weight for receiving good reviews
	}
}

// InitializeReputation sets up initial reputation for a new contributor
func (rt *ReputationTracker) InitializeReputation(address string) {
	record := &ReputationRecord{
		Address:            address,
		CurrentReputation:  rt.baseReputation,
		PeakReputation:     rt.baseReputation,
		BaselineReputation: rt.baseReputation,
		LastUpdate:         time.Now(),
		QualityHistory:     make([]QualityEvent, 0),
		PeerReviewHistory:  make([]PeerReviewEvent, 0),
		SlashingHistory:    make([]ReputationSlashing, 0),
		TotalContributions: 0,
		QualityAverage:     rt.baseReputation * 10, // Convert to quality scale
		ConsistencyScore:   1.0,
		PeerRating:        rt.baseReputation,
		IsRecovering:      false,
	}
	
	rt.reputations[address] = record
}

// UpdateReputation updates reputation based on a new contribution
func (rt *ReputationTracker) UpdateReputation(address string, contrib Contribution) {
	record, exists := rt.reputations[address]
	if !exists {
		rt.InitializeReputation(address)
		record = rt.reputations[address]
	}
	
	// Apply time decay first
	rt.applyTimeDecay(record)
	
	// Calculate quality impact
	qualityImpact := rt.calculateQualityImpact(contrib.QualityScore, contrib.Type)
	
	// Record quality event
	qualityEvent := QualityEvent{
		Timestamp:    contrib.Timestamp,
		QualityScore: contrib.QualityScore,
		ContribType:  contrib.Type,
		ImpactWeight: rt.calculateImpactWeight(contrib),
	}
	record.QualityHistory = append(record.QualityHistory, qualityEvent)
	
	// Update metrics
	rt.updateQualityAverage(record)
	rt.updateConsistencyScore(record)
	
	// Apply reputation change
	oldReputation := record.CurrentReputation
	record.CurrentReputation += qualityImpact
	
	// Apply bounds
	record.CurrentReputation = math.Max(rt.minReputation, 
		math.Min(rt.maxReputation, record.CurrentReputation))
	
	// Update peak if necessary
	if record.CurrentReputation > record.PeakReputation {
		record.PeakReputation = record.CurrentReputation
	}
	
	// Update baseline (slowly moving average)
	alpha := 0.1 // Smoothing factor for moving average
	record.BaselineReputation = alpha*record.CurrentReputation + (1-alpha)*record.BaselineReputation
	
	// Check for recovery completion
	if record.IsRecovering {
		rt.checkRecoveryCompletion(record)
	}
	
	record.TotalContributions++
	record.LastUpdate = time.Now()
	
	// Log significant changes
	if math.Abs(record.CurrentReputation - oldReputation) > 0.5 {
		// Significant reputation change - could log this for transparency
	}
}

// calculateQualityImpact calculates the reputation impact of a quality score
func (rt *ReputationTracker) calculateQualityImpact(qualityScore float64, contribType ContributionType) float64 {
	// Base impact calculation
	var impact float64
	
	if qualityScore >= rt.excellentThreshold {
		// Excellent contributions boost reputation
		impact = 0.1 + (qualityScore-rt.excellentThreshold)*0.01 // 0.1 to 0.25 boost
	} else if qualityScore >= rt.poorThreshold {
		// Average contributions have minimal impact
		impact = (qualityScore - rt.poorThreshold) * 0.002 // -0.002 to 0.05
	} else {
		// Poor contributions hurt reputation
		impact = -0.1 - (rt.poorThreshold-qualityScore)*0.005 // -0.1 to -0.225 penalty
	}
	
	// Apply contribution type multiplier
	switch contribType {
	case Security:
		impact *= 1.5 // Security contributions have higher impact
	case Testing:
		impact *= 1.2 // Testing contributions are valuable
	case Documentation:
		impact *= 1.1 // Documentation is important
	case CodeReview:
		impact *= 1.3 // Code reviews help the community
	case IssueResolution:
		impact *= 1.15 // Issue resolution demonstrates capability
	default:
		impact *= 1.0 // Regular code contributions
	}
	
	return impact
}

// calculateImpactWeight calculates the weight of a contribution based on its size and complexity
func (rt *ReputationTracker) calculateImpactWeight(contrib Contribution) float64 {
	// Base weight from lines changed
	linesChanged := contrib.LinesAdded + contrib.LinesModified
	baseWeight := math.Log10(float64(linesChanged) + 1) // Logarithmic scaling
	
	// Adjust for contribution type
	switch contrib.Type {
	case Security:
		return baseWeight * 2.0 // Security changes have high impact regardless of size
	case Documentation:
		return math.Min(baseWeight * 0.8, 2.0) // Documentation impact caps out lower
	case Testing:
		return baseWeight * 1.5 // Testing has good impact scaling
	default:
		return math.Min(baseWeight, 5.0) // Cap regular contributions
	}
}

// RecordPeerReview records a peer review event
func (rt *ReputationTracker) RecordPeerReview(address string, isReviewer bool, rating float64, reviewType ReviewType) {
	record, exists := rt.reputations[address]
	if !exists {
		rt.InitializeReputation(address)
		record = rt.reputations[address]
	}
	
	// Apply time decay first
	rt.applyTimeDecay(record)
	
	// Record the review event
	reviewEvent := PeerReviewEvent{
		Timestamp:  time.Now(),
		IsReviewer: isReviewer,
		Rating:     rating,
		ReviewType: reviewType,
	}
	record.PeerReviewHistory = append(record.PeerReviewHistory, reviewEvent)
	
	// Calculate reputation impact
	var impact float64
	if isReviewer {
		// Being a good reviewer (giving thorough, helpful reviews)
		if rating >= 4.0 {
			impact = 0.05 // Small boost for being a helpful reviewer
		} else if rating < 2.0 {
			impact = -0.02 // Small penalty for poor reviews
		}
		impact *= rt.reviewerWeight
	} else {
		// Receiving good reviews (having quality contributions reviewed positively)
		if rating >= 4.0 {
			impact = 0.08 // Boost for receiving good reviews
		} else if rating < 2.0 {
			impact = -0.05 // Penalty for receiving poor reviews
		} else {
			impact = (rating - 3.0) * 0.02 // Linear scaling around neutral
		}
		impact *= rt.revieweeWeight
	}
	
	// Apply the reputation change
	record.CurrentReputation += impact
	record.CurrentReputation = math.Max(rt.minReputation, 
		math.Min(rt.maxReputation, record.CurrentReputation))
	
	// Update peer rating average
	rt.updatePeerRating(record)
	
	record.LastUpdate = time.Now()
}

// ApplySlashing applies a reputation penalty for malicious behavior
func (rt *ReputationTracker) ApplySlashing(address string, reason SlashingReason) {
	record, exists := rt.reputations[address]
	if !exists {
		rt.InitializeReputation(address)
		record = rt.reputations[address]
	}
	
	// Calculate slashing amount based on reason
	var slashingMultiplier float64
	var recoveryPlan string
	
	switch reason {
	case MaliciousCode:
		slashingMultiplier = 0.5 // Slash 50% of current reputation
		recoveryPlan = "Demonstrate consistent high-quality contributions over 30 days"
	case FalseContribution:
		slashingMultiplier = 0.3 // Slash 30% of current reputation
		recoveryPlan = "Submit 5 verified contributions with >80% quality score"
	case DoubleProposal:
		slashingMultiplier = 0.4 // Slash 40% of current reputation
		recoveryPlan = "Demonstrate network participation without protocol violations for 14 days"
	case NetworkAttack:
		slashingMultiplier = 0.7 // Slash 70% of current reputation
		recoveryPlan = "Undergo community review and demonstrate positive contributions for 60 days"
	case QualityViolation:
		slashingMultiplier = 0.2 // Slash 20% of current reputation
		recoveryPlan = "Improve contribution quality with >75% average over next 10 contributions"
	default:
		slashingMultiplier = 0.25
		recoveryPlan = "Demonstrate improved behavior and contribution quality"
	}
	
	preSlashingRep := record.CurrentReputation
	slashingAmount := record.CurrentReputation * slashingMultiplier
	record.CurrentReputation -= slashingAmount
	
	// Ensure minimum reputation
	record.CurrentReputation = math.Max(rt.minReputation, record.CurrentReputation)
	
	// Record the slashing event
	slashingEvent := ReputationSlashing{
		Timestamp:       time.Now(),
		Reason:          reason,
		AmountSlashed:   slashingAmount,
		PreSlashingRep:  preSlashingRep,
		PostSlashingRep: record.CurrentReputation,
		RecoveryPlan:    recoveryPlan,
	}
	record.SlashingHistory = append(record.SlashingHistory, slashingEvent)
	
	// Enter recovery mode for significant slashing
	if slashingMultiplier >= 0.3 {
		record.IsRecovering = true
		record.RecoveryStartTime = time.Now()
		record.PreSlashingRep = preSlashingRep
	}
	
	record.LastUpdate = time.Now()
}

// applyTimeDecay applies exponential reputation decay over time
func (rt *ReputationTracker) applyTimeDecay(record *ReputationRecord) {
	now := time.Now()
	daysSinceUpdate := now.Sub(record.LastUpdate).Hours() / 24.0
	
	if daysSinceUpdate <= 0 {
		return // No decay if updated recently
	}
	
	// Apply exponential decay
	decayFactor := math.Pow(1.0-rt.decayRate, daysSinceUpdate)
	
	// Decay towards baseline, not towards zero
	targetReputation := record.BaselineReputation
	if record.CurrentReputation > targetReputation {
		// Decay high reputation towards baseline
		record.CurrentReputation = targetReputation + 
			(record.CurrentReputation-targetReputation)*decayFactor
	} else if record.CurrentReputation < targetReputation {
		// Allow recovery towards baseline (slower than decay)
		recoveryRate := rt.decayRate * 0.5 // Recovery is slower than decay
		if record.IsRecovering {
			recoveryRate *= 2.0 // Faster recovery for those in recovery mode
		}
		recoveryFactor := math.Pow(1.0+recoveryRate, daysSinceUpdate)
		record.CurrentReputation = targetReputation - 
			(targetReputation-record.CurrentReputation)/recoveryFactor
	}
	
	// Ensure bounds
	record.CurrentReputation = math.Max(rt.minReputation, 
		math.Min(rt.maxReputation, record.CurrentReputation))
}

// ApplyDecay applies daily reputation decay (called by consensus engine)
func (rt *ReputationTracker) ApplyDecay(address string) {
	record, exists := rt.reputations[address]
	if !exists {
		return
	}
	
	rt.applyTimeDecay(record)
}

// updateQualityAverage updates the rolling average of contribution quality
func (rt *ReputationTracker) updateQualityAverage(record *ReputationRecord) {
	if len(record.QualityHistory) == 0 {
		return
	}
	
	// Calculate weighted average of recent contributions (last 30 days)
	cutoffTime := time.Now().Add(-30 * 24 * time.Hour)
	totalScore := 0.0
	totalWeight := 0.0
	
	for _, event := range record.QualityHistory {
		if event.Timestamp.After(cutoffTime) {
			weight := event.ImpactWeight
			totalScore += event.QualityScore * weight
			totalWeight += weight
		}
	}
	
	if totalWeight > 0 {
		record.QualityAverage = totalScore / totalWeight
	}
}

// updateConsistencyScore calculates how consistent the contributor's quality is
func (rt *ReputationTracker) updateConsistencyScore(record *ReputationRecord) {
	if len(record.QualityHistory) < 3 {
		record.ConsistencyScore = 1.0 // Not enough data
		return
	}
	
	// Calculate standard deviation of recent quality scores
	recentEvents := make([]float64, 0)
	cutoffTime := time.Now().Add(-30 * 24 * time.Hour)
	
	for _, event := range record.QualityHistory {
		if event.Timestamp.After(cutoffTime) {
			recentEvents = append(recentEvents, event.QualityScore)
		}
	}
	
	if len(recentEvents) < 3 {
		record.ConsistencyScore = 1.0
		return
	}
	
	// Calculate mean
	mean := 0.0
	for _, score := range recentEvents {
		mean += score
	}
	mean /= float64(len(recentEvents))
	
	// Calculate standard deviation
	variance := 0.0
	for _, score := range recentEvents {
		variance += math.Pow(score-mean, 2)
	}
	variance /= float64(len(recentEvents))
	stdDev := math.Sqrt(variance)
	
	// Convert to consistency score (lower std dev = higher consistency)
	// Map std dev 0-25 to consistency 1.0-0.5
	record.ConsistencyScore = math.Max(0.5, 1.0-(stdDev/50.0))
}

// updatePeerRating updates the average peer rating
func (rt *ReputationTracker) updatePeerRating(record *ReputationRecord) {
	if len(record.PeerReviewHistory) == 0 {
		return
	}
	
	// Calculate weighted average of peer reviews (recent reviews weigh more)
	totalRating := 0.0
	totalWeight := 0.0
	now := time.Now()
	
	for _, review := range record.PeerReviewHistory {
		if !review.IsReviewer { // Only count reviews received, not given
			// Weight decreases with age (more recent reviews matter more)
			daysSince := now.Sub(review.Timestamp).Hours() / 24.0
			weight := math.Exp(-daysSince / 30.0) // Exponential decay over 30 days
			
			totalRating += review.Rating * weight
			totalWeight += weight
		}
	}
	
	if totalWeight > 0 {
		record.PeerRating = totalRating / totalWeight
	}
}

// checkRecoveryCompletion checks if a contributor has completed recovery
func (rt *ReputationTracker) checkRecoveryCompletion(record *ReputationRecord) {
	if !record.IsRecovering {
		return
	}
	
	// Check if reputation has recovered to 80% of pre-slashing level
	recoveryThreshold := record.PreSlashingRep * 0.8
	
	// Or if sufficient time has passed with good behavior
	daysSinceRecovery := time.Now().Sub(record.RecoveryStartTime).Hours() / 24.0
	
	if record.CurrentReputation >= recoveryThreshold || 
	   (daysSinceRecovery >= 30 && record.CurrentReputation >= record.BaselineReputation) {
		record.IsRecovering = false
	}
}

// GetReputation returns the current reputation score for an address
func (rt *ReputationTracker) GetReputation(address string) float64 {
	record, exists := rt.reputations[address]
	if !exists {
		return rt.baseReputation
	}
	
	// Apply time decay before returning
	rt.applyTimeDecay(record)
	return record.CurrentReputation
}

// GetDetailedReputation returns detailed reputation information
func (rt *ReputationTracker) GetDetailedReputation(address string) *ReputationRecord {
	record, exists := rt.reputations[address]
	if !exists {
		return nil
	}
	
	// Apply time decay before returning
	rt.applyTimeDecay(record)
	
	// Return a copy to prevent external modification
	return &ReputationRecord{
		Address:            record.Address,
		CurrentReputation:  record.CurrentReputation,
		PeakReputation:     record.PeakReputation,
		BaselineReputation: record.BaselineReputation,
		LastUpdate:         record.LastUpdate,
		TotalContributions: record.TotalContributions,
		QualityAverage:     record.QualityAverage,
		ConsistencyScore:   record.ConsistencyScore,
		PeerRating:        record.PeerRating,
		IsRecovering:      record.IsRecovering,
		RecoveryStartTime: record.RecoveryStartTime,
		PreSlashingRep:    record.PreSlashingRep,
		// Note: Not copying history slices for security/performance
	}
}

// GetTopContributors returns the top N contributors by reputation
func (rt *ReputationTracker) GetTopContributors(n int) []*ReputationRecord {
	// Apply decay to all records first
	for _, record := range rt.reputations {
		rt.applyTimeDecay(record)
	}
	
	// Create slice of all contributors
	contributors := make([]*ReputationRecord, 0, len(rt.reputations))
	for _, record := range rt.reputations {
		contributors = append(contributors, record)
	}
	
	// Sort by current reputation (bubble sort for simplicity)
	for i := 0; i < len(contributors)-1; i++ {
		for j := 0; j < len(contributors)-i-1; j++ {
			if contributors[j].CurrentReputation < contributors[j+1].CurrentReputation {
				contributors[j], contributors[j+1] = contributors[j+1], contributors[j]
			}
		}
	}
	
	// Return top N
	if n > len(contributors) {
		n = len(contributors)
	}
	
	result := make([]*ReputationRecord, n)
	for i := 0; i < n; i++ {
		// Return copies to prevent external modification
		result[i] = rt.GetDetailedReputation(contributors[i].Address)
	}
	
	return result
}