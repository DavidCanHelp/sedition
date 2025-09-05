// Package contribution includes quality analysis
package contribution

import (
	"math"

	"github.com/davidcanhelp/sedition/config"
)

// QualityAnalyzer analyzes contribution quality
type QualityAnalyzer struct {
	config *config.ConsensusConfig
}

// NewQualityAnalyzer creates a new quality analyzer
func NewQualityAnalyzer() *QualityAnalyzer {
	return &QualityAnalyzer{
		config: config.DefaultConsensusConfig(),
	}
}

// NewQualityAnalyzerWithConfig creates a new quality analyzer with custom config
func NewQualityAnalyzerWithConfig(cfg *config.ConsensusConfig) *QualityAnalyzer {
	return &QualityAnalyzer{
		config: cfg,
	}
}

// AnalyzeContribution analyzes a contribution and returns a quality score
func (qa *QualityAnalyzer) AnalyzeContribution(contrib Contribution) (float64, error) {
	var score float64

	// Base score from test coverage (0-25 points)
	testScore := contrib.TestCoverage * 0.25

	// Documentation score (0-20 points)
	docScore := contrib.Documentation * 0.20

	// Complexity score (0-25 points)
	// Lower complexity is better
	complexityScore := 0.0
	if contrib.Complexity > 0 {
		// Inverse relationship: lower complexity = higher score
		complexityScore = math.Max(0, 25-contrib.Complexity*0.5)
	} else {
		complexityScore = 25 // Perfect score if complexity is 0
	}

	// Type bonus (0-15 points)
	typeScore := qa.getTypeScore(contrib.Type)

	// Review score (0-15 points)
	reviewScore := 0.0
	if contrib.PeerReviews > 0 {
		// Average review score normalized to 15 points
		reviewScore = (contrib.ReviewScore / 5.0) * 15
	}

	// Calculate total score
	score = testScore + docScore + complexityScore + typeScore + reviewScore

	// Apply modifiers based on contribution size
	sizeModifier := qa.getSizeModifier(contrib)
	score *= sizeModifier

	// Ensure score is between 0 and 100
	score = math.Max(0, math.Min(100, score))

	return score, nil
}

// getTypeScore returns a score based on contribution type
func (qa *QualityAnalyzer) getTypeScore(contribType Type) float64 {
	switch contribType {
	case Security:
		return 15.0 // Maximum bonus for security contributions
	case Testing:
		return 12.0
	case CodeReview:
		return 10.0
	case CodeCommit, PullRequest:
		return 8.0
	case IssueResolution:
		return 7.0
	case Documentation:
		return 5.0
	default:
		return 3.0
	}
}

// getSizeModifier returns a modifier based on contribution size
func (qa *QualityAnalyzer) getSizeModifier(contrib Contribution) float64 {
	totalLines := contrib.LinesAdded + contrib.LinesModified + contrib.LinesDeleted

	switch {
	case totalLines == 0:
		return 0.5 // Minimal contribution
	case totalLines < 10:
		return 0.8 // Small contribution
	case totalLines < 50:
		return 1.0 // Normal contribution
	case totalLines < 200:
		return 1.1 // Substantial contribution
	case totalLines < 500:
		return 1.0 // Large contribution (no bonus to avoid gaming)
	default:
		return 0.9 // Very large contribution (slight penalty for potential complexity)
	}
}

// CalculateAverageQuality calculates average quality from multiple contributions
func (qa *QualityAnalyzer) CalculateAverageQuality(contribs []Contribution) float64 {
	if len(contribs) == 0 {
		return 0.0
	}

	totalQuality := 0.0
	for _, contrib := range contribs {
		totalQuality += contrib.QualityScore
	}

	return totalQuality / float64(len(contribs))
}

// IsHighQuality checks if a quality score meets the threshold
func (qa *QualityAnalyzer) IsHighQuality(score float64) bool {
	return score >= qa.config.QualityThreshold
}

// GetQualityTier returns the quality tier for a score
func (qa *QualityAnalyzer) GetQualityTier(score float64) string {
	switch {
	case score >= 90:
		return "Exceptional"
	case score >= 75:
		return "High"
	case score >= 60:
		return "Good"
	case score >= 40:
		return "Average"
	case score >= 20:
		return "Below Average"
	default:
		return "Poor"
	}
}

// AnalyzeTrend analyzes quality trend over time
func (qa *QualityAnalyzer) AnalyzeTrend(contribs []Contribution) string {
	if len(contribs) < 3 {
		return "Insufficient data"
	}

	// Calculate average quality for first and last third
	firstThird := len(contribs) / 3
	lastThirdStart := len(contribs) - firstThird

	firstAvg := qa.CalculateAverageQuality(contribs[:firstThird])
	lastAvg := qa.CalculateAverageQuality(contribs[lastThirdStart:])

	difference := lastAvg - firstAvg

	switch {
	case difference > 10:
		return "Improving"
	case difference > 0:
		return "Slightly Improving"
	case difference < -10:
		return "Declining"
	case difference < 0:
		return "Slightly Declining"
	default:
		return "Stable"
	}
}
