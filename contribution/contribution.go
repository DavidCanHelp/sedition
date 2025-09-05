// Package contribution handles code contribution tracking and quality analysis
package contribution

import (
	"time"
)

// Contribution represents a single code contribution
type Contribution struct {
	ID            string
	Timestamp     time.Time
	Type          Type
	LinesAdded    int
	LinesModified int
	LinesDeleted  int
	TestCoverage  float64 // Percentage of test coverage added
	Complexity    float64 // Cyclomatic complexity score
	Documentation float64 // Documentation coverage score
	QualityScore  float64 // Overall quality score (0-100)
	PeerReviews   int     // Number of peer reviews received
	ReviewScore   float64 // Average review score
}

// Type defines the type of contribution
type Type int

const (
	CodeCommit Type = iota
	PullRequest
	IssueResolution
	Documentation
	Testing
	CodeReview
	Security
)

// String returns the string representation of a contribution type
func (t Type) String() string {
	switch t {
	case CodeCommit:
		return "CodeCommit"
	case PullRequest:
		return "PullRequest"
	case IssueResolution:
		return "IssueResolution"
	case Documentation:
		return "Documentation"
	case Testing:
		return "Testing"
	case CodeReview:
		return "CodeReview"
	case Security:
		return "Security"
	default:
		return "Unknown"
	}
}

// NewContribution creates a new contribution with basic information
func NewContribution(id string, contribType Type) Contribution {
	return Contribution{
		ID:        id,
		Timestamp: time.Now(),
		Type:      contribType,
	}
}

// CalculateImpact calculates the impact score of a contribution
func (c *Contribution) CalculateImpact() float64 {
	// Base impact from lines changed
	linesImpact := float64(c.LinesAdded+c.LinesModified) * 0.1

	// Type multiplier
	var typeMultiplier float64
	switch c.Type {
	case Security:
		typeMultiplier = 2.0
	case Testing:
		typeMultiplier = 1.5
	case CodeReview:
		typeMultiplier = 1.3
	case CodeCommit, PullRequest:
		typeMultiplier = 1.0
	case Documentation:
		typeMultiplier = 0.8
	default:
		typeMultiplier = 0.5
	}

	// Quality factor
	qualityFactor := c.QualityScore / 100.0

	// Test coverage bonus
	coverageBonus := c.TestCoverage / 100.0

	// Peer review factor
	reviewFactor := 1.0
	if c.PeerReviews > 0 {
		reviewFactor = 1.0 + (c.ReviewScore/5.0)*0.2
	}

	impact := linesImpact * typeMultiplier * qualityFactor * (1 + coverageBonus) * reviewFactor

	// Cap at 100
	if impact > 100 {
		impact = 100
	}

	return impact
}

// IsHighQuality determines if the contribution meets high quality standards
func (c *Contribution) IsHighQuality(threshold float64) bool {
	return c.QualityScore >= threshold
}

// HasTestCoverage checks if the contribution includes test coverage
func (c *Contribution) HasTestCoverage() bool {
	return c.TestCoverage > 0
}

// IsReviewed checks if the contribution has been peer reviewed
func (c *Contribution) IsReviewed() bool {
	return c.PeerReviews > 0
}

// GetAge returns the age of the contribution
func (c *Contribution) GetAge() time.Duration {
	return time.Since(c.Timestamp)
}

// Summary returns a summary of the contribution
func (c *Contribution) Summary() map[string]interface{} {
	return map[string]interface{}{
		"id":            c.ID,
		"type":          c.Type.String(),
		"timestamp":     c.Timestamp,
		"quality_score": c.QualityScore,
		"impact":        c.CalculateImpact(),
		"lines_changed": c.LinesAdded + c.LinesModified + c.LinesDeleted,
		"test_coverage": c.TestCoverage,
		"peer_reviews":  c.PeerReviews,
		"age_hours":     c.GetAge().Hours(),
	}
}
