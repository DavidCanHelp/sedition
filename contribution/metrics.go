// Package contribution includes metrics calculation
package contribution

import (
	"math"
	"time"
)

// MetricsCalculator calculates various metrics for contributions and validators
type MetricsCalculator struct {
	windowSize time.Duration
}

// NewMetricsCalculator creates a new metrics calculator
func NewMetricsCalculator() *MetricsCalculator {
	return &MetricsCalculator{
		windowSize: 7 * 24 * time.Hour, // 7 days default window
	}
}

// NewMetricsCalculatorWithWindow creates a new metrics calculator with custom window
func NewMetricsCalculatorWithWindow(window time.Duration) *MetricsCalculator {
	return &MetricsCalculator{
		windowSize: window,
	}
}

// ContributionMetrics holds calculated metrics for contributions
type ContributionMetrics struct {
	TotalContributions  int
	AverageQuality      float64
	MedianQuality       float64
	StandardDeviation   float64
	TotalLinesChanged   int
	AverageTestCoverage float64
	ContributionRate    float64 // Contributions per day
	HighQualityRatio    float64 // Ratio of high quality contributions
	ReviewedRatio       float64 // Ratio of reviewed contributions
	TypeDistribution    map[Type]int
}

// CalculateMetrics calculates metrics for a set of contributions
func (mc *MetricsCalculator) CalculateMetrics(contribs []Contribution) *ContributionMetrics {
	if len(contribs) == 0 {
		return &ContributionMetrics{
			TypeDistribution: make(map[Type]int),
		}
	}

	metrics := &ContributionMetrics{
		TotalContributions: len(contribs),
		TypeDistribution:   make(map[Type]int),
	}

	// Collect basic statistics
	qualities := make([]float64, len(contribs))
	totalQuality := 0.0
	totalLines := 0
	totalCoverage := 0.0
	highQualityCount := 0
	reviewedCount := 0

	var earliestTime, latestTime time.Time

	for i, contrib := range contribs {
		// Quality metrics
		qualities[i] = contrib.QualityScore
		totalQuality += contrib.QualityScore

		// Lines changed
		totalLines += contrib.LinesAdded + contrib.LinesModified + contrib.LinesDeleted

		// Test coverage
		totalCoverage += contrib.TestCoverage

		// High quality check (> 75)
		if contrib.QualityScore >= 75 {
			highQualityCount++
		}

		// Review check
		if contrib.PeerReviews > 0 {
			reviewedCount++
		}

		// Type distribution
		metrics.TypeDistribution[contrib.Type]++

		// Time tracking
		if earliestTime.IsZero() || contrib.Timestamp.Before(earliestTime) {
			earliestTime = contrib.Timestamp
		}
		if latestTime.IsZero() || contrib.Timestamp.After(latestTime) {
			latestTime = contrib.Timestamp
		}
	}

	// Calculate averages
	metrics.AverageQuality = totalQuality / float64(len(contribs))
	metrics.AverageTestCoverage = totalCoverage / float64(len(contribs))
	metrics.TotalLinesChanged = totalLines
	metrics.HighQualityRatio = float64(highQualityCount) / float64(len(contribs))
	metrics.ReviewedRatio = float64(reviewedCount) / float64(len(contribs))

	// Calculate median quality
	metrics.MedianQuality = mc.calculateMedian(qualities)

	// Calculate standard deviation
	metrics.StandardDeviation = mc.calculateStandardDeviation(qualities, metrics.AverageQuality)

	// Calculate contribution rate (per day)
	if !earliestTime.IsZero() && !latestTime.IsZero() {
		duration := latestTime.Sub(earliestTime)
		if duration > 0 {
			days := duration.Hours() / 24
			if days > 0 {
				metrics.ContributionRate = float64(len(contribs)) / days
			}
		}
	}

	return metrics
}

// calculateMedian calculates the median of a slice of values
func (mc *MetricsCalculator) calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Sort values
	sorted := make([]float64, len(values))
	copy(sorted, values)

	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Calculate median
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// calculateStandardDeviation calculates the standard deviation
func (mc *MetricsCalculator) calculateStandardDeviation(values []float64, mean float64) float64 {
	if len(values) <= 1 {
		return 0
	}

	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}

	variance := sumSquares / float64(len(values)-1)
	return math.Sqrt(variance)
}

// CalculateVelocity calculates the contribution velocity over time
func (mc *MetricsCalculator) CalculateVelocity(contribs []Contribution) float64 {
	if len(contribs) < 2 {
		return 0
	}

	// Filter contributions within window
	cutoff := time.Now().Add(-mc.windowSize)
	recentContribs := 0

	for _, contrib := range contribs {
		if contrib.Timestamp.After(cutoff) {
			recentContribs++
		}
	}

	// Calculate daily velocity
	days := mc.windowSize.Hours() / 24
	return float64(recentContribs) / days
}

// CalculateConsistency calculates how consistent contributions are over time
func (mc *MetricsCalculator) CalculateConsistency(contribs []Contribution) float64 {
	if len(contribs) < 2 {
		return 0
	}

	// Group contributions by day
	dayContribs := make(map[string]int)
	for _, contrib := range contribs {
		day := contrib.Timestamp.Format("2006-01-02")
		dayContribs[day]++
	}

	if len(dayContribs) == 0 {
		return 0
	}

	// Calculate average and standard deviation of daily contributions
	total := 0
	for _, count := range dayContribs {
		total += count
	}
	avg := float64(total) / float64(len(dayContribs))

	// Calculate coefficient of variation (lower is more consistent)
	sumSquares := 0.0
	for _, count := range dayContribs {
		diff := float64(count) - avg
		sumSquares += diff * diff
	}

	if avg == 0 {
		return 0
	}

	stdDev := math.Sqrt(sumSquares / float64(len(dayContribs)))
	coeffVar := stdDev / avg

	// Convert to consistency score (0-100, higher is better)
	consistency := math.Max(0, 100*(1-coeffVar))
	return consistency
}

// GetProductivityScore calculates an overall productivity score
func (mc *MetricsCalculator) GetProductivityScore(metrics *ContributionMetrics) float64 {
	if metrics.TotalContributions == 0 {
		return 0
	}

	// Weight different factors
	qualityWeight := 0.3
	volumeWeight := 0.2
	consistencyWeight := 0.2
	reviewWeight := 0.15
	coverageWeight := 0.15

	// Normalize volume (assume 10 contributions/day is excellent)
	volumeScore := math.Min(100, metrics.ContributionRate*10)

	// Calculate weighted score
	score := metrics.AverageQuality*qualityWeight +
		volumeScore*volumeWeight +
		metrics.ReviewedRatio*100*reviewWeight +
		metrics.AverageTestCoverage*coverageWeight +
		metrics.HighQualityRatio*100*consistencyWeight

	return math.Min(100, score)
}
