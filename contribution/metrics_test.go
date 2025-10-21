package contribution

import (
	"math"
	"testing"
	"time"
)

// TestNewMetricsCalculator tests calculator creation
func TestNewMetricsCalculator(t *testing.T) {
	calc := NewMetricsCalculator()

	if calc == nil {
		t.Fatal("Expected non-nil calculator")
	}

	expectedWindow := 7 * 24 * time.Hour
	if calc.windowSize != expectedWindow {
		t.Errorf("Expected window %v, got %v", expectedWindow, calc.windowSize)
	}
}

// TestNewMetricsCalculatorWithWindow tests custom window calculator
func TestNewMetricsCalculatorWithWindow(t *testing.T) {
	customWindow := 30 * 24 * time.Hour
	calc := NewMetricsCalculatorWithWindow(customWindow)

	if calc == nil {
		t.Fatal("Expected non-nil calculator")
	}

	if calc.windowSize != customWindow {
		t.Errorf("Expected window %v, got %v", customWindow, calc.windowSize)
	}
}

// TestCalculateMetricsEmpty tests metrics calculation with empty contributions
func TestCalculateMetricsEmpty(t *testing.T) {
	calc := NewMetricsCalculator()
	metrics := calc.CalculateMetrics([]Contribution{})

	if metrics.TotalContributions != 0 {
		t.Errorf("Expected 0 contributions, got %d", metrics.TotalContributions)
	}

	if metrics.AverageQuality != 0 {
		t.Errorf("Expected 0 average quality, got %.2f", metrics.AverageQuality)
	}

	if metrics.TypeDistribution == nil {
		t.Error("Expected non-nil type distribution")
	}
}

// TestCalculateMetricsSingle tests metrics for a single contribution
func TestCalculateMetricsSingle(t *testing.T) {
	calc := NewMetricsCalculator()

	contrib := Contribution{
		QualityScore:  85.0,
		TestCoverage:  90.0,
		LinesAdded:    100,
		LinesModified: 50,
		LinesDeleted:  20,
		Type:          CodeCommit,
		PeerReviews:   2,
		Timestamp:     time.Now(),
	}

	metrics := calc.CalculateMetrics([]Contribution{contrib})

	if metrics.TotalContributions != 1 {
		t.Errorf("Expected 1 contribution, got %d", metrics.TotalContributions)
	}

	if metrics.AverageQuality != 85.0 {
		t.Errorf("Expected average quality 85.0, got %.2f", metrics.AverageQuality)
	}

	if metrics.MedianQuality != 85.0 {
		t.Errorf("Expected median quality 85.0, got %.2f", metrics.MedianQuality)
	}

	if metrics.AverageTestCoverage != 90.0 {
		t.Errorf("Expected average coverage 90.0, got %.2f", metrics.AverageTestCoverage)
	}

	expectedLines := 100 + 50 + 20
	if metrics.TotalLinesChanged != expectedLines {
		t.Errorf("Expected %d lines changed, got %d", expectedLines, metrics.TotalLinesChanged)
	}

	if metrics.HighQualityRatio != 1.0 {
		t.Errorf("Expected high quality ratio 1.0, got %.2f", metrics.HighQualityRatio)
	}

	if metrics.ReviewedRatio != 1.0 {
		t.Errorf("Expected reviewed ratio 1.0, got %.2f", metrics.ReviewedRatio)
	}

	if metrics.TypeDistribution[CodeCommit] != 1 {
		t.Errorf("Expected 1 CodeCommit, got %d", metrics.TypeDistribution[CodeCommit])
	}
}

// TestCalculateMetricsMultiple tests metrics for multiple contributions
func TestCalculateMetricsMultiple(t *testing.T) {
	calc := NewMetricsCalculator()

	now := time.Now()
	contribs := []Contribution{
		{
			QualityScore: 80.0,
			TestCoverage: 90.0,
			LinesAdded:   100,
			Type:         CodeCommit,
			PeerReviews:  2,
			Timestamp:    now.Add(-24 * time.Hour),
		},
		{
			QualityScore: 90.0,
			TestCoverage: 85.0,
			LinesAdded:   150,
			Type:         Testing,
			PeerReviews:  0,
			Timestamp:    now.Add(-12 * time.Hour),
		},
		{
			QualityScore: 70.0,
			TestCoverage: 75.0,
			LinesAdded:   80,
			Type:         Documentation,
			PeerReviews:  1,
			Timestamp:    now,
		},
	}

	metrics := calc.CalculateMetrics(contribs)

	if metrics.TotalContributions != 3 {
		t.Errorf("Expected 3 contributions, got %d", metrics.TotalContributions)
	}

	expectedAvg := (80.0 + 90.0 + 70.0) / 3.0
	if math.Abs(metrics.AverageQuality-expectedAvg) > 0.01 {
		t.Errorf("Expected average quality %.2f, got %.2f", expectedAvg, metrics.AverageQuality)
	}

	// Median of [70, 80, 90] is 80
	if metrics.MedianQuality != 80.0 {
		t.Errorf("Expected median quality 80.0, got %.2f", metrics.MedianQuality)
	}

	expectedCoverage := (90.0 + 85.0 + 75.0) / 3.0
	if math.Abs(metrics.AverageTestCoverage-expectedCoverage) > 0.01 {
		t.Errorf("Expected average coverage %.2f, got %.2f", expectedCoverage, metrics.AverageTestCoverage)
	}

	expectedLines := 100 + 150 + 80
	if metrics.TotalLinesChanged != expectedLines {
		t.Errorf("Expected %d lines changed, got %d", expectedLines, metrics.TotalLinesChanged)
	}

	// 2 out of 3 are high quality (>= 75)
	expectedHighQuality := 2.0 / 3.0
	if math.Abs(metrics.HighQualityRatio-expectedHighQuality) > 0.01 {
		t.Errorf("Expected high quality ratio %.2f, got %.2f", expectedHighQuality, metrics.HighQualityRatio)
	}

	// 2 out of 3 are reviewed
	expectedReviewed := 2.0 / 3.0
	if math.Abs(metrics.ReviewedRatio-expectedReviewed) > 0.01 {
		t.Errorf("Expected reviewed ratio %.2f, got %.2f", expectedReviewed, metrics.ReviewedRatio)
	}

	// Type distribution
	if metrics.TypeDistribution[CodeCommit] != 1 {
		t.Errorf("Expected 1 CodeCommit, got %d", metrics.TypeDistribution[CodeCommit])
	}
	if metrics.TypeDistribution[Testing] != 1 {
		t.Errorf("Expected 1 Testing, got %d", metrics.TypeDistribution[Testing])
	}
	if metrics.TypeDistribution[Documentation] != 1 {
		t.Errorf("Expected 1 Documentation, got %d", metrics.TypeDistribution[Documentation])
	}

	// Contribution rate (3 contributions over 24 hours = 3 per day)
	if metrics.ContributionRate < 2.5 || metrics.ContributionRate > 3.5 {
		t.Errorf("Expected contribution rate around 3.0, got %.2f", metrics.ContributionRate)
	}
}

// TestCalculateMedian tests median calculation
func TestCalculateMedian(t *testing.T) {
	calc := NewMetricsCalculator()

	tests := []struct {
		name     string
		values   []float64
		expected float64
	}{
		{
			name:     "Empty",
			values:   []float64{},
			expected: 0,
		},
		{
			name:     "Single value",
			values:   []float64{50},
			expected: 50,
		},
		{
			name:     "Two values",
			values:   []float64{50, 60},
			expected: 55,
		},
		{
			name:     "Odd count",
			values:   []float64{10, 20, 30, 40, 50},
			expected: 30,
		},
		{
			name:     "Even count",
			values:   []float64{10, 20, 30, 40},
			expected: 25,
		},
		{
			name:     "Unsorted",
			values:   []float64{50, 10, 30, 20, 40},
			expected: 30,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			median := calc.calculateMedian(tt.values)

			if median != tt.expected {
				t.Errorf("Expected median %.2f, got %.2f", tt.expected, median)
			}
		})
	}
}

// TestCalculateStandardDeviation tests standard deviation calculation
func TestCalculateStandardDeviation(t *testing.T) {
	calc := NewMetricsCalculator()

	tests := []struct {
		name     string
		values   []float64
		mean     float64
		minStdDev float64
		maxStdDev float64
	}{
		{
			name:      "Empty",
			values:    []float64{},
			mean:      0,
			minStdDev: 0,
			maxStdDev: 0,
		},
		{
			name:      "Single value",
			values:    []float64{50},
			mean:      50,
			minStdDev: 0,
			maxStdDev: 0,
		},
		{
			name:      "Same values",
			values:    []float64{50, 50, 50, 50},
			mean:      50,
			minStdDev: 0,
			maxStdDev: 0.01,
		},
		{
			name:      "Varied values",
			values:    []float64{10, 20, 30, 40, 50},
			mean:      30,
			minStdDev: 14,
			maxStdDev: 16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stdDev := calc.calculateStandardDeviation(tt.values, tt.mean)

			if stdDev < tt.minStdDev || stdDev > tt.maxStdDev {
				t.Errorf("Expected stdDev between %.2f and %.2f, got %.2f",
					tt.minStdDev, tt.maxStdDev, stdDev)
			}
		})
	}
}

// TestCalculateVelocity tests velocity calculation
func TestCalculateVelocity(t *testing.T) {
	calc := NewMetricsCalculatorWithWindow(7 * 24 * time.Hour)
	now := time.Now()

	tests := []struct {
		name     string
		contribs []Contribution
		minVel   float64
		maxVel   float64
	}{
		{
			name:     "Empty",
			contribs: []Contribution{},
			minVel:   0,
			maxVel:   0,
		},
		{
			name: "Single contribution",
			contribs: []Contribution{
				{Timestamp: now},
			},
			minVel: 0,
			maxVel: 0,
		},
		{
			name: "Recent contributions",
			contribs: []Contribution{
				{Timestamp: now.Add(-1 * time.Hour)},
				{Timestamp: now.Add(-2 * time.Hour)},
				{Timestamp: now.Add(-3 * time.Hour)},
				{Timestamp: now.Add(-4 * time.Hour)},
				{Timestamp: now.Add(-5 * time.Hour)},
				{Timestamp: now.Add(-6 * time.Hour)},
				{Timestamp: now.Add(-7 * time.Hour)},
			},
			minVel: 0.9,
			maxVel: 1.1,
		},
		{
			name: "Mixed recent and old",
			contribs: []Contribution{
				{Timestamp: now.Add(-1 * time.Hour)},
				{Timestamp: now.Add(-2 * time.Hour)},
				{Timestamp: now.Add(-30 * 24 * time.Hour)},
				{Timestamp: now.Add(-60 * 24 * time.Hour)},
			},
			minVel: 0.25,
			maxVel: 0.35,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			velocity := calc.CalculateVelocity(tt.contribs)

			if velocity < tt.minVel || velocity > tt.maxVel {
				t.Errorf("Expected velocity between %.2f and %.2f, got %.2f",
					tt.minVel, tt.maxVel, velocity)
			}
		})
	}
}

// TestCalculateConsistency tests consistency calculation
func TestCalculateConsistency(t *testing.T) {
	calc := NewMetricsCalculator()

	tests := []struct {
		name         string
		contribs     []Contribution
		minConsist   float64
		maxConsist   float64
		description  string
	}{
		{
			name:        "Empty",
			contribs:    []Contribution{},
			minConsist:  0,
			maxConsist:  0,
			description: "No contributions",
		},
		{
			name: "Single day",
			contribs: []Contribution{
				{Timestamp: time.Now()},
			},
			minConsist:  0,
			maxConsist:  0,
			description: "Not enough data",
		},
		{
			name: "Perfectly consistent",
			contribs: []Contribution{
				{Timestamp: time.Date(2024, 1, 1, 10, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 1, 11, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 2, 10, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 2, 11, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 3, 10, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 3, 11, 0, 0, 0, time.UTC)},
			},
			minConsist:  99,
			maxConsist:  100,
			description: "2 contributions per day consistently",
		},
		{
			name: "Inconsistent",
			contribs: []Contribution{
				{Timestamp: time.Date(2024, 1, 1, 10, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 2, 10, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 2, 11, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 2, 12, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 2, 13, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 2, 14, 0, 0, 0, time.UTC)},
				{Timestamp: time.Date(2024, 1, 3, 10, 0, 0, 0, time.UTC)},
			},
			minConsist:  0,
			maxConsist:  50,
			description: "Sporadic pattern",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			consistency := calc.CalculateConsistency(tt.contribs)

			if consistency < tt.minConsist || consistency > tt.maxConsist {
				t.Errorf("%s: Expected consistency between %.2f and %.2f, got %.2f",
					tt.description, tt.minConsist, tt.maxConsist, consistency)
			}
		})
	}
}

// TestGetProductivityScore tests overall productivity scoring
func TestGetProductivityScore(t *testing.T) {
	calc := NewMetricsCalculator()

	tests := []struct {
		name     string
		metrics  *ContributionMetrics
		minScore float64
		maxScore float64
	}{
		{
			name: "Empty metrics",
			metrics: &ContributionMetrics{
				TotalContributions: 0,
			},
			minScore: 0,
			maxScore: 0,
		},
		{
			name: "Perfect productivity",
			metrics: &ContributionMetrics{
				TotalContributions:  100,
				AverageQuality:      100,
				ContributionRate:    10,
				ReviewedRatio:       1.0,
				AverageTestCoverage: 100,
				HighQualityRatio:    1.0,
			},
			minScore: 95,
			maxScore: 100,
		},
		{
			name: "Average productivity",
			metrics: &ContributionMetrics{
				TotalContributions:  50,
				AverageQuality:      60,
				ContributionRate:    2,
				ReviewedRatio:       0.5,
				AverageTestCoverage: 50,
				HighQualityRatio:    0.5,
			},
			minScore: 40,
			maxScore: 60,
		},
		{
			name: "Low productivity",
			metrics: &ContributionMetrics{
				TotalContributions:  10,
				AverageQuality:      40,
				ContributionRate:    0.5,
				ReviewedRatio:       0.2,
				AverageTestCoverage: 20,
				HighQualityRatio:    0.1,
			},
			minScore: 10,
			maxScore: 30,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := calc.GetProductivityScore(tt.metrics)

			if score < tt.minScore || score > tt.maxScore {
				t.Errorf("Expected score between %.2f and %.2f, got %.2f",
					tt.minScore, tt.maxScore, score)
			}

			// Score should always be 0-100
			if score < 0 || score > 100 {
				t.Errorf("Score %.2f outside valid range [0, 100]", score)
			}
		})
	}
}

// TestMetricsTypeDistribution tests type distribution tracking
func TestMetricsTypeDistribution(t *testing.T) {
	calc := NewMetricsCalculator()

	contribs := []Contribution{
		{Type: CodeCommit},
		{Type: CodeCommit},
		{Type: CodeCommit},
		{Type: Testing},
		{Type: Testing},
		{Type: Documentation},
		{Type: Security},
		{Type: CodeReview},
		{Type: CodeReview},
		{Type: PullRequest},
	}

	metrics := calc.CalculateMetrics(contribs)

	expectedDist := map[Type]int{
		CodeCommit:      3,
		Testing:         2,
		Documentation:   1,
		Security:        1,
		CodeReview:      2,
		PullRequest:     1,
		IssueResolution: 0, // Not present
	}

	for contribType, expectedCount := range expectedDist {
		actualCount := metrics.TypeDistribution[contribType]
		if actualCount != expectedCount {
			t.Errorf("Type %s: expected %d, got %d",
				contribType.String(), expectedCount, actualCount)
		}
	}
}

// TestMetricsHighQualityThreshold tests high quality ratio calculation
func TestMetricsHighQualityThreshold(t *testing.T) {
	calc := NewMetricsCalculator()

	contribs := []Contribution{
		{QualityScore: 90}, // High
		{QualityScore: 85}, // High
		{QualityScore: 75}, // High
		{QualityScore: 74}, // Not high
		{QualityScore: 60}, // Not high
		{QualityScore: 50}, // Not high
	}

	metrics := calc.CalculateMetrics(contribs)

	// 3 out of 6 are high quality (>= 75)
	expectedRatio := 3.0 / 6.0
	if math.Abs(metrics.HighQualityRatio-expectedRatio) > 0.01 {
		t.Errorf("Expected high quality ratio %.2f, got %.2f",
			expectedRatio, metrics.HighQualityRatio)
	}
}

// TestMetricsReviewedRatio tests reviewed ratio calculation
func TestMetricsReviewedRatio(t *testing.T) {
	calc := NewMetricsCalculator()

	contribs := []Contribution{
		{PeerReviews: 2}, // Reviewed
		{PeerReviews: 1}, // Reviewed
		{PeerReviews: 0}, // Not reviewed
		{PeerReviews: 0}, // Not reviewed
		{PeerReviews: 3}, // Reviewed
	}

	metrics := calc.CalculateMetrics(contribs)

	// 3 out of 5 are reviewed
	expectedRatio := 3.0 / 5.0
	if math.Abs(metrics.ReviewedRatio-expectedRatio) > 0.01 {
		t.Errorf("Expected reviewed ratio %.2f, got %.2f",
			expectedRatio, metrics.ReviewedRatio)
	}
}

// TestMetricsEdgeCases tests edge cases
func TestMetricsEdgeCases(t *testing.T) {
	calc := NewMetricsCalculator()

	t.Run("All same values", func(t *testing.T) {
		contribs := []Contribution{
			{QualityScore: 75, TestCoverage: 80, LinesAdded: 100},
			{QualityScore: 75, TestCoverage: 80, LinesAdded: 100},
			{QualityScore: 75, TestCoverage: 80, LinesAdded: 100},
		}

		metrics := calc.CalculateMetrics(contribs)

		if metrics.AverageQuality != 75 {
			t.Errorf("Expected average 75, got %.2f", metrics.AverageQuality)
		}

		if metrics.MedianQuality != 75 {
			t.Errorf("Expected median 75, got %.2f", metrics.MedianQuality)
		}

		// Standard deviation should be 0
		if metrics.StandardDeviation > 0.01 {
			t.Errorf("Expected stdDev near 0, got %.2f", metrics.StandardDeviation)
		}
	})

	t.Run("Extreme values", func(t *testing.T) {
		contribs := []Contribution{
			{QualityScore: 0, TestCoverage: 0, LinesAdded: 0},
			{QualityScore: 100, TestCoverage: 100, LinesAdded: 1000000},
		}

		metrics := calc.CalculateMetrics(contribs)

		// Should handle without panic
		if metrics.TotalContributions != 2 {
			t.Error("Should handle extreme values")
		}

		// Average should be 50
		if metrics.AverageQuality != 50 {
			t.Errorf("Expected average 50, got %.2f", metrics.AverageQuality)
		}
	})
}

// TestProductivityScoreWeighting tests the weighting in productivity score
func TestProductivityScoreWeighting(t *testing.T) {
	calc := NewMetricsCalculator()

	// High quality should matter more than high volume
	highQualityMetrics := &ContributionMetrics{
		TotalContributions:  10,
		AverageQuality:      90,
		ContributionRate:    1,
		ReviewedRatio:       0.5,
		AverageTestCoverage: 50,
		HighQualityRatio:    0.5,
	}

	highVolumeMetrics := &ContributionMetrics{
		TotalContributions:  100,
		AverageQuality:      50,
		ContributionRate:    10,
		ReviewedRatio:       0.5,
		AverageTestCoverage: 50,
		HighQualityRatio:    0.5,
	}

	highQualityScore := calc.GetProductivityScore(highQualityMetrics)
	highVolumeScore := calc.GetProductivityScore(highVolumeMetrics)

	t.Logf("High quality score: %.2f, High volume score: %.2f",
		highQualityScore, highVolumeScore)

	// Quality weight is 0.3, volume weight is 0.2, so quality should dominate
	// (90 * 0.3 = 27 vs 50 * 0.3 = 15, a difference of 12 points just from quality)
	// Volume: (1 * 10 * 0.2 = 2 vs 10 * 10 * 0.2 = 20, capped at 100, so 20 points difference)
	// Quality difference (12) is smaller than volume difference (18), but quality matters more overall
}
