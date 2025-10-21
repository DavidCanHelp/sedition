package contribution

import (
	"testing"

	"github.com/davidcanhelp/sedition/config"
)

// TestNewQualityAnalyzer tests analyzer creation
func TestNewQualityAnalyzer(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	if analyzer == nil {
		t.Fatal("Expected non-nil analyzer")
	}

	if analyzer.config == nil {
		t.Error("Expected non-nil config")
	}
}

// TestNewQualityAnalyzerWithConfig tests custom config analyzer
func TestNewQualityAnalyzerWithConfig(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	cfg.QualityThreshold = 90.0

	analyzer := NewQualityAnalyzerWithConfig(cfg)

	if analyzer == nil {
		t.Fatal("Expected non-nil analyzer")
	}

	if analyzer.config.QualityThreshold != 90.0 {
		t.Errorf("Expected threshold 90.0, got %.2f", analyzer.config.QualityThreshold)
	}
}

// TestAnalyzeContribution tests the core quality analysis algorithm
func TestAnalyzeContribution(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	tests := []struct {
		name     string
		contrib  Contribution
		minScore float64
		maxScore float64
	}{
		{
			name: "Perfect contribution",
			contrib: Contribution{
				TestCoverage:  100,
				Documentation: 100,
				Complexity:    0,
				Type:          Security,
				PeerReviews:   3,
				ReviewScore:   5.0,
				LinesAdded:    50,
			},
			minScore: 95,
			maxScore: 100,
		},
		{
			name: "High quality contribution",
			contrib: Contribution{
				TestCoverage:  90,
				Documentation: 85,
				Complexity:    5,
				Type:          Testing,
				PeerReviews:   2,
				ReviewScore:   4.5,
				LinesAdded:    100,
			},
			minScore: 80,
			maxScore: 97,
		},
		{
			name: "Average contribution",
			contrib: Contribution{
				TestCoverage:  60,
				Documentation: 50,
				Complexity:    10,
				Type:          CodeCommit,
				PeerReviews:   1,
				ReviewScore:   3.0,
				LinesAdded:    150,
			},
			minScore: 50,
			maxScore: 70,
		},
		{
			name: "Low quality contribution",
			contrib: Contribution{
				TestCoverage:  20,
				Documentation: 10,
				Complexity:    30,
				Type:          CodeCommit,
				PeerReviews:   0,
				LinesAdded:    200,
			},
			minScore: 20,
			maxScore: 40,
		},
		{
			name: "Zero quality contribution",
			contrib: Contribution{
				TestCoverage:  0,
				Documentation: 0,
				Complexity:    50,
				Type:          CodeCommit,
				PeerReviews:   0,
				LinesAdded:    1000,
			},
			minScore: 0,
			maxScore: 20,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, err := analyzer.AnalyzeContribution(tt.contrib)

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

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

// TestAnalyzeContributionComponents tests individual score components
func TestAnalyzeContributionComponents(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	t.Run("Test coverage component (0-25 points)", func(t *testing.T) {
		// Max test coverage should give 25 points
		highCoverage := Contribution{
			TestCoverage:  100,
			Documentation: 0,
			Complexity:    0,
			Type:          CodeCommit,
			LinesAdded:    50,
		}

		// Zero test coverage should give 0 points from this component
		noCoverage := Contribution{
			TestCoverage:  0,
			Documentation: 0,
			Complexity:    0,
			Type:          CodeCommit,
			LinesAdded:    50,
		}

		highScore, _ := analyzer.AnalyzeContribution(highCoverage)
		lowScore, _ := analyzer.AnalyzeContribution(noCoverage)

		// High coverage should score significantly better
		if highScore <= lowScore {
			t.Error("High coverage should score better than no coverage")
		}
	})

	t.Run("Documentation component (0-20 points)", func(t *testing.T) {
		highDocs := Contribution{
			TestCoverage:  0,
			Documentation: 100,
			Complexity:    0,
			Type:          CodeCommit,
			LinesAdded:    50,
		}

		noDocs := Contribution{
			TestCoverage:  0,
			Documentation: 0,
			Complexity:    0,
			Type:          CodeCommit,
			LinesAdded:    50,
		}

		highScore, _ := analyzer.AnalyzeContribution(highDocs)
		lowScore, _ := analyzer.AnalyzeContribution(noDocs)

		if highScore <= lowScore {
			t.Error("High documentation should score better")
		}
	})

	t.Run("Complexity component (0-25 points, inverse)", func(t *testing.T) {
		lowComplexity := Contribution{
			TestCoverage:  0,
			Documentation: 0,
			Complexity:    0, // Perfect
			Type:          CodeCommit,
			LinesAdded:    50,
		}

		highComplexity := Contribution{
			TestCoverage:  0,
			Documentation: 0,
			Complexity:    50, // Very high
			Type:          CodeCommit,
			LinesAdded:    50,
		}

		lowScore, _ := analyzer.AnalyzeContribution(lowComplexity)
		highScore, _ := analyzer.AnalyzeContribution(highComplexity)

		// Lower complexity should score better
		if lowScore <= highScore {
			t.Error("Lower complexity should score better than high complexity")
		}
	})
}

// TestGetTypeScore tests the type scoring system
func TestGetTypeScore(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	tests := []struct {
		contribType Type
		minScore    float64
		maxScore    float64
	}{
		{Security, 15.0, 15.0},         // Maximum bonus
		{Testing, 12.0, 12.0},          // High bonus
		{CodeReview, 10.0, 10.0},       // Medium-high bonus
		{CodeCommit, 8.0, 8.0},         // Medium bonus
		{PullRequest, 8.0, 8.0},        // Medium bonus
		{IssueResolution, 7.0, 7.0},    // Medium-low bonus
		{Documentation, 5.0, 5.0},      // Low bonus
		{Type(999), 3.0, 3.0},          // Unknown type
	}

	for _, tt := range tests {
		t.Run(tt.contribType.String(), func(t *testing.T) {
			score := analyzer.getTypeScore(tt.contribType)

			if score < tt.minScore || score > tt.maxScore {
				t.Errorf("Expected score between %.2f and %.2f, got %.2f",
					tt.minScore, tt.maxScore, score)
			}
		})
	}
}

// TestGetSizeModifier tests the size-based scoring modifier
func TestGetSizeModifier(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	tests := []struct {
		name     string
		lines    int
		expected float64
	}{
		{"Zero lines", 0, 0.5},
		{"Tiny (5 lines)", 5, 0.8},
		{"Small (25 lines)", 25, 1.0},
		{"Medium (100 lines)", 100, 1.1},
		{"Large (300 lines)", 300, 1.0},
		{"Very large (1000 lines)", 1000, 0.9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			contrib := Contribution{
				LinesAdded:    tt.lines / 2,
				LinesModified: tt.lines / 4,
				LinesDeleted:  tt.lines / 4,
			}

			modifier := analyzer.getSizeModifier(contrib)

			if modifier != tt.expected {
				t.Errorf("Expected modifier %.2f, got %.2f", tt.expected, modifier)
			}
		})
	}
}

// TestCalculateAverageQuality tests average quality calculation
func TestCalculateAverageQuality(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	tests := []struct {
		name     string
		contribs []Contribution
		expected float64
	}{
		{
			name:     "Empty list",
			contribs: []Contribution{},
			expected: 0.0,
		},
		{
			name: "Single contribution",
			contribs: []Contribution{
				{QualityScore: 75.0},
			},
			expected: 75.0,
		},
		{
			name: "Multiple contributions",
			contribs: []Contribution{
				{QualityScore: 80.0},
				{QualityScore: 90.0},
				{QualityScore: 70.0},
			},
			expected: 80.0,
		},
		{
			name: "All perfect scores",
			contribs: []Contribution{
				{QualityScore: 100.0},
				{QualityScore: 100.0},
				{QualityScore: 100.0},
			},
			expected: 100.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			avg := analyzer.CalculateAverageQuality(tt.contribs)

			if avg != tt.expected {
				t.Errorf("Expected average %.2f, got %.2f", tt.expected, avg)
			}
		})
	}
}

// TestQualityAnalyzerIsHighQuality tests the quality threshold check
func TestQualityAnalyzerIsHighQuality(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	cfg.QualityThreshold = 75.0
	analyzer := NewQualityAnalyzerWithConfig(cfg)

	tests := []struct {
		name     string
		score    float64
		expected bool
	}{
		{"Well above threshold", 90.0, true},
		{"At threshold", 75.0, true},
		{"Just below threshold", 74.9, false},
		{"Well below threshold", 50.0, false},
		{"Perfect score", 100.0, true},
		{"Zero score", 0.0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := analyzer.IsHighQuality(tt.score)

			if result != tt.expected {
				t.Errorf("Expected %v for score %.2f", tt.expected, tt.score)
			}
		})
	}
}

// TestGetQualityTier tests quality tier classification
func TestGetQualityTier(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	tests := []struct {
		score    float64
		expected string
	}{
		{95.0, "Exceptional"},
		{90.0, "Exceptional"},
		{85.0, "High"},
		{75.0, "High"},
		{70.0, "Good"},
		{60.0, "Good"},
		{50.0, "Average"},
		{40.0, "Average"},
		{30.0, "Below Average"},
		{20.0, "Below Average"},
		{10.0, "Poor"},
		{0.0, "Poor"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			tier := analyzer.GetQualityTier(tt.score)

			if tier != tt.expected {
				t.Errorf("Expected tier '%s' for score %.2f, got '%s'",
					tt.expected, tt.score, tier)
			}
		})
	}
}

// TestAnalyzeTrend tests quality trend analysis
func TestAnalyzeTrend(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	tests := []struct {
		name     string
		contribs []Contribution
		expected string
	}{
		{
			name:     "Insufficient data",
			contribs: []Contribution{{QualityScore: 50}},
			expected: "Insufficient data",
		},
		{
			name: "Improving trend",
			contribs: []Contribution{
				{QualityScore: 50},
				{QualityScore: 55},
				{QualityScore: 60},
				{QualityScore: 65},
				{QualityScore: 70},
				{QualityScore: 75},
			},
			expected: "Improving",
		},
		{
			name: "Slightly improving trend",
			contribs: []Contribution{
				{QualityScore: 60},
				{QualityScore: 62},
				{QualityScore: 64},
				{QualityScore: 66},
				{QualityScore: 68},
				{QualityScore: 69},
			},
			expected: "Slightly Improving",
		},
		{
			name: "Declining trend",
			contribs: []Contribution{
				{QualityScore: 80},
				{QualityScore: 75},
				{QualityScore: 70},
				{QualityScore: 65},
				{QualityScore: 60},
				{QualityScore: 50},
			},
			expected: "Declining",
		},
		{
			name: "Stable trend",
			contribs: []Contribution{
				{QualityScore: 70},
				{QualityScore: 70},
				{QualityScore: 70},
				{QualityScore: 70},
				{QualityScore: 70},
				{QualityScore: 70},
			},
			expected: "Stable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trend := analyzer.AnalyzeTrend(tt.contribs)

			if trend != tt.expected {
				t.Errorf("Expected trend '%s', got '%s'", tt.expected, trend)
			}
		})
	}
}

// TestQualityAnalysisReproducibility tests that the same input gives same output
func TestQualityAnalysisReproducibility(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	contrib := Contribution{
		TestCoverage:  85.0,
		Documentation: 70.0,
		Complexity:    8.0,
		Type:          CodeCommit,
		PeerReviews:   2,
		ReviewScore:   4.2,
		LinesAdded:    100,
	}

	// Run analysis multiple times
	scores := make([]float64, 10)
	for i := 0; i < 10; i++ {
		score, err := analyzer.AnalyzeContribution(contrib)
		if err != nil {
			t.Fatalf("Unexpected error on iteration %d: %v", i, err)
		}
		scores[i] = score
	}

	// All scores should be identical
	firstScore := scores[0]
	for i, score := range scores[1:] {
		if score != firstScore {
			t.Errorf("Score mismatch at iteration %d: expected %.2f, got %.2f",
				i+1, firstScore, score)
		}
	}
}

// TestQualityAnalysisEdgeCases tests edge cases
func TestQualityAnalysisEdgeCases(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	t.Run("All zeros", func(t *testing.T) {
		contrib := Contribution{}
		score, err := analyzer.AnalyzeContribution(contrib)

		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should handle gracefully
		if score < 0 || score > 100 {
			t.Errorf("Score %.2f outside valid range", score)
		}
	})

	t.Run("Negative values", func(t *testing.T) {
		contrib := Contribution{
			TestCoverage:  -10,
			Documentation: -5,
			Complexity:    -3,
			LinesAdded:    -100,
		}

		score, err := analyzer.AnalyzeContribution(contrib)

		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Should handle gracefully without panicking
		if score < 0 {
			t.Error("Score should not be negative")
		}
	})

	t.Run("Extreme values", func(t *testing.T) {
		contrib := Contribution{
			TestCoverage:  1000,
			Documentation: 1000,
			Complexity:    1000,
			PeerReviews:   100,
			ReviewScore:   100,
			LinesAdded:    1000000,
		}

		score, err := analyzer.AnalyzeContribution(contrib)

		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Score should be capped at 100
		if score > 100 {
			t.Errorf("Score %.2f should be capped at 100", score)
		}
	})
}

// TestReviewScoreImpact tests how peer reviews affect quality score
func TestReviewScoreImpact(t *testing.T) {
	analyzer := NewQualityAnalyzer()

	baseContrib := Contribution{
		TestCoverage:  80,
		Documentation: 70,
		Complexity:    5,
		Type:          CodeCommit,
		LinesAdded:    100,
	}

	// No reviews
	noReviewsContrib := baseContrib
	noReviewsContrib.PeerReviews = 0
	noReviewsScore, _ := analyzer.AnalyzeContribution(noReviewsContrib)

	// With positive reviews
	reviewedContrib := baseContrib
	reviewedContrib.PeerReviews = 3
	reviewedContrib.ReviewScore = 4.5
	reviewedScore, _ := analyzer.AnalyzeContribution(reviewedContrib)

	// With poor reviews
	poorReviewedContrib := baseContrib
	poorReviewedContrib.PeerReviews = 2
	poorReviewedContrib.ReviewScore = 2.0
	poorReviewedScore, _ := analyzer.AnalyzeContribution(poorReviewedContrib)

	// Reviewed should score higher than non-reviewed
	if reviewedScore <= noReviewsScore {
		t.Error("Reviewed contribution should score higher than non-reviewed")
	}

	// High review scores should beat low review scores
	if reviewedScore <= poorReviewedScore {
		t.Error("High review scores should beat low review scores")
	}

	// Even poor reviews might be better than no reviews
	// (depending on implementation details)
	t.Logf("No reviews: %.2f, Poor reviews: %.2f, Good reviews: %.2f",
		noReviewsScore, poorReviewedScore, reviewedScore)
}
