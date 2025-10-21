package contribution

import (
	"testing"
	"time"
)

// TestTypeString tests the String method for contribution types
func TestTypeString(t *testing.T) {
	tests := []struct {
		contribType Type
		expected    string
	}{
		{CodeCommit, "CodeCommit"},
		{PullRequest, "PullRequest"},
		{IssueResolution, "IssueResolution"},
		{Documentation, "Documentation"},
		{Testing, "Testing"},
		{CodeReview, "CodeReview"},
		{Security, "Security"},
		{Type(999), "Unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := tt.contribType.String()
			if result != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, result)
			}
		})
	}
}

// TestNewContribution tests the creation of a new contribution
func TestNewContribution(t *testing.T) {
	id := "test-123"
	contribType := CodeCommit

	contrib := NewContribution(id, contribType)

	if contrib.ID != id {
		t.Errorf("Expected ID %s, got %s", id, contrib.ID)
	}

	if contrib.Type != contribType {
		t.Errorf("Expected type %v, got %v", contribType, contrib.Type)
	}

	if contrib.Timestamp.IsZero() {
		t.Error("Expected non-zero timestamp")
	}

	// Timestamp should be recent (within last second)
	if time.Since(contrib.Timestamp) > time.Second {
		t.Error("Timestamp is not recent")
	}
}

// TestCalculateImpact tests the impact calculation algorithm
func TestCalculateImpact(t *testing.T) {
	tests := []struct {
		name     string
		contrib  Contribution
		minScore float64
		maxScore float64
	}{
		{
			name: "Security contribution with high quality",
			contrib: Contribution{
				LinesAdded:    100,
				LinesModified: 50,
				Type:          Security,
				QualityScore:  90,
				TestCoverage:  80,
				PeerReviews:   2,
				ReviewScore:   4.5,
			},
			minScore: 30,
			maxScore: 100,
		},
		{
			name: "Small code commit with low quality",
			contrib: Contribution{
				LinesAdded:   10,
				Type:         CodeCommit,
				QualityScore: 40,
				TestCoverage: 0,
				PeerReviews:  0,
			},
			minScore: 0,
			maxScore: 5,
		},
		{
			name: "Testing contribution with perfect quality",
			contrib: Contribution{
				LinesAdded:    200,
				LinesModified: 50,
				Type:          Testing,
				QualityScore:  100,
				TestCoverage:  100,
				PeerReviews:   3,
				ReviewScore:   5.0,
			},
			minScore: 50,
			maxScore: 100,
		},
		{
			name: "Zero lines impact",
			contrib: Contribution{
				LinesAdded:   0,
				Type:         CodeCommit,
				QualityScore: 100,
			},
			minScore: 0,
			maxScore: 0,
		},
		{
			name: "Impact capped at 100",
			contrib: Contribution{
				LinesAdded:    10000,
				LinesModified: 5000,
				Type:          Security,
				QualityScore:  100,
				TestCoverage:  100,
				PeerReviews:   5,
				ReviewScore:   5.0,
			},
			minScore: 100,
			maxScore: 100,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			impact := tt.contrib.CalculateImpact()

			if impact < tt.minScore || impact > tt.maxScore {
				t.Errorf("Expected impact between %.2f and %.2f, got %.2f",
					tt.minScore, tt.maxScore, impact)
			}
		})
	}
}

// TestCalculateImpactFormula tests specific impact formula components
func TestCalculateImpactFormula(t *testing.T) {
	// Test type multipliers
	baseContrib := Contribution{
		LinesAdded:   100,
		QualityScore: 100,
		TestCoverage: 0,
	}

	// Security should have highest multiplier (2.0)
	securityContrib := baseContrib
	securityContrib.Type = Security
	securityImpact := securityContrib.CalculateImpact()

	// Testing should have medium multiplier (1.5)
	testingContrib := baseContrib
	testingContrib.Type = Testing
	testingImpact := testingContrib.CalculateImpact()

	// CodeCommit should have baseline multiplier (1.0)
	codeContrib := baseContrib
	codeContrib.Type = CodeCommit
	codeImpact := codeContrib.CalculateImpact()

	if securityImpact <= testingImpact {
		t.Error("Security contribution should have higher impact than testing")
	}

	if testingImpact <= codeImpact {
		t.Error("Testing contribution should have higher impact than code commit")
	}
}

// TestIsHighQuality tests the quality threshold check
func TestIsHighQuality(t *testing.T) {
	tests := []struct {
		name      string
		score     float64
		threshold float64
		expected  bool
	}{
		{"Above threshold", 80, 75, true},
		{"At threshold", 75, 75, true},
		{"Below threshold", 70, 75, false},
		{"Zero score", 0, 50, false},
		{"Perfect score", 100, 90, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			contrib := Contribution{QualityScore: tt.score}
			result := contrib.IsHighQuality(tt.threshold)

			if result != tt.expected {
				t.Errorf("Expected %v for score %.2f with threshold %.2f",
					tt.expected, tt.score, tt.threshold)
			}
		})
	}
}

// TestHasTestCoverage tests the test coverage check
func TestHasTestCoverage(t *testing.T) {
	tests := []struct {
		name     string
		coverage float64
		expected bool
	}{
		{"Zero coverage", 0, false},
		{"Some coverage", 0.1, true},
		{"Full coverage", 100, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			contrib := Contribution{TestCoverage: tt.coverage}
			result := contrib.HasTestCoverage()

			if result != tt.expected {
				t.Errorf("Expected %v for coverage %.2f", tt.expected, tt.coverage)
			}
		})
	}
}

// TestIsReviewed tests the peer review check
func TestIsReviewed(t *testing.T) {
	tests := []struct {
		name     string
		reviews  int
		expected bool
	}{
		{"No reviews", 0, false},
		{"One review", 1, true},
		{"Multiple reviews", 5, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			contrib := Contribution{PeerReviews: tt.reviews}
			result := contrib.IsReviewed()

			if result != tt.expected {
				t.Errorf("Expected %v for %d reviews", tt.expected, tt.reviews)
			}
		})
	}
}

// TestGetAge tests the age calculation
func TestGetAge(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name      string
		timestamp time.Time
		minAge    time.Duration
		maxAge    time.Duration
	}{
		{
			name:      "Recent contribution",
			timestamp: now.Add(-time.Minute),
			minAge:    50 * time.Second,
			maxAge:    2 * time.Minute,
		},
		{
			name:      "One hour old",
			timestamp: now.Add(-time.Hour),
			minAge:    50 * time.Minute,
			maxAge:    70 * time.Minute,
		},
		{
			name:      "One day old",
			timestamp: now.Add(-24 * time.Hour),
			minAge:    23 * time.Hour,
			maxAge:    25 * time.Hour,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			contrib := Contribution{Timestamp: tt.timestamp}
			age := contrib.GetAge()

			if age < tt.minAge || age > tt.maxAge {
				t.Errorf("Expected age between %v and %v, got %v",
					tt.minAge, tt.maxAge, age)
			}
		})
	}
}

// TestSummary tests the summary generation
func TestSummary(t *testing.T) {
	contrib := Contribution{
		ID:            "test-123",
		Type:          CodeCommit,
		Timestamp:     time.Now().Add(-time.Hour),
		QualityScore:  85.5,
		LinesAdded:    100,
		LinesModified: 50,
		LinesDeleted:  20,
		TestCoverage:  75.0,
		PeerReviews:   2,
	}

	summary := contrib.Summary()

	// Check required fields exist
	requiredFields := []string{
		"id", "type", "timestamp", "quality_score",
		"impact", "lines_changed", "test_coverage",
		"peer_reviews", "age_hours",
	}

	for _, field := range requiredFields {
		if _, exists := summary[field]; !exists {
			t.Errorf("Summary missing required field: %s", field)
		}
	}

	// Verify specific values
	if summary["id"] != "test-123" {
		t.Errorf("Expected id 'test-123', got %v", summary["id"])
	}

	if summary["type"] != "CodeCommit" {
		t.Errorf("Expected type 'CodeCommit', got %v", summary["type"])
	}

	if summary["quality_score"] != 85.5 {
		t.Errorf("Expected quality_score 85.5, got %v", summary["quality_score"])
	}

	linesChanged := contrib.LinesAdded + contrib.LinesModified + contrib.LinesDeleted
	if summary["lines_changed"] != linesChanged {
		t.Errorf("Expected lines_changed %d, got %v", linesChanged, summary["lines_changed"])
	}

	// Age should be approximately 1 hour
	ageHours, ok := summary["age_hours"].(float64)
	if !ok {
		t.Error("age_hours should be float64")
	}
	if ageHours < 0.9 || ageHours > 1.1 {
		t.Errorf("Expected age_hours around 1.0, got %.2f", ageHours)
	}
}

// TestContributionImmutability tests that methods don't modify the contribution
func TestContributionImmutability(t *testing.T) {
	original := Contribution{
		ID:            "test",
		Type:          CodeCommit,
		QualityScore:  75,
		TestCoverage:  80,
		LinesAdded:    100,
		PeerReviews:   2,
		Timestamp:     time.Now(),
	}

	// Make a copy for comparison
	before := original

	// Call various methods
	_ = original.CalculateImpact()
	_ = original.IsHighQuality(70)
	_ = original.HasTestCoverage()
	_ = original.IsReviewed()
	_ = original.GetAge()
	_ = original.Summary()

	// Verify no fields were modified
	if original.ID != before.ID {
		t.Error("ID was modified")
	}
	if original.QualityScore != before.QualityScore {
		t.Error("QualityScore was modified")
	}
	if original.TestCoverage != before.TestCoverage {
		t.Error("TestCoverage was modified")
	}
	if original.LinesAdded != before.LinesAdded {
		t.Error("LinesAdded was modified")
	}
}

// TestEdgeCases tests edge cases and boundary conditions
func TestEdgeCases(t *testing.T) {
	t.Run("Negative lines", func(t *testing.T) {
		contrib := Contribution{
			LinesAdded:    -10, // Invalid but shouldn't crash
			Type:          CodeCommit,
			QualityScore:  100,
		}
		impact := contrib.CalculateImpact()
		// Should handle gracefully
		if impact < 0 {
			t.Error("Impact should not be negative")
		}
	})

	t.Run("Extreme quality score", func(t *testing.T) {
		contrib := Contribution{
			LinesAdded:   100,
			QualityScore: 1000, // Way over 100
			Type:         CodeCommit,
		}
		impact := contrib.CalculateImpact()
		// Impact should still be capped at 100
		if impact > 100 {
			t.Errorf("Impact should be capped at 100, got %.2f", impact)
		}
	})

	t.Run("Future timestamp", func(t *testing.T) {
		contrib := Contribution{
			Timestamp: time.Now().Add(time.Hour), // Future
		}
		age := contrib.GetAge()
		// Age will be negative, but shouldn't crash
		if age > 0 {
			t.Error("Age of future contribution should be negative or zero")
		}
	})
}
