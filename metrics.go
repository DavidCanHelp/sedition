// Package poc implements comprehensive contribution metrics calculation
// This module provides detailed analysis of contributor performance across
// multiple dimensions including productivity, quality, collaboration, and impact.
//
// Mathematical Foundation:
// The metrics system uses weighted scoring across multiple categories:
// - Productivity Metrics (25%): Volume and frequency of contributions
// - Quality Metrics (30%): Code quality, testing, documentation standards
// - Collaboration Metrics (20%): Peer review participation, mentoring
// - Impact Metrics (15%): Bug fixes, performance improvements, feature adoption
// - Innovation Metrics (10%): Novel approaches, architectural contributions
//
// Each metric is normalized to a 0-100 scale and combined using weighted averages
package poc

import (
	"fmt"
	"math"
	"time"
)

// MetricsCalculator computes comprehensive contribution metrics
type MetricsCalculator struct {
	// Metric weights (must sum to 1.0)
	productivityWeight   float64
	qualityWeight       float64
	collaborationWeight float64
	impactWeight        float64
	innovationWeight    float64
	
	// Time windows for analysis
	shortTermWindow  time.Duration // Recent performance (7 days)
	mediumTermWindow time.Duration // Medium-term trends (30 days)
	longTermWindow   time.Duration // Long-term patterns (180 days)
	
	// Baseline values for normalization
	avgContributionsPerWeek float64
	avgCodeLinesPerContrib  float64
	avgReviewsPerWeek      float64
}

// ContributorMetrics holds comprehensive metrics for a contributor
type ContributorMetrics struct {
	Address           string
	OverallScore      float64
	LastUpdated       time.Time
	
	// Primary metric categories
	ProductivityScore   float64
	QualityScore       float64
	CollaborationScore float64
	ImpactScore        float64
	InnovationScore    float64
	
	// Detailed productivity metrics
	Productivity ProductivityMetrics
	
	// Detailed quality metrics  
	Quality QualityMetrics
	
	// Detailed collaboration metrics
	Collaboration CollaborationMetrics
	
	// Detailed impact metrics
	Impact ImpactMetrics
	
	// Detailed innovation metrics
	Innovation InnovationMetrics
	
	// Trend analysis
	Trends TrendAnalysis
}

// ProductivityMetrics tracks contribution volume and frequency
type ProductivityMetrics struct {
	ContributionsLastWeek    int
	ContributionsLastMonth   int
	ContributionsLastQuarter int
	AvgContribSize          float64 // Average lines of code per contribution
	ContributionFrequency   float64 // Contributions per week
	ConsistencyScore        float64 // Regularity of contributions (0-100)
	VelocityScore          float64 // Rate of contribution increase/decrease
	BurnoutRiskScore       float64 // Risk of contributor burnout (0-100)
}

// CollaborationMetrics tracks peer interaction and teamwork
type CollaborationMetrics struct {
	PeerReviewsGiven       int
	PeerReviewsReceived    int
	AvgReviewTurnaround    time.Duration
	ReviewQualityScore     float64 // Quality of reviews given
	MentorshipActivities   int     // Times helped other contributors
	CommunityEngagement    float64 // Participation in discussions, etc.
	ConflictResolution     float64 // Ability to resolve disagreements
	KnowledgeSharing      float64 // Documentation, tutorials, etc.
}

// ImpactMetrics measures the real-world effect of contributions
type ImpactMetrics struct {
	BugFixCount           int
	CriticalBugFixCount   int
	PerformanceImprovements int
	FeatureAdoption       float64 // How widely used are contributed features
	UserSatisfaction      float64 // Feedback from feature users
	SecurityImprovements  int     // Security enhancements contributed
	TechnicalDebtReduction float64 // Amount of technical debt addressed
	SystemStabilityImpact float64 // Impact on overall system stability
}

// InnovationMetrics tracks creative and architectural contributions
type InnovationMetrics struct {
	ArchitecturalContributions int
	AlgorithmImprovements     int
	NovelApproaches          int
	PatternIntroductions     int     // New design patterns introduced
	ToolsCreated            int     // Development tools contributed
	ProcessImprovements     int     // Development process enhancements
	ResearchContributions   int     // Research papers, analysis, etc.
	CreativityScore        float64 // Overall creativity assessment (0-100)
}

// TrendAnalysis provides trend analysis across time periods
type TrendAnalysis struct {
	ShortTermTrend   TrendDirection // 7-day trend
	MediumTermTrend  TrendDirection // 30-day trend
	LongTermTrend    TrendDirection // 180-day trend
	SeasonalPatterns []SeasonalPattern
	PeakPerformance  PeakPeriod
	LowPerformance   PeakPeriod
}

// TrendDirection indicates performance trend direction
type TrendDirection int

const (
	TrendUnknown TrendDirection = iota
	TrendImproving
	TrendStable
	TrendDeclining
)

// SeasonalPattern identifies recurring performance patterns
type SeasonalPattern struct {
	Pattern     string    // Description of the pattern
	Confidence  float64   // Confidence in pattern detection (0-1)
	NextPeak    time.Time // Predicted next peak
	NextTrough  time.Time // Predicted next trough
}

// PeakPeriod identifies periods of exceptional performance
type PeakPeriod struct {
	StartTime    time.Time
	EndTime      time.Time
	Score        float64
	Description  string
}

// NewMetricsCalculator creates a new metrics calculator with default weights
func NewMetricsCalculator() *MetricsCalculator {
	return &MetricsCalculator{
		// Default metric weights
		productivityWeight:   0.25,
		qualityWeight:       0.30,
		collaborationWeight: 0.20,
		impactWeight:        0.15,
		innovationWeight:    0.10,
		
		// Time windows
		shortTermWindow:  7 * 24 * time.Hour,   // 1 week
		mediumTermWindow: 30 * 24 * time.Hour,  // 1 month
		longTermWindow:   180 * 24 * time.Hour, // 6 months
		
		// Network baselines (would be calculated from historical data)
		avgContributionsPerWeek: 3.0,
		avgCodeLinesPerContrib:  150.0,
		avgReviewsPerWeek:      2.0,
	}
}

// CalculateMetrics computes comprehensive metrics for a contributor
func (mc *MetricsCalculator) CalculateMetrics(validator *Validator, history []Contribution, reviews []PeerReviewEvent) *ContributorMetrics {
	metrics := &ContributorMetrics{
		Address:     validator.Address,
		LastUpdated: time.Now(),
	}
	
	// Calculate each metric category
	metrics.Productivity = mc.calculateProductivityMetrics(history)
	metrics.ProductivityScore = mc.normalizeProductivityScore(metrics.Productivity)
	
	metrics.Quality = mc.extractQualityMetrics(history)
	metrics.QualityScore = mc.normalizeQualityScore(metrics.Quality)
	
	metrics.Collaboration = mc.calculateCollaborationMetrics(reviews, history)
	metrics.CollaborationScore = mc.normalizeCollaborationScore(metrics.Collaboration)
	
	metrics.Impact = mc.calculateImpactMetrics(history)
	metrics.ImpactScore = mc.normalizeImpactScore(metrics.Impact)
	
	metrics.Innovation = mc.calculateInnovationMetrics(history)
	metrics.InnovationScore = mc.normalizeInnovationScore(metrics.Innovation)
	
	// Calculate trend analysis
	metrics.Trends = mc.calculateTrendAnalysis(history)
	
	// Calculate overall weighted score
	metrics.OverallScore = 
		metrics.ProductivityScore * mc.productivityWeight +
		metrics.QualityScore * mc.qualityWeight +
		metrics.CollaborationScore * mc.collaborationWeight +
		metrics.ImpactScore * mc.impactWeight +
		metrics.InnovationScore * mc.innovationWeight
	
	return metrics
}

// calculateProductivityMetrics calculates productivity-related metrics
func (mc *MetricsCalculator) calculateProductivityMetrics(history []Contribution) ProductivityMetrics {
	now := time.Now()
	weekAgo := now.Add(-mc.shortTermWindow)
	monthAgo := now.Add(-mc.mediumTermWindow)
	quarterAgo := now.Add(-mc.longTermWindow)
	
	metrics := ProductivityMetrics{}
	
	// Count contributions by time period
	totalLines := 0
	contributionTimes := make([]time.Time, 0)
	
	for _, contrib := range history {
		contributionTimes = append(contributionTimes, contrib.Timestamp)
		linesChanged := contrib.LinesAdded + contrib.LinesModified
		totalLines += linesChanged
		
		if contrib.Timestamp.After(weekAgo) {
			metrics.ContributionsLastWeek++
		}
		if contrib.Timestamp.After(monthAgo) {
			metrics.ContributionsLastMonth++
		}
		if contrib.Timestamp.After(quarterAgo) {
			metrics.ContributionsLastQuarter++
		}
	}
	
	// Calculate averages
	if len(history) > 0 {
		metrics.AvgContribSize = float64(totalLines) / float64(len(history))
	}
	
	// Calculate frequency (contributions per week over last month)
	if metrics.ContributionsLastMonth > 0 {
		weeksInMonth := 4.33 // Average weeks per month
		metrics.ContributionFrequency = float64(metrics.ContributionsLastMonth) / weeksInMonth
	}
	
	// Calculate consistency score based on regularity
	metrics.ConsistencyScore = mc.calculateConsistencyScore(contributionTimes)
	
	// Calculate velocity (trend in contribution rate)
	metrics.VelocityScore = mc.calculateVelocityScore(contributionTimes)
	
	// Calculate burnout risk
	metrics.BurnoutRiskScore = mc.calculateBurnoutRisk(metrics)
	
	return metrics
}

// calculateConsistencyScore measures regularity of contributions
func (mc *MetricsCalculator) calculateConsistencyScore(times []time.Time) float64 {
	if len(times) < 3 {
		return 50.0 // Neutral score for insufficient data
	}
	
	// Calculate intervals between contributions
	intervals := make([]time.Duration, 0)
	for i := 1; i < len(times); i++ {
		interval := times[i].Sub(times[i-1])
		intervals = append(intervals, interval)
	}
	
	// Calculate coefficient of variation
	mean := time.Duration(0)
	for _, interval := range intervals {
		mean += interval
	}
	mean = mean / time.Duration(len(intervals))
	
	variance := time.Duration(0)
	for _, interval := range intervals {
		diff := interval - mean
		variance += time.Duration(diff.Nanoseconds() * diff.Nanoseconds())
	}
	variance = variance / time.Duration(len(intervals))
	
	stdDev := time.Duration(math.Sqrt(float64(variance.Nanoseconds())))
	
	// Coefficient of variation (lower is more consistent)
	if mean == 0 {
		return 50.0
	}
	
	cv := float64(stdDev.Nanoseconds()) / float64(mean.Nanoseconds())
	
	// Convert to 0-100 score (lower CV = higher consistency)
	consistency := math.Max(0, 100.0 - cv*100.0)
	return math.Min(100.0, consistency)
}

// calculateVelocityScore measures acceleration/deceleration in contributions
func (mc *MetricsCalculator) calculateVelocityScore(times []time.Time) float64 {
	if len(times) < 6 {
		return 50.0 // Neutral for insufficient data
	}
	
	now := time.Now()
	recentPeriod := now.Add(-mc.shortTermWindow)
	olderPeriod := now.Add(-mc.mediumTermWindow)
	
	recentCount := 0
	olderCount := 0
	
	for _, t := range times {
		if t.After(recentPeriod) {
			recentCount++
		} else if t.After(olderPeriod) {
			olderCount++
		}
	}
	
	// Calculate velocity as ratio of recent to older contributions
	if olderCount == 0 {
		return 75.0 // Assume positive trend if no older data
	}
	
	velocity := float64(recentCount) / float64(olderCount)
	
	// Convert to 0-100 score (1.0 = 50, >1.0 = improvement, <1.0 = decline)
	score := 50.0 + (velocity-1.0)*25.0
	return math.Max(0, math.Min(100.0, score))
}

// calculateBurnoutRisk assesses contributor burnout risk
func (mc *MetricsCalculator) calculateBurnoutRisk(productivity ProductivityMetrics) float64 {
	risk := 0.0
	
	// High contribution frequency increases burnout risk
	if productivity.ContributionFrequency > mc.avgContributionsPerWeek*2 {
		risk += 30.0
	}
	
	// Low consistency suggests irregular work patterns (potential burnout)
	if productivity.ConsistencyScore < 30.0 {
		risk += 20.0
	}
	
	// Declining velocity suggests fatigue
	if productivity.VelocityScore < 30.0 {
		risk += 25.0
	}
	
	// Very large average contribution size might indicate overwork
	if productivity.AvgContribSize > mc.avgCodeLinesPerContrib*3 {
		risk += 15.0
	}
	
	// High recent activity compared to historical
	if productivity.ContributionsLastWeek > int(mc.avgContributionsPerWeek*2) {
		risk += 10.0
	}
	
	return math.Min(100.0, risk)
}

// extractQualityMetrics extracts quality metrics from contribution history
func (mc *MetricsCalculator) extractQualityMetrics(history []Contribution) QualityMetrics {
	if len(history) == 0 {
		return QualityMetrics{}
	}
	
	totalQuality := 0.0
	totalComplexity := 0.0
	totalCoverage := 0.0
	totalDocumentation := 0.0
	vulnerabilities := 0
	styleViolations := 0
	
	for _, contrib := range history {
		totalQuality += contrib.QualityScore
		totalComplexity += contrib.Complexity
		totalCoverage += contrib.TestCoverage
		totalDocumentation += contrib.Documentation
		
		// In a real implementation, these would be calculated from actual analysis
		if contrib.QualityScore < 60 {
			vulnerabilities += 2 // Assume poor quality indicates potential issues
		}
		if contrib.QualityScore < 70 {
			styleViolations += 3 // Assume style issues
		}
	}
	
	count := float64(len(history))
	
	return QualityMetrics{
		OverallScore:       totalQuality / count,
		ComplexityScore:    100.0 - (totalComplexity/count)*5, // Lower complexity = higher score
		TestCoverageScore:  (totalCoverage / count) * 1.25,    // Bonus for high coverage
		DocumentationScore: (totalDocumentation / count) * 1.1, // Slight bonus
		SecurityScore:      math.Max(0, 90.0-float64(vulnerabilities)*5),
		CyclomaticComplexity: int(totalComplexity / count),
		TestCoveragePercent:  totalCoverage / count,
		DocumentationRatio:   totalDocumentation / count / 100.0,
		VulnerabilityCount:   vulnerabilities,
		StyleViolations:      styleViolations,
	}
}

// calculateCollaborationMetrics calculates collaboration-related metrics
func (mc *MetricsCalculator) calculateCollaborationMetrics(reviews []PeerReviewEvent, history []Contribution) CollaborationMetrics {
	metrics := CollaborationMetrics{}
	
	totalReviewScore := 0.0
	reviewCount := 0
	turnaroundTimes := make([]time.Duration, 0)
	
	for _, review := range reviews {
		if review.IsReviewer {
			metrics.PeerReviewsGiven++
			totalReviewScore += review.Rating
			reviewCount++
			
			// In a real implementation, calculate actual turnaround time
			// For now, estimate based on rating (higher rating = faster turnaround)
			estimatedTurnaround := time.Duration(float64(48*time.Hour) / review.Rating)
			turnaroundTimes = append(turnaroundTimes, estimatedTurnaround)
		} else {
			metrics.PeerReviewsReceived++
		}
	}
	
	if reviewCount > 0 {
		metrics.ReviewQualityScore = (totalReviewScore / float64(reviewCount)) * 20 // Scale to 100
	}
	
	if len(turnaroundTimes) > 0 {
		total := time.Duration(0)
		for _, t := range turnaroundTimes {
			total += t
		}
		metrics.AvgReviewTurnaround = total / time.Duration(len(turnaroundTimes))
	}
	
	// Calculate other collaboration metrics based on contribution patterns
	for _, contrib := range history {
		// Documentation contributions indicate knowledge sharing
		if contrib.Type == Documentation {
			metrics.KnowledgeSharing += 10.0
		}
		
		// Code reviews indicate community engagement
		if contrib.Type == CodeReview {
			metrics.CommunityEngagement += 5.0
		}
		
		// High-quality contributions with good reviews suggest mentorship
		if contrib.QualityScore > 85 && contrib.ReviewScore > 4.0 {
			metrics.MentorshipActivities++
		}
	}
	
	// Normalize scores
	metrics.KnowledgeSharing = math.Min(100.0, metrics.KnowledgeSharing)
	metrics.CommunityEngagement = math.Min(100.0, metrics.CommunityEngagement)
	metrics.ConflictResolution = 70.0 // Default - would be calculated from actual data
	
	return metrics
}

// calculateImpactMetrics calculates real-world impact metrics
func (mc *MetricsCalculator) calculateImpactMetrics(history []Contribution) ImpactMetrics {
	metrics := ImpactMetrics{}
	
	for _, contrib := range history {
		// Categorize contributions by type and quality
		switch contrib.Type {
		case IssueResolution:
			if contrib.QualityScore > 80 {
				metrics.BugFixCount++
				if contrib.QualityScore > 90 {
					metrics.CriticalBugFixCount++
				}
			}
		case Security:
			metrics.SecurityImprovements++
		case CodeCommit:
			// Large, high-quality commits likely improve performance
			if contrib.LinesAdded > 100 && contrib.QualityScore > 85 {
				metrics.PerformanceImprovements++
			}
		}
		
		// High-quality contributions with good reviews suggest user satisfaction
		if contrib.QualityScore > 80 && contrib.ReviewScore > 4.0 {
			metrics.UserSatisfaction += contrib.ReviewScore * 5 // Scale up
		}
		
		// Test coverage improvements reduce technical debt
		if contrib.TestCoverage > 80 {
			metrics.TechnicalDebtReduction += contrib.TestCoverage / 10
		}
	}
	
	// Normalize user satisfaction
	if len(history) > 0 {
		metrics.UserSatisfaction = math.Min(100.0, metrics.UserSatisfaction/float64(len(history)))
	}
	
	// Calculate feature adoption (would be based on actual usage data)
	metrics.FeatureAdoption = 75.0 // Default assumption
	
	// Calculate system stability impact
	metrics.SystemStabilityImpact = math.Min(100.0, 
		float64(metrics.BugFixCount)*10 + float64(metrics.SecurityImprovements)*15)
	
	return metrics
}

// calculateInnovationMetrics calculates innovation and creativity metrics
func (mc *MetricsCalculator) calculateInnovationMetrics(history []Contribution) InnovationMetrics {
	metrics := InnovationMetrics{}
	creativityScore := 0.0
	
	for _, contrib := range history {
		// Large, complex contributions may be architectural
		if contrib.LinesAdded > 500 && contrib.Complexity > 5 && contrib.QualityScore > 75 {
			metrics.ArchitecturalContributions++
			creativityScore += 15
		}
		
		// High-quality, well-tested contributions may introduce new patterns
		if contrib.QualityScore > 85 && contrib.TestCoverage > 80 && contrib.Documentation > 75 {
			metrics.PatternIntroductions++
			creativityScore += 10
		}
		
		// Security contributions often require novel approaches
		if contrib.Type == Security && contrib.QualityScore > 80 {
			metrics.NovelApproaches++
			creativityScore += 12
		}
		
		// Documentation contributions may include research
		if contrib.Type == Documentation && contrib.LinesAdded > 100 {
			metrics.ResearchContributions++
			creativityScore += 8
		}
	}
	
	// Normalize creativity score
	if len(history) > 0 {
		metrics.CreativityScore = math.Min(100.0, creativityScore/float64(len(history)))
	}
	
	return metrics
}

// calculateTrendAnalysis performs trend analysis across different time periods
func (mc *MetricsCalculator) calculateTrendAnalysis(history []Contribution) TrendAnalysis {
	trends := TrendAnalysis{}
	
	now := time.Now()
	
	// Calculate contribution counts for different periods
	shortTermCount := 0
	mediumTermCount := 0
	longTermCount := 0
	
	shortTermBoundary := now.Add(-mc.shortTermWindow)
	mediumTermBoundary := now.Add(-mc.mediumTermWindow)
	longTermBoundary := now.Add(-mc.longTermWindow)
	
	for _, contrib := range history {
		if contrib.Timestamp.After(shortTermBoundary) {
			shortTermCount++
		}
		if contrib.Timestamp.After(mediumTermBoundary) {
			mediumTermCount++
		}
		if contrib.Timestamp.After(longTermBoundary) {
			longTermCount++
		}
	}
	
	// Calculate trends based on relative activity levels
	trends.ShortTermTrend = mc.calculateTrendDirection(shortTermCount, 
		mediumTermCount-shortTermCount)
	trends.MediumTermTrend = mc.calculateTrendDirection(mediumTermCount,
		longTermCount-mediumTermCount)
	trends.LongTermTrend = mc.calculateTrendDirection(longTermCount,
		len(history)-longTermCount)
	
	// Find peak and low performance periods
	trends.PeakPerformance = mc.findPeakPeriod(history, true)
	trends.LowPerformance = mc.findPeakPeriod(history, false)
	
	return trends
}

// calculateTrendDirection determines if trend is improving, stable, or declining
func (mc *MetricsCalculator) calculateTrendDirection(recent, older int) TrendDirection {
	if older == 0 {
		return TrendUnknown
	}
	
	ratio := float64(recent) / float64(older)
	
	if ratio > 1.2 {
		return TrendImproving
	} else if ratio < 0.8 {
		return TrendDeclining
	} else {
		return TrendStable
	}
}

// findPeakPeriod finds the period of highest (or lowest) performance
func (mc *MetricsCalculator) findPeakPeriod(history []Contribution, findPeak bool) PeakPeriod {
	if len(history) == 0 {
		return PeakPeriod{}
	}
	
	// Group contributions by week and find peak week
	weeklyScores := make(map[string]float64)
	weeklyContribs := make(map[string][]Contribution)
	
	for _, contrib := range history {
		// Get week identifier (year-week)
		year, week := contrib.Timestamp.ISOWeek()
		weekKey := fmt.Sprintf("%d-W%02d", year, week)
		
		weeklyScores[weekKey] += contrib.QualityScore
		weeklyContribs[weekKey] = append(weeklyContribs[weekKey], contrib)
	}
	
	// Find the peak (or trough) week
	var bestWeek string
	var bestScore float64
	first := true
	
	for week, score := range weeklyScores {
		avgScore := score / float64(len(weeklyContribs[week]))
		
		if first || (findPeak && avgScore > bestScore) || (!findPeak && avgScore < bestScore) {
			bestScore = avgScore
			bestWeek = week
			first = false
		}
	}
	
	if bestWeek == "" {
		return PeakPeriod{}
	}
	
	// Calculate start and end times for the best week
	contribs := weeklyContribs[bestWeek]
	startTime := contribs[0].Timestamp
	endTime := contribs[0].Timestamp
	
	for _, contrib := range contribs {
		if contrib.Timestamp.Before(startTime) {
			startTime = contrib.Timestamp
		}
		if contrib.Timestamp.After(endTime) {
			endTime = contrib.Timestamp
		}
	}
	
	description := fmt.Sprintf("Week with %d contributions averaging %.1f quality score",
		len(contribs), bestScore)
	
	return PeakPeriod{
		StartTime:   startTime,
		EndTime:     endTime,
		Score:       bestScore,
		Description: description,
	}
}

// Normalization functions to convert raw metrics to 0-100 scores

func (mc *MetricsCalculator) normalizeProductivityScore(prod ProductivityMetrics) float64 {
	// Weight different productivity factors
	frequencyScore := math.Min(100.0, (prod.ContributionFrequency/mc.avgContributionsPerWeek)*50)
	sizeScore := math.Min(100.0, (prod.AvgContribSize/mc.avgCodeLinesPerContrib)*50)
	velocityScore := prod.VelocityScore
	consistencyScore := prod.ConsistencyScore
	burnoutPenalty := prod.BurnoutRiskScore * 0.2 // Reduce score if high burnout risk
	
	score := (frequencyScore*0.3 + sizeScore*0.2 + velocityScore*0.25 + 
		     consistencyScore*0.25) - burnoutPenalty
	
	return math.Max(0, math.Min(100.0, score))
}

func (mc *MetricsCalculator) normalizeQualityScore(qual QualityMetrics) float64 {
	return qual.OverallScore // Already normalized
}

func (mc *MetricsCalculator) normalizeCollaborationScore(collab CollaborationMetrics) float64 {
	reviewScore := math.Min(100.0, collab.ReviewQualityScore)
	engagementScore := collab.CommunityEngagement
	knowledgeScore := collab.KnowledgeSharing
	mentorshipScore := math.Min(100.0, float64(collab.MentorshipActivities)*10)
	
	score := reviewScore*0.3 + engagementScore*0.25 + knowledgeScore*0.25 + mentorshipScore*0.2
	
	return math.Max(0, math.Min(100.0, score))
}

func (mc *MetricsCalculator) normalizeImpactScore(impact ImpactMetrics) float64 {
	bugScore := math.Min(100.0, float64(impact.BugFixCount)*10)
	perfScore := math.Min(100.0, float64(impact.PerformanceImprovements)*15)
	securityScore := math.Min(100.0, float64(impact.SecurityImprovements)*20)
	userSatScore := impact.UserSatisfaction
	stabilityScore := impact.SystemStabilityImpact
	
	score := bugScore*0.2 + perfScore*0.15 + securityScore*0.2 + 
		     userSatScore*0.25 + stabilityScore*0.2
	
	return math.Max(0, math.Min(100.0, score))
}

func (mc *MetricsCalculator) normalizeInnovationScore(innov InnovationMetrics) float64 {
	archScore := math.Min(100.0, float64(innov.ArchitecturalContributions)*25)
	patternScore := math.Min(100.0, float64(innov.PatternIntroductions)*20)
	novelScore := math.Min(100.0, float64(innov.NovelApproaches)*15)
	creativityScore := innov.CreativityScore
	
	score := archScore*0.3 + patternScore*0.25 + novelScore*0.2 + creativityScore*0.25
	
	return math.Max(0, math.Min(100.0, score))
}

// Helper function for string formatting is now imported at the top