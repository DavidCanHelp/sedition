// Package poc implements quality analysis for code contributions
// This module evaluates the quality of code contributions using multiple metrics:
// - Cyclomatic complexity analysis
// - Test coverage assessment
// - Documentation coverage
// - Code style and best practices
// - Security vulnerability detection
package poc

import (
	"errors"
	"math"
	"regexp"
	"strings"
)

// QualityAnalyzer analyzes the quality of code contributions
type QualityAnalyzer struct {
	// Weight factors for different quality metrics (sum should equal 1.0)
	complexityWeight    float64
	testCoverageWeight  float64
	documentationWeight float64
	styleWeight         float64
	securityWeight      float64
	
	// Thresholds for quality assessment
	maxComplexity       float64
	minTestCoverage     float64
	minDocumentation    float64
	
	// Pattern matchers for analysis
	functionPattern     *regexp.Regexp
	commentPattern      *regexp.Regexp
	testPattern         *regexp.Regexp
	securityPatterns    []*regexp.Regexp
}

// QualityMetrics holds detailed quality analysis results
type QualityMetrics struct {
	OverallScore      float64 // Final quality score (0-100)
	ComplexityScore   float64 // Cyclomatic complexity score
	TestCoverageScore float64 // Test coverage adequacy score
	DocumentationScore float64 // Documentation completeness score
	StyleScore        float64 // Code style compliance score
	SecurityScore     float64 // Security assessment score
	
	// Detailed metrics
	CyclomaticComplexity int     // Raw cyclomatic complexity value
	TestCoveragePercent  float64 // Actual test coverage percentage
	DocumentationRatio   float64 // Documentation to code ratio
	VulnerabilityCount   int     // Number of potential security issues
	StyleViolations      int     // Number of style violations
	
	// Recommendations
	Recommendations []string // Suggestions for improvement
}

// CodeAnalysis represents the analysis of a code contribution
type CodeAnalysis struct {
	Language        string
	TotalLines      int
	CodeLines       int
	CommentLines    int
	BlankLines      int
	Functions       []FunctionMetrics
	TestFiles       []string
	SecurityIssues  []SecurityIssue
	StyleIssues     []StyleIssue
}

// FunctionMetrics holds metrics for individual functions
type FunctionMetrics struct {
	Name                string
	LineCount           int
	CyclomaticComplexity int
	Parameters          int
	HasDocumentation    bool
	HasTests           bool
}

// SecurityIssue represents a potential security vulnerability
type SecurityIssue struct {
	Type        SecurityIssueType
	Severity    Severity
	Line        int
	Description string
	Suggestion  string
}

// SecurityIssueType defines types of security issues
type SecurityIssueType int

const (
	SQLInjection SecurityIssueType = iota
	XSSVulnerability
	BufferOverflow
	InsecureRandom
	WeakCrypto
	AuthenticationBypass
	AuthorizationBypass
	InformationDisclosure
)

// StyleIssue represents a code style violation
type StyleIssue struct {
	Type        StyleIssueType
	Line        int
	Description string
	Suggestion  string
}

// StyleIssueType defines types of style issues
type StyleIssueType int

const (
	NamingConvention StyleIssueType = iota
	Indentation
	LineLength
	MissingDocumentation
	UnusedVariable
	DeadCode
	MagicNumber
)

// Severity levels for issues
type Severity int

const (
	Low Severity = iota
	Medium
	High
	Critical
)

// NewQualityAnalyzer creates a new quality analyzer with default settings
func NewQualityAnalyzer() *QualityAnalyzer {
	qa := &QualityAnalyzer{
		// Quality metric weights
		complexityWeight:    0.25,
		testCoverageWeight:  0.30,
		documentationWeight: 0.20,
		styleWeight:         0.15,
		securityWeight:      0.10,
		
		// Quality thresholds
		maxComplexity:       10.0, // McCabe's recommended maximum
		minTestCoverage:     80.0, // Minimum acceptable test coverage
		minDocumentation:    70.0, // Minimum documentation coverage
	}
	
	// Compile regex patterns
	qa.functionPattern = regexp.MustCompile(`(?m)^func\s+(\w+)\s*\(`)
	qa.commentPattern = regexp.MustCompile(`(?m)^\s*//|^\s*/\*|\*/`)
	qa.testPattern = regexp.MustCompile(`(?m)func\s+Test\w+|func\s+Benchmark\w+|_test\.go$`)
	
	// Security vulnerability patterns
	qa.securityPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)sql.*query.*\+`),          // SQL injection risk
		regexp.MustCompile(`(?i)eval\s*\(`),               // Code injection risk
		regexp.MustCompile(`(?i)exec\s*\(`),               // Command injection risk
		regexp.MustCompile(`(?i)rand\.Read\s*\(`),         // Weak random number generation
		regexp.MustCompile(`(?i)\bmd5\b|\bsha1\b`),        // Weak cryptographic hash
		regexp.MustCompile(`(?i)password.*=.*["'][^"']+["']`), // Hardcoded passwords
	}
	
	return qa
}

// AnalyzeContribution performs comprehensive quality analysis on a contribution
func (qa *QualityAnalyzer) AnalyzeContribution(contrib Contribution) (float64, error) {
	// This is a simplified implementation - in a real system, you would
	// analyze the actual code diff/files submitted with the contribution
	
	// For now, we'll use the contribution metadata to estimate quality
	metrics := qa.calculateQualityMetrics(contrib)
	
	return metrics.OverallScore, nil
}

// calculateQualityMetrics calculates detailed quality metrics
func (qa *QualityAnalyzer) calculateQualityMetrics(contrib Contribution) *QualityMetrics {
	metrics := &QualityMetrics{
		Recommendations: make([]string, 0),
	}
	
	// Calculate complexity score (inverse relationship - lower complexity = higher score)
	if contrib.Complexity > 0 {
		complexityScore := math.Max(0, 100.0 * (1.0 - (contrib.Complexity / qa.maxComplexity)))
		metrics.ComplexityScore = math.Min(100.0, complexityScore)
		metrics.CyclomaticComplexity = int(contrib.Complexity)
		
		if contrib.Complexity > qa.maxComplexity {
			metrics.Recommendations = append(metrics.Recommendations,
				"Consider refactoring complex functions to reduce cyclomatic complexity")
		}
	} else {
		metrics.ComplexityScore = 85.0 // Default for non-code contributions
	}
	
	// Calculate test coverage score
	metrics.TestCoverageScore = math.Min(100.0, contrib.TestCoverage * 1.25) // Bonus for high coverage
	metrics.TestCoveragePercent = contrib.TestCoverage
	
	if contrib.TestCoverage < qa.minTestCoverage {
		metrics.Recommendations = append(metrics.Recommendations,
			"Increase test coverage to meet project standards")
	}
	
	// Calculate documentation score
	metrics.DocumentationScore = math.Min(100.0, contrib.Documentation * 1.1)
	metrics.DocumentationRatio = contrib.Documentation / 100.0
	
	if contrib.Documentation < qa.minDocumentation {
		metrics.Recommendations = append(metrics.Recommendations,
			"Add more comprehensive documentation for new features")
	}
	
	// Calculate style score based on peer reviews
	if contrib.PeerReviews > 0 && contrib.ReviewScore > 0 {
		// High review scores indicate good style compliance
		metrics.StyleScore = math.Min(100.0, contrib.ReviewScore * 20.0) // Scale 5-point to 100-point
		
		if contrib.ReviewScore < 4.0 {
			metrics.Recommendations = append(metrics.Recommendations,
				"Address code style issues identified in peer reviews")
		}
	} else {
		metrics.StyleScore = 75.0 // Default for unreviewed contributions
		metrics.Recommendations = append(metrics.Recommendations,
			"Submit contribution for peer review to validate code quality")
	}
	
	// Calculate security score
	metrics.SecurityScore = qa.calculateSecurityScore(contrib)
	
	// Calculate overall weighted score
	metrics.OverallScore = 
		metrics.ComplexityScore * qa.complexityWeight +
		metrics.TestCoverageScore * qa.testCoverageWeight +
		metrics.DocumentationScore * qa.documentationWeight +
		metrics.StyleScore * qa.styleWeight +
		metrics.SecurityScore * qa.securityWeight
	
	// Apply contribution type modifiers
	metrics.OverallScore = qa.applyContributionTypeModifier(contrib.Type, metrics.OverallScore)
	
	// Ensure score is within valid range
	metrics.OverallScore = math.Max(0.0, math.Min(100.0, metrics.OverallScore))
	
	return metrics
}

// calculateSecurityScore assesses security aspects of the contribution
func (qa *QualityAnalyzer) calculateSecurityScore(contrib Contribution) float64 {
	baseScore := 90.0 // Start with high security score
	
	// Penalty for security-sensitive contribution types
	switch contrib.Type {
	case Security:
		baseScore = 95.0 // Security contributions should have high standards
	case CodeCommit:
		// Regular code commits have standard security assessment
		if contrib.LinesAdded > 100 {
			baseScore -= 5.0 // Larger changes have higher risk
		}
	}
	
	// In a real implementation, this would analyze actual code for:
	// - SQL injection vulnerabilities
	// - XSS vulnerabilities  
	// - Buffer overflows
	// - Insecure random number generation
	// - Weak cryptographic practices
	// - Authentication/authorization bypasses
	
	return baseScore
}

// applyContributionTypeModifier adjusts score based on contribution type
func (qa *QualityAnalyzer) applyContributionTypeModifier(contribType ContributionType, score float64) float64 {
	switch contribType {
	case Testing:
		// Testing contributions get bonus for improving project quality
		return math.Min(100.0, score * 1.1)
	case Documentation:
		// Documentation contributions are valuable but different quality metrics
		return math.Min(100.0, score * 1.05)
	case Security:
		// Security contributions have higher quality standards
		return score * 0.95 // Slight penalty to ensure thorough review
	case CodeReview:
		// Code reviews improve overall project quality
		return math.Min(100.0, score * 1.08)
	case IssueResolution:
		// Issue resolution demonstrates problem-solving capability
		return math.Min(100.0, score * 1.03)
	default:
		return score
	}
}

// AnalyzeCodeDiff analyzes a code diff for detailed metrics
func (qa *QualityAnalyzer) AnalyzeCodeDiff(diff string, language string) (*CodeAnalysis, error) {
	if diff == "" {
		return nil, errors.New("empty code diff provided")
	}
	
	analysis := &CodeAnalysis{
		Language:       language,
		Functions:      make([]FunctionMetrics, 0),
		TestFiles:      make([]string, 0),
		SecurityIssues: make([]SecurityIssue, 0),
		StyleIssues:    make([]StyleIssue, 0),
	}
	
	lines := strings.Split(diff, "\n")
	analysis.TotalLines = len(lines)
	
	for i, line := range lines {
		// Count different types of lines
		trimmedLine := strings.TrimSpace(line)
		
		if trimmedLine == "" {
			analysis.BlankLines++
		} else if qa.commentPattern.MatchString(line) {
			analysis.CommentLines++
		} else {
			analysis.CodeLines++
		}
		
		// Analyze functions
		if matches := qa.functionPattern.FindStringSubmatch(line); len(matches) > 1 {
			funcMetrics := qa.analyzeFunctionMetrics(lines, i, matches[1])
			analysis.Functions = append(analysis.Functions, funcMetrics)
		}
		
		// Check for security issues
		qa.checkSecurityIssues(line, i, analysis)
		
		// Check for style issues
		qa.checkStyleIssues(line, i, analysis)
	}
	
	// Identify test files
	qa.identifyTestFiles(diff, analysis)
	
	return analysis, nil
}

// analyzeFunctionMetrics analyzes metrics for a specific function
func (qa *QualityAnalyzer) analyzeFunctionMetrics(lines []string, startLine int, funcName string) FunctionMetrics {
	metrics := FunctionMetrics{
		Name: funcName,
	}
	
	// Count function lines and calculate complexity
	braceCount := 0
	inFunction := false
	
	for i := startLine; i < len(lines); i++ {
		line := lines[i]
		
		if strings.Contains(line, "{") {
			braceCount += strings.Count(line, "{")
			inFunction = true
		}
		
		if strings.Contains(line, "}") {
			braceCount -= strings.Count(line, "}")
			if inFunction && braceCount == 0 {
				metrics.LineCount = i - startLine + 1
				break
			}
		}
		
		if inFunction {
			// Calculate cyclomatic complexity
			complexity := 0
			complexity += strings.Count(line, "if ")
			complexity += strings.Count(line, "for ")
			complexity += strings.Count(line, "while ")
			complexity += strings.Count(line, "switch ")
			complexity += strings.Count(line, "case ")
			complexity += strings.Count(line, "&&")
			complexity += strings.Count(line, "||")
			
			metrics.CyclomaticComplexity += complexity
		}
	}
	
	// Check for documentation (comment before function)
	if startLine > 0 && qa.commentPattern.MatchString(lines[startLine-1]) {
		metrics.HasDocumentation = true
	}
	
	return metrics
}

// checkSecurityIssues checks for potential security vulnerabilities
func (qa *QualityAnalyzer) checkSecurityIssues(line string, lineNum int, analysis *CodeAnalysis) {
	for i, pattern := range qa.securityPatterns {
		if pattern.MatchString(line) {
			issue := SecurityIssue{
				Type:     SecurityIssueType(i),
				Severity: qa.getSecuritySeverity(SecurityIssueType(i)),
				Line:     lineNum + 1,
			}
			
			switch SecurityIssueType(i) {
			case 0: // SQL injection pattern
				issue.Description = "Potential SQL injection vulnerability detected"
				issue.Suggestion = "Use parameterized queries or prepared statements"
			case 1: // Code injection
				issue.Description = "Potential code injection vulnerability with eval()"
				issue.Suggestion = "Avoid dynamic code execution, use safer alternatives"
			case 2: // Command injection
				issue.Description = "Potential command injection vulnerability"
				issue.Suggestion = "Validate and sanitize input before executing commands"
			case 3: // Weak random
				issue.Description = "Using weak random number generation"
				issue.Suggestion = "Use cryptographically secure random number generator"
			case 4: // Weak crypto
				issue.Description = "Using weak cryptographic hash function"
				issue.Suggestion = "Use SHA-256 or stronger hash functions"
			case 5: // Hardcoded password
				issue.Description = "Potential hardcoded password detected"
				issue.Suggestion = "Store credentials securely using environment variables"
			}
			
			analysis.SecurityIssues = append(analysis.SecurityIssues, issue)
		}
	}
}

// checkStyleIssues checks for code style violations
func (qa *QualityAnalyzer) checkStyleIssues(line string, lineNum int, analysis *CodeAnalysis) {
	// Check line length
	if len(line) > 120 {
		issue := StyleIssue{
			Type:        LineLength,
			Line:        lineNum + 1,
			Description: "Line exceeds maximum length of 120 characters",
			Suggestion:  "Break long lines into multiple lines for better readability",
		}
		analysis.StyleIssues = append(analysis.StyleIssues, issue)
	}
	
	// Check for magic numbers
	magicNumberPattern := regexp.MustCompile(`\b[0-9]{2,}\b`)
	if magicNumberPattern.MatchString(line) && !strings.Contains(line, "//") {
		issue := StyleIssue{
			Type:        MagicNumber,
			Line:        lineNum + 1,
			Description: "Magic number detected - consider using named constants",
			Suggestion:  "Define meaningful constant names for numeric values",
		}
		analysis.StyleIssues = append(analysis.StyleIssues, issue)
	}
	
	// Check indentation (simplified - assumes spaces)
	if strings.HasPrefix(line, "\t") {
		issue := StyleIssue{
			Type:        Indentation,
			Line:        lineNum + 1,
			Description: "Using tabs instead of spaces for indentation",
			Suggestion:  "Use consistent spacing (typically 2 or 4 spaces) for indentation",
		}
		analysis.StyleIssues = append(analysis.StyleIssues, issue)
	}
}

// identifyTestFiles identifies test files in the contribution
func (qa *QualityAnalyzer) identifyTestFiles(diff string, analysis *CodeAnalysis) {
	lines := strings.Split(diff, "\n")
	
	for _, line := range lines {
		if strings.HasPrefix(line, "+++") || strings.HasPrefix(line, "---") {
			filename := strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(line, "+++"), "---"))
			if qa.testPattern.MatchString(filename) {
				analysis.TestFiles = append(analysis.TestFiles, filename)
			}
		}
	}
}

// getSecuritySeverity returns the severity level for a security issue type
func (qa *QualityAnalyzer) getSecuritySeverity(issueType SecurityIssueType) Severity {
	switch issueType {
	case SQLInjection, XSSVulnerability, BufferOverflow:
		return Critical
	case AuthenticationBypass, AuthorizationBypass:
		return High
	case InformationDisclosure, WeakCrypto:
		return Medium
	case InsecureRandom:
		return Low
	default:
		return Medium
	}
}

// GetQualityThresholds returns the current quality thresholds
func (qa *QualityAnalyzer) GetQualityThresholds() map[string]float64 {
	return map[string]float64{
		"maxComplexity":    qa.maxComplexity,
		"minTestCoverage":  qa.minTestCoverage,
		"minDocumentation": qa.minDocumentation,
	}
}

// UpdateQualityThresholds updates the quality assessment thresholds
func (qa *QualityAnalyzer) UpdateQualityThresholds(thresholds map[string]float64) {
	if val, exists := thresholds["maxComplexity"]; exists {
		qa.maxComplexity = val
	}
	if val, exists := thresholds["minTestCoverage"]; exists {
		qa.minTestCoverage = val
	}
	if val, exists := thresholds["minDocumentation"]; exists {
		qa.minDocumentation = val
	}
}