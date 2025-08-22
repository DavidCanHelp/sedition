// Package github provides GitHub integration for real-world code analysis
package github

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	poc "github.com/davidcanhelp/sedition"
)

// GitHubClient handles GitHub API interactions
type GitHubClient struct {
	token      string
	baseURL    string
	httpClient *http.Client
	rateLimit  *RateLimit
}

// RateLimit tracks GitHub API rate limits
type RateLimit struct {
	Limit     int
	Remaining int
	ResetTime time.Time
}

// Repository represents a GitHub repository
type Repository struct {
	ID              int64     `json:"id"`
	Name            string    `json:"name"`
	FullName        string    `json:"full_name"`
	Owner           *User     `json:"owner"`
	Description     string    `json:"description"`
	Language        string    `json:"language"`
	StargazersCount int       `json:"stargazers_count"`
	ForksCount      int       `json:"forks_count"`
	Size            int       `json:"size"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	PushedAt        time.Time `json:"pushed_at"`
}

// User represents a GitHub user
type User struct {
	ID       int64  `json:"id"`
	Login    string `json:"login"`
	Name     string `json:"name"`
	Email    string `json:"email"`
	Company  string `json:"company"`
	Location string `json:"location"`
}

// Commit represents a GitHub commit
type Commit struct {
	SHA         string    `json:"sha"`
	Message     string    `json:"message"`
	Author      *User     `json:"author"`
	Committer   *User     `json:"committer"`
	Timestamp   time.Time `json:"timestamp"`
	Stats       *CommitStats `json:"stats"`
	Files       []*CommitFile `json:"files"`
	Parents     []string  `json:"parents"`
	URL         string    `json:"url"`
}

// CommitStats represents commit statistics
type CommitStats struct {
	Additions int `json:"additions"`
	Deletions int `json:"deletions"`
	Total     int `json:"total"`
}

// CommitFile represents a file changed in a commit
type CommitFile struct {
	Filename  string `json:"filename"`
	Status    string `json:"status"` // added, modified, removed
	Additions int    `json:"additions"`
	Deletions int    `json:"deletions"`
	Changes   int    `json:"changes"`
	Patch     string `json:"patch"`
}

// PullRequest represents a GitHub pull request
type PullRequest struct {
	ID          int64     `json:"id"`
	Number      int       `json:"number"`
	Title       string    `json:"title"`
	Body        string    `json:"body"`
	User        *User     `json:"user"`
	State       string    `json:"state"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	MergedAt    *time.Time `json:"merged_at"`
	Commits     int       `json:"commits"`
	Additions   int       `json:"additions"`
	Deletions   int       `json:"deletions"`
	ChangedFiles int      `json:"changed_files"`
}

// Review represents a pull request review
type Review struct {
	ID           int64     `json:"id"`
	User         *User     `json:"user"`
	Body         string    `json:"body"`
	State        string    `json:"state"` // APPROVED, CHANGES_REQUESTED, COMMENTED
	SubmittedAt  time.Time `json:"submitted_at"`
	CommitID     string    `json:"commit_id"`
}

// QualityAnalysis represents comprehensive code quality analysis
type QualityAnalysis struct {
	Repository      *Repository      `json:"repository"`
	Commit          *Commit         `json:"commit"`
	PullRequest     *PullRequest    `json:"pull_request,omitempty"`
	
	// Code metrics
	LinesOfCode     int             `json:"lines_of_code"`
	CyclomaticComplexity float64    `json:"cyclomatic_complexity"`
	TestCoverage    float64         `json:"test_coverage"`
	Documentation   float64         `json:"documentation"`
	
	// Quality scores
	OverallScore    float64         `json:"overall_score"`
	CodeQuality     float64         `json:"code_quality"`
	TestQuality     float64         `json:"test_quality"`
	SecurityScore   float64         `json:"security_score"`
	
	// Collaboration metrics
	ReviewCount     int             `json:"review_count"`
	ReviewScore     float64         `json:"review_score"`
	CollaboratorCount int           `json:"collaborator_count"`
	
	// Innovation metrics
	NoveltyScore    float64         `json:"novelty_score"`
	ImpactScore     float64         `json:"impact_score"`
	
	// Flags and issues
	SecurityIssues  []string        `json:"security_issues"`
	QualityIssues   []string        `json:"quality_issues"`
	Warnings        []string        `json:"warnings"`
	
	AnalyzedAt      time.Time       `json:"analyzed_at"`
}

// NewGitHubClient creates a new GitHub client
func NewGitHubClient(token string) *GitHubClient {
	return &GitHubClient{
		token:   token,
		baseURL: "https://api.github.com",
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		rateLimit: &RateLimit{},
	}
}

// GetRepository fetches repository information
func (c *GitHubClient) GetRepository(owner, repo string) (*Repository, error) {
	url := fmt.Sprintf("%s/repos/%s/%s", c.baseURL, owner, repo)
	
	resp, err := c.makeRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var repository Repository
	if err := json.NewDecoder(resp.Body).Decode(&repository); err != nil {
		return nil, fmt.Errorf("failed to decode repository: %w", err)
	}
	
	return &repository, nil
}

// GetCommits fetches commits from a repository
func (c *GitHubClient) GetCommits(owner, repo string, since time.Time, limit int) ([]*Commit, error) {
	url := fmt.Sprintf("%s/repos/%s/%s/commits", c.baseURL, owner, repo)
	if !since.IsZero() {
		url += fmt.Sprintf("?since=%s", since.Format(time.RFC3339))
	}
	if limit > 0 {
		separator := "?"
		if strings.Contains(url, "?") {
			separator = "&"
		}
		url += fmt.Sprintf("%sper_page=%d", separator, limit)
	}
	
	resp, err := c.makeRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var commits []*Commit
	if err := json.NewDecoder(resp.Body).Decode(&commits); err != nil {
		return nil, fmt.Errorf("failed to decode commits: %w", err)
	}
	
	// Get detailed information for each commit
	for i, commit := range commits {
		detailed, err := c.GetCommitDetails(owner, repo, commit.SHA)
		if err == nil {
			commits[i] = detailed
		}
	}
	
	return commits, nil
}

// GetCommitDetails fetches detailed commit information
func (c *GitHubClient) GetCommitDetails(owner, repo, sha string) (*Commit, error) {
	url := fmt.Sprintf("%s/repos/%s/%s/commits/%s", c.baseURL, owner, repo, sha)
	
	resp, err := c.makeRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var commit Commit
	if err := json.NewDecoder(resp.Body).Decode(&commit); err != nil {
		return nil, fmt.Errorf("failed to decode commit details: %w", err)
	}
	
	return &commit, nil
}

// GetPullRequest fetches pull request information
func (c *GitHubClient) GetPullRequest(owner, repo string, number int) (*PullRequest, error) {
	url := fmt.Sprintf("%s/repos/%s/%s/pulls/%d", c.baseURL, owner, repo, number)
	
	resp, err := c.makeRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var pr PullRequest
	if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
		return nil, fmt.Errorf("failed to decode pull request: %w", err)
	}
	
	return &pr, nil
}

// GetPullRequestReviews fetches reviews for a pull request
func (c *GitHubClient) GetPullRequestReviews(owner, repo string, number int) ([]*Review, error) {
	url := fmt.Sprintf("%s/repos/%s/%s/pulls/%d/reviews", c.baseURL, owner, repo, number)
	
	resp, err := c.makeRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var reviews []*Review
	if err := json.NewDecoder(resp.Body).Decode(&reviews); err != nil {
		return nil, fmt.Errorf("failed to decode reviews: %w", err)
	}
	
	return reviews, nil
}

// AnalyzeCommitQuality performs comprehensive quality analysis on a commit
func (c *GitHubClient) AnalyzeCommitQuality(owner, repo, sha string) (*QualityAnalysis, error) {
	// Fetch commit details
	commit, err := c.GetCommitDetails(owner, repo, sha)
	if err != nil {
		return nil, fmt.Errorf("failed to get commit details: %w", err)
	}
	
	// Fetch repository information
	repository, err := c.GetRepository(owner, repo)
	if err != nil {
		return nil, fmt.Errorf("failed to get repository: %w", err)
	}
	
	analysis := &QualityAnalysis{
		Repository: repository,
		Commit:     commit,
		AnalyzedAt: time.Now(),
	}
	
	// Analyze code metrics
	c.analyzeCodeMetrics(analysis)
	
	// Analyze collaboration
	c.analyzeCollaboration(analysis, owner, repo)
	
	// Security analysis
	c.analyzeCodeSecurity(analysis)
	
	// Calculate overall scores
	c.calculateQualityScores(analysis)
	
	return analysis, nil
}

// analyzeCodeMetrics analyzes code complexity and quality metrics
func (c *GitHubClient) analyzeCodeMetrics(analysis *QualityAnalysis) {
	if analysis.Commit == nil || analysis.Commit.Stats == nil {
		return
	}
	
	totalChanges := analysis.Commit.Stats.Total
	analysis.LinesOfCode = totalChanges
	
	// Calculate cyclomatic complexity (simplified heuristic)
	complexityScore := 1.0
	for _, file := range analysis.Commit.Files {
		if strings.HasSuffix(file.Filename, ".go") ||
		   strings.HasSuffix(file.Filename, ".js") ||
		   strings.HasSuffix(file.Filename, ".py") ||
		   strings.HasSuffix(file.Filename, ".java") {
			
			// Count complexity indicators in patch
			complexity := c.calculateFileComplexity(file.Patch)
			complexityScore += complexity
		}
	}
	
	// Normalize complexity
	if totalChanges > 0 {
		analysis.CyclomaticComplexity = complexityScore / float64(totalChanges) * 100
	}
	
	// Analyze test coverage
	analysis.TestCoverage = c.analyzeTestCoverage(analysis.Commit.Files)
	
	// Analyze documentation
	analysis.Documentation = c.analyzeDocumentation(analysis.Commit.Files)
}

// calculateFileComplexity calculates complexity score for a file patch
func (c *GitHubClient) calculateFileComplexity(patch string) float64 {
	if patch == "" {
		return 0.0
	}
	
	complexity := 1.0
	
	// Count complexity indicators
	complexityPatterns := []string{
		`\bif\b`, `\belse\b`, `\bfor\b`, `\bwhile\b`, `\bswitch\b`, `\bcase\b`,
		`\btry\b`, `\bcatch\b`, `\bfinally\b`, `\bthrow\b`, `\breturn\b`,
		`&&`, `\|\|`, `\?`, `:`, `\bbreak\b`, `\bcontinue\b`,
	}
	
	for _, pattern := range complexityPatterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllString(patch, -1)
		complexity += float64(len(matches)) * 0.5
	}
	
	// Penalize deeply nested structures
	braceDepth := 0
	maxDepth := 0
	for _, char := range patch {
		switch char {
		case '{':
			braceDepth++
			if braceDepth > maxDepth {
				maxDepth = braceDepth
			}
		case '}':
			braceDepth--
		}
	}
	
	complexity += float64(maxDepth) * 2.0
	
	return complexity
}

// analyzeTestCoverage analyzes test coverage based on files changed
func (c *GitHubClient) analyzeTestCoverage(files []*CommitFile) float64 {
	totalFiles := 0
	testFiles := 0
	
	for _, file := range files {
		if file.Status == "removed" {
			continue
		}
		
		totalFiles++
		
		// Check if it's a test file
		filename := strings.ToLower(file.Filename)
		if strings.Contains(filename, "test") ||
		   strings.Contains(filename, "spec") ||
		   strings.HasSuffix(filename, "_test.go") ||
		   strings.HasSuffix(filename, "_test.py") ||
		   strings.HasSuffix(filename, ".spec.js") ||
		   strings.HasSuffix(filename, ".test.js") {
			testFiles++
		}
	}
	
	if totalFiles == 0 {
		return 50.0 // Neutral score if no files
	}
	
	// Calculate coverage score
	coverage := float64(testFiles) / float64(totalFiles) * 100
	
	// Adjust for commit size (larger commits should have more tests)
	if totalFiles > 5 && testFiles == 0 {
		coverage = 0.0 // Penalize large commits without tests
	} else if totalFiles <= 2 && testFiles > 0 {
		coverage = 100.0 // Small commits with tests get full score
	}
	
	return math.Min(100.0, coverage)
}

// analyzeDocumentation analyzes documentation quality
func (c *GitHubClient) analyzeDocumentation(files []*CommitFile) float64 {
	totalCodeFiles := 0
	documentedFiles := 0
	totalComments := 0
	
	for _, file := range files {
		if file.Status == "removed" {
			continue
		}
		
		// Check if it's a code file
		filename := strings.ToLower(file.Filename)
		if strings.HasSuffix(filename, ".go") ||
		   strings.HasSuffix(filename, ".js") ||
		   strings.HasSuffix(filename, ".py") ||
		   strings.HasSuffix(filename, ".java") ||
		   strings.HasSuffix(filename, ".cpp") ||
		   strings.HasSuffix(filename, ".c") {
			
			totalCodeFiles++
			
			// Count comments and documentation in patch
			comments := c.countComments(file.Patch, filename)
			totalComments += comments
			
			if comments > 0 {
				documentedFiles++
			}
		}
		
		// Check for documentation files
		if strings.HasSuffix(filename, ".md") ||
		   strings.HasSuffix(filename, ".rst") ||
		   strings.HasSuffix(filename, ".txt") ||
		   strings.Contains(filename, "readme") ||
		   strings.Contains(filename, "doc") {
			documentedFiles++
		}
	}
	
	if totalCodeFiles == 0 {
		return 70.0 // Neutral score for non-code commits
	}
	
	// Calculate documentation score
	docCoverage := float64(documentedFiles) / float64(totalCodeFiles) * 100
	
	// Bonus for comprehensive commenting
	if totalComments > totalCodeFiles*2 {
		docCoverage = math.Min(100.0, docCoverage*1.2)
	}
	
	return docCoverage
}

// countComments counts comments in a file patch
func (c *GitHubClient) countComments(patch, filename string) int {
	if patch == "" {
		return 0
	}
	
	commentPatterns := []string{}
	
	// Language-specific comment patterns
	if strings.HasSuffix(filename, ".go") ||
	   strings.HasSuffix(filename, ".js") ||
	   strings.HasSuffix(filename, ".java") ||
	   strings.HasSuffix(filename, ".cpp") ||
	   strings.HasSuffix(filename, ".c") {
		commentPatterns = []string{`//.*`, `/\*.*?\*/`}
	} else if strings.HasSuffix(filename, ".py") {
		commentPatterns = []string{`#.*`, `""".*?"""`, `'''.*?'''`}
	}
	
	comments := 0
	for _, pattern := range commentPatterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllString(patch, -1)
		for _, match := range matches {
			if len(strings.TrimSpace(match)) > 5 { // Ignore very short comments
				comments++
			}
		}
	}
	
	return comments
}

// analyzeCollaboration analyzes collaboration metrics
func (c *GitHubClient) analyzeCollaboration(analysis *QualityAnalysis, owner, repo string) {
	if analysis.Commit == nil {
		return
	}
	
	// For simplicity, set default collaboration values
	// In a full implementation, this would analyze PR reviews, discussions, etc.
	analysis.ReviewCount = 0
	analysis.ReviewScore = 50.0
	analysis.CollaboratorCount = 1
	
	// Try to find related PR
	if prNumber := c.extractPRNumber(analysis.Commit.Message); prNumber > 0 {
		if pr, err := c.GetPullRequest(owner, repo, prNumber); err == nil {
			analysis.PullRequest = pr
			
			// Get reviews
			if reviews, err := c.GetPullRequestReviews(owner, repo, prNumber); err == nil {
				analysis.ReviewCount = len(reviews)
				analysis.ReviewScore = c.calculateReviewScore(reviews)
			}
		}
	}
}

// extractPRNumber extracts PR number from commit message
func (c *GitHubClient) extractPRNumber(message string) int {
	// Look for patterns like "#123" or "PR #123" or "Merge pull request #123"
	patterns := []string{
		`#(\d+)`,
		`PR #(\d+)`,
		`pull request #(\d+)`,
	}
	
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindStringSubmatch(message)
		if len(matches) > 1 {
			if num, err := strconv.Atoi(matches[1]); err == nil {
				return num
			}
		}
	}
	
	return 0
}

// calculateReviewScore calculates overall review score
func (c *GitHubClient) calculateReviewScore(reviews []*Review) float64 {
	if len(reviews) == 0 {
		return 0.0
	}
	
	approvals := 0
	changesRequested := 0
	comments := 0
	
	for _, review := range reviews {
		switch review.State {
		case "APPROVED":
			approvals++
		case "CHANGES_REQUESTED":
			changesRequested++
		case "COMMENTED":
			comments++
		}
	}
	
	// Calculate weighted score
	score := float64(approvals*100 + comments*50 - changesRequested*25)
	if len(reviews) > 0 {
		score = score / float64(len(reviews))
	}
	
	return math.Max(0, math.Min(100, score))
}

// analyzeCodeSecurity performs basic security analysis
func (c *GitHubClient) analyzeCodeSecurity(analysis *QualityAnalysis) {
	if analysis.Commit == nil {
		return
	}
	
	securityIssues := []string{}
	securityScore := 100.0
	
	for _, file := range analysis.Commit.Files {
		issues := c.findSecurityIssues(file.Patch, file.Filename)
		securityIssues = append(securityIssues, issues...)
		securityScore -= float64(len(issues)) * 10 // Deduct 10 points per issue
	}
	
	analysis.SecurityIssues = securityIssues
	analysis.SecurityScore = math.Max(0, securityScore)
}

// findSecurityIssues finds potential security issues in code
func (c *GitHubClient) findSecurityIssues(patch, filename string) []string {
	if patch == "" {
		return nil
	}
	
	issues := []string{}
	
	// Common security anti-patterns
	securityPatterns := map[string]string{
		`password\s*=\s*["'][^"']+["']`:      "Hardcoded password detected",
		`api[_-]?key\s*=\s*["'][^"']+["']`:  "Hardcoded API key detected",
		`secret\s*=\s*["'][^"']+["']`:       "Hardcoded secret detected",
		`eval\s*\(`:                         "Dangerous eval() usage",
		`exec\s*\(`:                         "Dangerous exec() usage",
		`system\s*\(`:                       "System command execution",
		`shell_exec\s*\(`:                   "Shell command execution",
		`\$_GET\[`:                          "Potential XSS vulnerability",
		`\$_POST\[`:                         "Potential injection vulnerability",
		`innerHTML\s*=`:                     "Potential XSS vulnerability",
		`document\.write\s*\(`:              "Dangerous document.write usage",
		`dangerouslySetInnerHTML`:           "React XSS vulnerability",
		`SELECT\s+.*\s+FROM\s+.*\s+WHERE.*\+`: "SQL injection vulnerability",
	}
	
	for pattern, description := range securityPatterns {
		re := regexp.MustCompile(`(?i)` + pattern)
		if re.MatchString(patch) {
			issues = append(issues, fmt.Sprintf("%s in %s", description, filename))
		}
	}
	
	return issues
}

// calculateQualityScores calculates final quality scores
func (c *GitHubClient) calculateQualityScores(analysis *QualityAnalysis) {
	// Code quality score (weighted average)
	analysis.CodeQuality = (
		analysis.CyclomaticComplexity*0.3 +
		analysis.TestCoverage*0.4 +
		analysis.Documentation*0.3)
	
	// Test quality score
	if analysis.TestCoverage > 80 {
		analysis.TestQuality = 95.0
	} else if analysis.TestCoverage > 50 {
		analysis.TestQuality = 75.0
	} else if analysis.TestCoverage > 20 {
		analysis.TestQuality = 50.0
	} else {
		analysis.TestQuality = 25.0
	}
	
	// Calculate novelty score based on file types and changes
	analysis.NoveltyScore = c.calculateNoveltyScore(analysis.Commit)
	
	// Calculate impact score based on lines changed and files affected
	analysis.ImpactScore = c.calculateImpactScore(analysis.Commit)
	
	// Overall score (weighted combination)
	weights := map[string]float64{
		"code":         0.25,
		"test":         0.20,
		"security":     0.20,
		"review":       0.15,
		"novelty":      0.10,
		"impact":       0.10,
	}
	
	analysis.OverallScore = (
		analysis.CodeQuality*weights["code"] +
		analysis.TestQuality*weights["test"] +
		analysis.SecurityScore*weights["security"] +
		analysis.ReviewScore*weights["review"] +
		analysis.NoveltyScore*weights["novelty"] +
		analysis.ImpactScore*weights["impact"])
	
	// Quality issues
	if analysis.OverallScore < 50 {
		analysis.QualityIssues = append(analysis.QualityIssues, "Low overall quality score")
	}
	if analysis.TestCoverage < 30 {
		analysis.QualityIssues = append(analysis.QualityIssues, "Insufficient test coverage")
	}
	if analysis.CyclomaticComplexity > 15 {
		analysis.QualityIssues = append(analysis.QualityIssues, "High cyclomatic complexity")
	}
	if analysis.SecurityScore < 80 {
		analysis.QualityIssues = append(analysis.QualityIssues, "Security concerns detected")
	}
}

// calculateNoveltyScore calculates how novel/innovative the changes are
func (c *GitHubClient) calculateNoveltyScore(commit *Commit) float64 {
	if commit == nil || commit.Stats == nil {
		return 50.0
	}
	
	noveltyScore := 50.0 // Base score
	
	// New files are more novel
	newFiles := 0
	for _, file := range commit.Files {
		if file.Status == "added" {
			newFiles++
		}
	}
	
	if newFiles > 0 {
		noveltyScore += float64(newFiles) * 10
	}
	
	// Large additions suggest new functionality
	if commit.Stats.Additions > 100 {
		noveltyScore += 15.0
	} else if commit.Stats.Additions > 50 {
		noveltyScore += 10.0
	}
	
	// Check for innovative patterns
	for _, file := range commit.Files {
		if strings.Contains(file.Patch, "algorithm") ||
		   strings.Contains(file.Patch, "optimization") ||
		   strings.Contains(file.Patch, "innovation") ||
		   strings.Contains(file.Patch, "novel") {
			noveltyScore += 5.0
		}
	}
	
	return math.Min(100.0, noveltyScore)
}

// calculateImpactScore calculates the potential impact of changes
func (c *GitHubClient) calculateImpactScore(commit *Commit) float64 {
	if commit == nil || commit.Stats == nil {
		return 50.0
	}
	
	impactScore := 50.0
	
	// More files affected = higher impact
	filesChanged := len(commit.Files)
	if filesChanged > 10 {
		impactScore += 20.0
	} else if filesChanged > 5 {
		impactScore += 10.0
	}
	
	// Total lines changed
	totalChanges := commit.Stats.Total
	if totalChanges > 500 {
		impactScore += 25.0
	} else if totalChanges > 200 {
		impactScore += 15.0
	} else if totalChanges > 50 {
		impactScore += 10.0
	}
	
	// Check for critical files
	for _, file := range commit.Files {
		filename := strings.ToLower(file.Filename)
		if strings.Contains(filename, "main") ||
		   strings.Contains(filename, "core") ||
		   strings.Contains(filename, "engine") ||
		   strings.Contains(filename, "server") ||
		   strings.Contains(filename, "api") {
			impactScore += 10.0
		}
	}
	
	return math.Min(100.0, impactScore)
}

// makeRequest makes an HTTP request to GitHub API
func (c *GitHubClient) makeRequest(method, url string, body []byte) (*http.Response, error) {
	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		return nil, err
	}
	
	// Add authentication
	if c.token != "" {
		req.Header.Set("Authorization", "token "+c.token)
	}
	
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	req.Header.Set("User-Agent", "Sedition-PoC-Analyzer/1.0")
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	
	// Update rate limit info
	c.updateRateLimit(resp)
	
	if resp.StatusCode >= 400 {
		body, _ := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("GitHub API error %d: %s", resp.StatusCode, string(body))
	}
	
	return resp, nil
}

// updateRateLimit updates rate limit information from response headers
func (c *GitHubClient) updateRateLimit(resp *http.Response) {
	if limit := resp.Header.Get("X-RateLimit-Limit"); limit != "" {
		if val, err := strconv.Atoi(limit); err == nil {
			c.rateLimit.Limit = val
		}
	}
	
	if remaining := resp.Header.Get("X-RateLimit-Remaining"); remaining != "" {
		if val, err := strconv.Atoi(remaining); err == nil {
			c.rateLimit.Remaining = val
		}
	}
	
	if reset := resp.Header.Get("X-RateLimit-Reset"); reset != "" {
		if val, err := strconv.ParseInt(reset, 10, 64); err == nil {
			c.rateLimit.ResetTime = time.Unix(val, 0)
		}
	}
}

// ConvertToPoC converts GitHub analysis to PoC contribution
func (analysis *QualityAnalysis) ConvertToPoC() *poc.Contribution {
	if analysis.Commit == nil {
		return nil
	}
	
	contribution := &poc.Contribution{
		ID:              analysis.Commit.SHA,
		Timestamp:       analysis.AnalyzedAt,
		Type:            poc.CodeCommit,
		LinesAdded:      analysis.Commit.Stats.Additions,
		LinesModified:   analysis.Commit.Stats.Deletions, // Approximation
		TestCoverage:    analysis.TestCoverage,
		Complexity:      analysis.CyclomaticComplexity,
		Documentation:   analysis.Documentation,
		QualityScore:    analysis.OverallScore,
		PeerReviews:     analysis.ReviewCount,
		ReviewScore:     analysis.ReviewScore,
	}
	
	return contribution
}

// SaveAnalysis saves analysis to file
func (analysis *QualityAnalysis) SaveAnalysis(filename string) error {
	data, err := json.MarshalIndent(analysis, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal analysis: %w", err)
	}
	
	// Ensure directory exists
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}
	
	err = ioutil.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write analysis file: %w", err)
	}
	
	return nil
}

// Additional math import needed
import "math"