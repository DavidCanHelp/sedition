package ml

import (
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"regexp"
	"strings"
)

// MLQualityAnalyzer provides machine learning-based code quality analysis
type MLQualityAnalyzer struct {
	models          map[string]*MLModel
	featureScaler   *StandardScaler
	vocabularyIndex map[string]int
	maxVocabSize    int
	modelWeights    map[string]float64
}

// MLModel represents a trained machine learning model
type MLModel struct {
	Type            string                 `json:"type"`
	Weights         [][]float64            `json:"weights"`
	Biases          []float64              `json:"biases"`
	Architecture    []int                  `json:"architecture"`
	ActivationFunc  string                 `json:"activation_func"`
	OutputFunc      string                 `json:"output_func"`
	Hyperparameters map[string]interface{} `json:"hyperparameters"`
	TrainingMetrics *TrainingMetrics       `json:"training_metrics"`
}

// TrainingMetrics stores model performance metrics
type TrainingMetrics struct {
	Accuracy        float64       `json:"accuracy"`
	Precision       float64       `json:"precision"`
	Recall          float64       `json:"recall"`
	F1Score         float64       `json:"f1_score"`
	AUC             float64       `json:"auc"`
	ConfusionMatrix [][]int       `json:"confusion_matrix"`
	LossHistory     []float64     `json:"loss_history"`
	ValidationLoss  []float64     `json:"validation_loss"`
	EpochMetrics    []EpochMetric `json:"epoch_metrics"`
}

// EpochMetric represents metrics for a single training epoch
type EpochMetric struct {
	Epoch          int     `json:"epoch"`
	TrainingLoss   float64 `json:"training_loss"`
	ValidationLoss float64 `json:"validation_loss"`
	TrainingAcc    float64 `json:"training_accuracy"`
	ValidationAcc  float64 `json:"validation_accuracy"`
	LearningRate   float64 `json:"learning_rate"`
}

// CodeFeatures represents extracted features from code
type CodeFeatures struct {
	// Structural features
	LinesOfCode          int `json:"lines_of_code"`
	CyclomaticComplexity int `json:"cyclomatic_complexity"`
	NestingDepth         int `json:"nesting_depth"`
	FunctionCount        int `json:"function_count"`
	ClassCount           int `json:"class_count"`
	InterfaceCount       int `json:"interface_count"`

	// Quality metrics
	CommentRatio       float64 `json:"comment_ratio"`
	TestCoverage       float64 `json:"test_coverage"`
	DuplicationRatio   float64 `json:"duplication_ratio"`
	TechnicalDebtRatio float64 `json:"technical_debt_ratio"`

	// Semantic features
	VariableNaming float64  `json:"variable_naming"`
	FunctionNaming float64  `json:"function_naming"`
	APIConsistency float64  `json:"api_consistency"`
	DesignPatterns []string `json:"design_patterns"`
	CodeSmells     []string `json:"code_smells"`

	// Security features
	SecurityVulns   int     `json:"security_vulnerabilities"`
	InputValidation float64 `json:"input_validation"`
	ErrorHandling   float64 `json:"error_handling"`

	// Performance features
	BigOComplexity string  `json:"big_o_complexity"`
	MemoryUsage    float64 `json:"memory_usage"`
	IOOperations   int     `json:"io_operations"`

	// Text-based features (for NLP)
	TokenVector       []float64 `json:"token_vector"`
	SemanticEmbedding []float64 `json:"semantic_embedding"`
	SyntaxFeatures    []float64 `json:"syntax_features"`
}

// QualityPrediction represents the ML model's quality prediction
type QualityPrediction struct {
	OverallQuality     float64            `json:"overall_quality"`
	Confidence         float64            `json:"confidence"`
	ComponentScores    map[string]float64 `json:"component_scores"`
	Recommendations    []string           `json:"recommendations"`
	RiskFactors        []string           `json:"risk_factors"`
	ModelContributions map[string]float64 `json:"model_contributions"`
	FeatureImportance  map[string]float64 `json:"feature_importance"`
	Explanations       []string           `json:"explanations"`
}

// StandardScaler for feature normalization
type StandardScaler struct {
	Mean []float64 `json:"mean"`
	Std  []float64 `json:"std"`
}

// NewMLQualityAnalyzer creates a new ML-based quality analyzer
func NewMLQualityAnalyzer() *MLQualityAnalyzer {
	analyzer := &MLQualityAnalyzer{
		models:          make(map[string]*MLModel),
		maxVocabSize:    10000,
		vocabularyIndex: make(map[string]int),
		modelWeights: map[string]float64{
			"structural":      0.25,
			"semantic":        0.30,
			"security":        0.20,
			"performance":     0.15,
			"maintainability": 0.10,
		},
	}

	// Initialize pre-trained models
	analyzer.initializeModels()
	return analyzer
}

// AnalyzeCode performs comprehensive ML-based code quality analysis
func (mla *MLQualityAnalyzer) AnalyzeCode(code string, language string, metadata map[string]interface{}) (*QualityPrediction, error) {
	// Extract comprehensive features
	features, err := mla.extractFeatures(code, language, metadata)
	if err != nil {
		return nil, fmt.Errorf("feature extraction failed: %v", err)
	}

	// Normalize features
	normalizedFeatures := mla.normalizeFeatures(features)

	// Run ensemble prediction
	prediction := mla.ensemblePredict(normalizedFeatures)

	// Generate explanations
	prediction.Explanations = mla.generateExplanations(features, prediction)

	return prediction, nil
}

// extractFeatures extracts comprehensive code features for ML analysis
func (mla *MLQualityAnalyzer) extractFeatures(code, language string, metadata map[string]interface{}) (*CodeFeatures, error) {
	features := &CodeFeatures{}

	// Basic structural analysis
	features.LinesOfCode = len(strings.Split(code, "\n"))
	features.FunctionCount = mla.countFunctions(code, language)
	features.ClassCount = mla.countClasses(code, language)
	features.CommentRatio = mla.calculateCommentRatio(code, language)

	// Advanced complexity analysis
	if language == "go" {
		goFeatures, err := mla.extractGoFeatures(code)
		if err == nil {
			features.CyclomaticComplexity = goFeatures.CyclomaticComplexity
			features.NestingDepth = goFeatures.NestingDepth
			features.InterfaceCount = goFeatures.InterfaceCount
		}
	}

	// Semantic analysis
	features.VariableNaming = mla.analyzeNamingConventions(code, "variable")
	features.FunctionNaming = mla.analyzeNamingConventions(code, "function")
	features.APIConsistency = mla.analyzeAPIConsistency(code)
	features.DesignPatterns = mla.detectDesignPatterns(code)
	features.CodeSmells = mla.detectCodeSmells(code)

	// Security analysis
	features.SecurityVulns = mla.detectSecurityVulnerabilities(code)
	features.InputValidation = mla.analyzeInputValidation(code)
	features.ErrorHandling = mla.analyzeErrorHandling(code)

	// Performance analysis
	features.BigOComplexity = mla.estimateComplexity(code)
	features.IOOperations = mla.countIOOperations(code)

	// Generate text-based features for NLP models
	features.TokenVector = mla.generateTokenVector(code)
	features.SemanticEmbedding = mla.generateSemanticEmbedding(code)
	features.SyntaxFeatures = mla.extractSyntaxFeatures(code, language)

	return features, nil
}

// extractGoFeatures extracts Go-specific features using AST analysis
func (mla *MLQualityAnalyzer) extractGoFeatures(code string) (*CodeFeatures, error) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	features := &CodeFeatures{}

	// AST visitor for complexity analysis
	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncDecl:
			features.FunctionCount++
			complexity := mla.calculateCyclomaticComplexity(x)
			features.CyclomaticComplexity += complexity

		case *ast.TypeSpec:
			if _, ok := x.Type.(*ast.InterfaceType); ok {
				features.InterfaceCount++
			} else if _, ok := x.Type.(*ast.StructType); ok {
				features.ClassCount++
			}

		case *ast.BlockStmt:
			depth := mla.calculateNestingDepth(x, 0)
			if depth > features.NestingDepth {
				features.NestingDepth = depth
			}
		}
		return true
	})

	return features, nil
}

// calculateCyclomaticComplexity calculates cyclomatic complexity for a function
func (mla *MLQualityAnalyzer) calculateCyclomaticComplexity(fn *ast.FuncDecl) int {
	complexity := 1 // Base complexity

	ast.Inspect(fn, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SwitchStmt, *ast.TypeSwitchStmt:
			complexity++
		case *ast.CaseClause:
			complexity++
		}
		return true
	})

	return complexity
}

// calculateNestingDepth calculates maximum nesting depth
func (mla *MLQualityAnalyzer) calculateNestingDepth(block *ast.BlockStmt, currentDepth int) int {
	maxDepth := currentDepth

	for _, stmt := range block.List {
		switch s := stmt.(type) {
		case *ast.IfStmt:
			if s.Body != nil {
				depth := mla.calculateNestingDepth(s.Body, currentDepth+1)
				if depth > maxDepth {
					maxDepth = depth
				}
			}
		case *ast.ForStmt:
			if s.Body != nil {
				depth := mla.calculateNestingDepth(s.Body, currentDepth+1)
				if depth > maxDepth {
					maxDepth = depth
				}
			}
		case *ast.BlockStmt:
			depth := mla.calculateNestingDepth(s, currentDepth+1)
			if depth > maxDepth {
				maxDepth = depth
			}
		}
	}

	return maxDepth
}

// analyzeNamingConventions analyzes naming quality using heuristics
func (mla *MLQualityAnalyzer) analyzeNamingConventions(code, entityType string) float64 {
	var patterns []string

	switch entityType {
	case "variable":
		patterns = []string{
			`\b[a-z][a-zA-Z0-9]*\b`, // camelCase
			`\b[a-z][a-z0-9_]*\b`,   // snake_case
		}
	case "function":
		patterns = []string{
			`\bfunc\s+[A-Z][a-zA-Z0-9]*`, // PascalCase (Go public)
			`\bfunc\s+[a-z][a-zA-Z0-9]*`, // camelCase (Go private)
		}
	}

	totalMatches := 0
	goodMatches := 0

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllString(code, -1)
		totalMatches += len(matches)
		goodMatches += len(matches)
	}

	// Check for bad naming patterns
	badPatterns := []string{
		`\b[a-z]\b`,           // single letter variables
		`\b(temp|tmp|data)\b`, // generic names
		`\b[A-Z]{2,}\b`,       // ALL_CAPS
	}

	for _, pattern := range badPatterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllString(code, -1)
		totalMatches += len(matches)
	}

	if totalMatches == 0 {
		return 0.5 // Neutral score if no patterns found
	}

	return float64(goodMatches) / float64(totalMatches)
}

// detectDesignPatterns detects common design patterns in code
func (mla *MLQualityAnalyzer) detectDesignPatterns(code string) []string {
	patterns := []string{}

	// Singleton pattern
	if matched, _ := regexp.MatchString(`(?i)singleton|getInstance`, code); matched {
		patterns = append(patterns, "Singleton")
	}

	// Factory pattern
	if matched, _ := regexp.MatchString(`(?i)factory|create[A-Z]`, code); matched {
		patterns = append(patterns, "Factory")
	}

	// Observer pattern
	if matched, _ := regexp.MatchString(`(?i)observer|notify|subscribe`, code); matched {
		patterns = append(patterns, "Observer")
	}

	// Strategy pattern
	if matched, _ := regexp.MatchString(`(?i)strategy|algorithm`, code); matched {
		patterns = append(patterns, "Strategy")
	}

	// Decorator pattern
	if matched, _ := regexp.MatchString(`(?i)decorator|wrap`, code); matched {
		patterns = append(patterns, "Decorator")
	}

	return patterns
}

// detectCodeSmells detects common code smells
func (mla *MLQualityAnalyzer) detectCodeSmells(code string) []string {
	smells := []string{}

	// Long method smell (>50 lines)
	lines := strings.Split(code, "\n")
	functionLines := 0
	inFunction := false

	for _, line := range lines {
		if matched, _ := regexp.MatchString(`^\s*func\s+`, line); matched {
			inFunction = true
			functionLines = 1
		} else if inFunction {
			if strings.Contains(line, "}") && len(strings.TrimSpace(line)) == 1 {
				if functionLines > 50 {
					smells = append(smells, "Long Method")
				}
				inFunction = false
				functionLines = 0
			} else {
				functionLines++
			}
		}
	}

	// God class smell (>500 lines)
	if len(lines) > 500 {
		smells = append(smells, "God Class")
	}

	// Duplicate code smell
	duplicateRatio := mla.calculateDuplicationRatio(code)
	if duplicateRatio > 0.15 {
		smells = append(smells, "Duplicate Code")
	}

	// Magic numbers smell
	if matched, _ := regexp.MatchString(`\b\d{2,}\b`, code); matched {
		smells = append(smells, "Magic Numbers")
	}

	// Dead code smell
	if matched, _ := regexp.MatchString(`(?i)todo|fixme|xxx|hack`, code); matched {
		smells = append(smells, "Technical Debt")
	}

	return smells
}

// detectSecurityVulnerabilities detects potential security issues
func (mla *MLQualityAnalyzer) detectSecurityVulnerabilities(code string) int {
	vulnCount := 0

	// SQL injection patterns
	sqlPatterns := []string{
		`(?i)query.*\+.*`,
		`(?i)exec.*\+.*`,
		`(?i)format.*%.*sql`,
	}

	for _, pattern := range sqlPatterns {
		if matched, _ := regexp.MatchString(pattern, code); matched {
			vulnCount++
		}
	}

	// XSS patterns
	xssPatterns := []string{
		`(?i)innerHTML.*\+`,
		`(?i)document\.write.*\+`,
		`(?i)eval\(`,
	}

	for _, pattern := range xssPatterns {
		if matched, _ := regexp.MatchString(pattern, code); matched {
			vulnCount++
		}
	}

	// Path traversal
	if matched, _ := regexp.MatchString(`\.\.\/|\.\.\\`, code); matched {
		vulnCount++
	}

	// Hardcoded secrets
	secretPatterns := []string{
		`(?i)password\s*=\s*["'][^"']{8,}["']`,
		`(?i)api[_-]?key\s*=\s*["'][^"']{16,}["']`,
		`(?i)secret\s*=\s*["'][^"']{16,}["']`,
	}

	for _, pattern := range secretPatterns {
		if matched, _ := regexp.MatchString(pattern, code); matched {
			vulnCount++
		}
	}

	return vulnCount
}

// generateTokenVector creates a token-based vector representation
func (mla *MLQualityAnalyzer) generateTokenVector(code string) []float64 {
	// Tokenize code
	tokens := mla.tokenizeCode(code)

	// Create bag-of-words vector
	vector := make([]float64, mla.maxVocabSize)
	tokenCounts := make(map[string]int)

	for _, token := range tokens {
		tokenCounts[token]++
	}

	// Convert to vector using vocabulary index
	for token, count := range tokenCounts {
		if idx, exists := mla.vocabularyIndex[token]; exists && idx < len(vector) {
			vector[idx] = float64(count)
		}
	}

	// Normalize vector (TF-IDF-like)
	totalTokens := float64(len(tokens))
	for i := range vector {
		if vector[i] > 0 {
			vector[i] = vector[i] / totalTokens * math.Log(1000.0/vector[i])
		}
	}

	return vector
}

// generateSemanticEmbedding creates semantic embeddings (simplified)
func (mla *MLQualityAnalyzer) generateSemanticEmbedding(code string) []float64 {
	// Simplified semantic embedding using hash-based features
	embedding := make([]float64, 300) // Standard embedding size

	// Use different code aspects to generate embedding
	aspects := []string{
		code,
		strings.ToLower(code),
		regexp.MustCompile(`\W+`).ReplaceAllString(code, " "),
	}

	for i, aspect := range aspects {
		hash := sha256.Sum256([]byte(aspect))
		for j := 0; j < len(embedding)/len(aspects); j++ {
			idx := i*len(embedding)/len(aspects) + j
			if idx < len(embedding) {
				embedding[idx] = float64(hash[j%32])/255.0*2.0 - 1.0
			}
		}
	}

	return embedding
}

// ensemblePredict combines predictions from multiple models
func (mla *MLQualityAnalyzer) ensemblePredict(features *CodeFeatures) *QualityPrediction {
	prediction := &QualityPrediction{
		ComponentScores:    make(map[string]float64),
		ModelContributions: make(map[string]float64),
		FeatureImportance:  make(map[string]float64),
		Recommendations:    []string{},
		RiskFactors:        []string{},
	}

	// Predict with each model
	structuralScore := mla.predictStructural(features)
	semanticScore := mla.predictSemantic(features)
	securityScore := mla.predictSecurity(features)
	performanceScore := mla.predictPerformance(features)
	maintainabilityScore := mla.predictMaintainability(features)

	// Store component scores
	prediction.ComponentScores["structural"] = structuralScore
	prediction.ComponentScores["semantic"] = semanticScore
	prediction.ComponentScores["security"] = securityScore
	prediction.ComponentScores["performance"] = performanceScore
	prediction.ComponentScores["maintainability"] = maintainabilityScore

	// Calculate weighted overall score
	prediction.OverallQuality = 0.0
	for component, score := range prediction.ComponentScores {
		weight := mla.modelWeights[component]
		prediction.OverallQuality += score * weight
		prediction.ModelContributions[component] = score * weight
	}

	// Calculate confidence based on score variance
	scores := []float64{structuralScore, semanticScore, securityScore, performanceScore, maintainabilityScore}
	prediction.Confidence = mla.calculateConfidence(scores)

	// Generate recommendations based on low scores
	if structuralScore < 0.7 {
		prediction.Recommendations = append(prediction.Recommendations, "Reduce cyclomatic complexity and nesting depth")
	}
	if securityScore < 0.8 {
		prediction.Recommendations = append(prediction.Recommendations, "Address security vulnerabilities and improve input validation")
		prediction.RiskFactors = append(prediction.RiskFactors, "Security vulnerabilities detected")
	}
	if performanceScore < 0.6 {
		prediction.Recommendations = append(prediction.Recommendations, "Optimize algorithmic complexity and reduce I/O operations")
	}

	return prediction
}

// predictStructural analyzes structural code quality
func (mla *MLQualityAnalyzer) predictStructural(features *CodeFeatures) float64 {
	score := 1.0

	// Penalize high complexity
	if features.CyclomaticComplexity > 10 {
		score -= 0.1 * float64(features.CyclomaticComplexity-10) / 10.0
	}

	// Penalize deep nesting
	if features.NestingDepth > 5 {
		score -= 0.1 * float64(features.NestingDepth-5) / 5.0
	}

	// Reward good function count ratio
	if features.LinesOfCode > 0 {
		funcRatio := float64(features.FunctionCount) / float64(features.LinesOfCode) * 100
		if funcRatio > 1 && funcRatio < 5 {
			score += 0.1
		}
	}

	return math.Max(0.0, math.Min(1.0, score))
}

// predictSemantic analyzes semantic code quality
func (mla *MLQualityAnalyzer) predictSemantic(features *CodeFeatures) float64 {
	score := 0.0

	// Naming quality
	score += features.VariableNaming * 0.3
	score += features.FunctionNaming * 0.3
	score += features.APIConsistency * 0.2

	// Design patterns boost
	score += float64(len(features.DesignPatterns)) * 0.05

	// Code smells penalty
	score -= float64(len(features.CodeSmells)) * 0.1

	// Comment ratio
	if features.CommentRatio > 0.1 && features.CommentRatio < 0.3 {
		score += 0.2
	}

	return math.Max(0.0, math.Min(1.0, score))
}

// predictSecurity analyzes security quality
func (mla *MLQualityAnalyzer) predictSecurity(features *CodeFeatures) float64 {
	score := 1.0

	// Security vulnerabilities penalty
	score -= float64(features.SecurityVulns) * 0.2

	// Input validation bonus
	score += features.InputValidation * 0.2

	// Error handling bonus
	score += features.ErrorHandling * 0.2

	return math.Max(0.0, math.Min(1.0, score))
}

// predictPerformance analyzes performance quality
func (mla *MLQualityAnalyzer) predictPerformance(features *CodeFeatures) float64 {
	score := 0.8 // Base performance score

	// Big O complexity analysis
	switch features.BigOComplexity {
	case "O(1)", "O(log n)":
		score += 0.2
	case "O(n)":
		// No change
	case "O(n log n)":
		score -= 0.1
	case "O(n^2)":
		score -= 0.2
	case "O(n^3)", "O(2^n)":
		score -= 0.4
	}

	// I/O operations penalty
	if features.IOOperations > 10 {
		score -= 0.1 * float64(features.IOOperations-10) / 10.0
	}

	return math.Max(0.0, math.Min(1.0, score))
}

// predictMaintainability analyzes maintainability quality
func (mla *MLQualityAnalyzer) predictMaintainability(features *CodeFeatures) float64 {
	score := 0.0

	// Comment ratio
	if features.CommentRatio > 0.1 {
		score += 0.3
	}

	// Test coverage
	score += features.TestCoverage * 0.4

	// Code duplication penalty
	score -= features.DuplicationRatio * 0.3

	// Technical debt penalty
	score -= features.TechnicalDebtRatio * 0.2

	return math.Max(0.0, math.Min(1.0, score))
}

// Helper functions

func (mla *MLQualityAnalyzer) countFunctions(code, language string) int {
	patterns := map[string]string{
		"go":     `^\s*func\s+`,
		"python": `^\s*def\s+`,
		"java":   `^\s*(public|private|protected)?\s*(static\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(`,
	}

	pattern, exists := patterns[language]
	if !exists {
		pattern = patterns["go"] // Default
	}

	re := regexp.MustCompile(pattern)
	matches := re.FindAllString(code, -1)
	return len(matches)
}

func (mla *MLQualityAnalyzer) countClasses(code, language string) int {
	patterns := map[string]string{
		"go":     `^\s*type\s+[A-Z][a-zA-Z0-9]*\s+struct`,
		"python": `^\s*class\s+`,
		"java":   `^\s*(public|private|protected)?\s*class\s+`,
	}

	pattern, exists := patterns[language]
	if !exists {
		pattern = patterns["go"] // Default
	}

	re := regexp.MustCompile(pattern)
	matches := re.FindAllString(code, -1)
	return len(matches)
}

func (mla *MLQualityAnalyzer) calculateCommentRatio(code, language string) float64 {
	lines := strings.Split(code, "\n")
	commentLines := 0
	totalLines := len(lines)

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "/*") {
			commentLines++
		}
	}

	if totalLines == 0 {
		return 0.0
	}

	return float64(commentLines) / float64(totalLines)
}

func (mla *MLQualityAnalyzer) analyzeAPIConsistency(code string) float64 {
	// Simplified API consistency analysis
	// Check for consistent naming patterns in function declarations
	funcPattern := regexp.MustCompile(`func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(`)
	matches := funcPattern.FindAllStringSubmatch(code, -1)

	if len(matches) < 2 {
		return 1.0 // Not enough functions to analyze
	}

	// Check naming consistency (camelCase vs PascalCase)
	camelCase := 0
	pascalCase := 0

	for _, match := range matches {
		if len(match) > 1 {
			funcName := match[1]
			if len(funcName) > 0 {
				firstChar := funcName[0]
				if firstChar >= 'A' && firstChar <= 'Z' {
					pascalCase++
				} else {
					camelCase++
				}
			}
		}
	}

	total := camelCase + pascalCase
	consistency := float64(max(camelCase, pascalCase)) / float64(total)
	return consistency
}

func (mla *MLQualityAnalyzer) calculateDuplicationRatio(code string) float64 {
	lines := strings.Split(code, "\n")
	lineMap := make(map[string]int)
	duplicateLines := 0

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) > 5 { // Ignore very short lines
			lineMap[trimmed]++
			if lineMap[trimmed] > 1 {
				duplicateLines++
			}
		}
	}

	if len(lines) == 0 {
		return 0.0
	}

	return float64(duplicateLines) / float64(len(lines))
}

func (mla *MLQualityAnalyzer) analyzeInputValidation(code string) float64 {
	validationPatterns := []string{
		`(?i)validate`,
		`(?i)check.*input`,
		`(?i)sanitize`,
		`if.*len\(.*\)`,
		`if.*nil`,
	}

	validationCount := 0
	for _, pattern := range validationPatterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllString(code, -1)
		validationCount += len(matches)
	}

	// Normalize by code length
	lines := len(strings.Split(code, "\n"))
	if lines == 0 {
		return 0.0
	}

	return math.Min(1.0, float64(validationCount)/float64(lines)*10)
}

func (mla *MLQualityAnalyzer) analyzeErrorHandling(code string) float64 {
	errorPatterns := []string{
		`if\s+err\s*!=\s*nil`,
		`(?i)try\s*{`,
		`(?i)catch\s*\(`,
		`(?i)throw\s+`,
		`(?i)error\s*:=`,
	}

	errorHandlingCount := 0
	for _, pattern := range errorPatterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllString(code, -1)
		errorHandlingCount += len(matches)
	}

	// Normalize by code length
	lines := len(strings.Split(code, "\n"))
	if lines == 0 {
		return 0.0
	}

	return math.Min(1.0, float64(errorHandlingCount)/float64(lines)*5)
}

func (mla *MLQualityAnalyzer) estimateComplexity(code string) string {
	// Simple heuristic-based complexity estimation

	// Check for nested loops
	nestedLoopPattern := regexp.MustCompile(`for.*{[^}]*for`)
	if nestedLoopPattern.MatchString(code) {
		return "O(n^2)"
	}

	// Check for single loops
	loopPattern := regexp.MustCompile(`for\s+`)
	loopMatches := loopPattern.FindAllString(code, -1)
	if len(loopMatches) > 0 {
		return "O(n)"
	}

	// Check for recursive patterns
	if matched, _ := regexp.MatchString(`func.*\{[^}]*\w+\([^)]*\)`, code); matched {
		return "O(log n)"
	}

	return "O(1)"
}

func (mla *MLQualityAnalyzer) countIOOperations(code string) int {
	ioPatterns := []string{
		`(?i)read`,
		`(?i)write`,
		`(?i)open`,
		`(?i)close`,
		`(?i)file`,
		`(?i)database`,
		`(?i)query`,
		`(?i)http`,
	}

	ioCount := 0
	for _, pattern := range ioPatterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllString(code, -1)
		ioCount += len(matches)
	}

	return ioCount
}

func (mla *MLQualityAnalyzer) tokenizeCode(code string) []string {
	// Simple tokenization
	re := regexp.MustCompile(`\w+`)
	tokens := re.FindAllString(code, -1)

	// Convert to lowercase and filter
	var filteredTokens []string
	for _, token := range tokens {
		if len(token) > 2 {
			filteredTokens = append(filteredTokens, strings.ToLower(token))
		}
	}

	return filteredTokens
}

func (mla *MLQualityAnalyzer) extractSyntaxFeatures(code, language string) []float64 {
	features := make([]float64, 50) // Fixed-size syntax feature vector

	// Count different syntax elements
	features[0] = float64(strings.Count(code, "{"))      // Braces
	features[1] = float64(strings.Count(code, "("))      // Parentheses
	features[2] = float64(strings.Count(code, "["))      // Brackets
	features[3] = float64(strings.Count(code, "if"))     // Conditionals
	features[4] = float64(strings.Count(code, "for"))    // Loops
	features[5] = float64(strings.Count(code, "func"))   // Functions
	features[6] = float64(strings.Count(code, "var"))    // Variables
	features[7] = float64(strings.Count(code, "return")) // Returns

	// Normalize by code length
	lines := float64(len(strings.Split(code, "\n")))
	if lines > 0 {
		for i := range features {
			features[i] /= lines
		}
	}

	return features
}

func (mla *MLQualityAnalyzer) normalizeFeatures(features *CodeFeatures) *CodeFeatures {
	if mla.featureScaler == nil {
		return features // No scaler available
	}

	// Create feature vector for normalization
	featureVector := []float64{
		float64(features.LinesOfCode),
		float64(features.CyclomaticComplexity),
		float64(features.NestingDepth),
		float64(features.FunctionCount),
		features.CommentRatio,
		features.VariableNaming,
		features.FunctionNaming,
		float64(features.SecurityVulns),
		features.InputValidation,
		features.ErrorHandling,
	}

	// Apply normalization
	for i, val := range featureVector {
		if i < len(mla.featureScaler.Mean) && i < len(mla.featureScaler.Std) {
			if mla.featureScaler.Std[i] != 0 {
				featureVector[i] = (val - mla.featureScaler.Mean[i]) / mla.featureScaler.Std[i]
			}
		}
	}

	// Update features with normalized values
	normalizedFeatures := *features
	normalizedFeatures.LinesOfCode = int(featureVector[0])
	normalizedFeatures.CyclomaticComplexity = int(featureVector[1])
	normalizedFeatures.NestingDepth = int(featureVector[2])
	normalizedFeatures.FunctionCount = int(featureVector[3])
	normalizedFeatures.CommentRatio = featureVector[4]
	normalizedFeatures.VariableNaming = featureVector[5]
	normalizedFeatures.FunctionNaming = featureVector[6]
	normalizedFeatures.SecurityVulns = int(featureVector[7])
	normalizedFeatures.InputValidation = featureVector[8]
	normalizedFeatures.ErrorHandling = featureVector[9]

	return &normalizedFeatures
}

func (mla *MLQualityAnalyzer) calculateConfidence(scores []float64) float64 {
	// Calculate variance to determine confidence
	mean := 0.0
	for _, score := range scores {
		mean += score
	}
	mean /= float64(len(scores))

	variance := 0.0
	for _, score := range scores {
		variance += (score - mean) * (score - mean)
	}
	variance /= float64(len(scores))

	// Higher variance = lower confidence
	confidence := 1.0 - math.Min(variance, 1.0)
	return confidence
}

func (mla *MLQualityAnalyzer) generateExplanations(features *CodeFeatures, prediction *QualityPrediction) []string {
	explanations := []string{}

	// Generate explanations based on feature values and predictions
	if features.CyclomaticComplexity > 10 {
		explanations = append(explanations, fmt.Sprintf("High cyclomatic complexity (%d) reduces code maintainability", features.CyclomaticComplexity))
	}

	if features.SecurityVulns > 0 {
		explanations = append(explanations, fmt.Sprintf("Detected %d potential security vulnerabilities", features.SecurityVulns))
	}

	if features.CommentRatio < 0.1 {
		explanations = append(explanations, "Low comment ratio may affect code maintainability")
	}

	if len(features.CodeSmells) > 0 {
		explanations = append(explanations, fmt.Sprintf("Detected code smells: %s", strings.Join(features.CodeSmells, ", ")))
	}

	if len(features.DesignPatterns) > 0 {
		explanations = append(explanations, fmt.Sprintf("Good use of design patterns: %s", strings.Join(features.DesignPatterns, ", ")))
	}

	return explanations
}

func (mla *MLQualityAnalyzer) initializeModels() {
	// Initialize pre-trained model weights (simplified)
	// In practice, these would be loaded from trained model files

	mla.featureScaler = &StandardScaler{
		Mean: []float64{100, 5, 3, 10, 0.15, 0.8, 0.8, 0, 0.7, 0.8},
		Std:  []float64{50, 3, 2, 5, 0.1, 0.2, 0.2, 1, 0.2, 0.2},
	}

	// Initialize vocabulary for token vectors
	commonTokens := []string{
		"func", "var", "if", "for", "return", "struct", "interface", "package",
		"import", "const", "type", "map", "slice", "channel", "goroutine",
		"error", "string", "int", "bool", "float", "byte", "rune",
	}

	for i, token := range commonTokens {
		if i < mla.maxVocabSize {
			mla.vocabularyIndex[token] = i
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
