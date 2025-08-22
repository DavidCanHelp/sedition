package optimization

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// SelfImprovingSystem continuously learns and improves without human intervention
// This makes the system genuinely better over time through experience
type SelfImprovingSystem struct {
	ctx                   context.Context
	cancel                context.CancelFunc
	mu                    sync.RWMutex
	experienceLearner     *ExperienceLearner
	performanceOptimizer  *AutomaticPerformanceOptimizer
	configurationEvolver  *ConfigurationEvolver
	anomalyLearner        *AnomalyLearner
	workloadPredictor     *WorkloadPredictor
	adaptiveAlgorithms    *AdaptiveAlgorithms
	knowledgeBase         *KnowledgeBase
	experimentRunner      *ExperimentRunner
	feedbackProcessor     *FeedbackProcessor
	improvementMetrics    *ImprovementMetrics
}

// ExperienceLearner learns from every consensus round
type ExperienceLearner struct {
	experienceBuffer      *CircularBuffer
	patternRecognizer     *PatternRecognizer
	outcomePredictor      *OutcomePredictor
	decisionTree          *DecisionTree
	experienceReplay      *ExperienceReplay
	learningRate          float64
	explorationRate       float64
	totalExperiences      int64
	successfulPatterns    map[string]*SuccessPattern
	failurePatterns       map[string]*FailurePattern
}

// AutomaticPerformanceOptimizer continuously tunes performance
type AutomaticPerformanceOptimizer struct {
	performanceHistory    *TimeSeriesDB
	optimizationTargets   map[string]float64
	currentSettings       map[string]interface{}
	optimizationSpace     *OptimizationSpace
	gradientOptimizer     *GradientOptimizer
	simulatedAnnealing    *SimulatedAnnealing
	particleSwarm         *ParticleSwarmOptimizer
	optimizationHistory   []*OptimizationAttempt
	improvementThreshold  float64
	autoTuning            bool
}

// ConfigurationEvolver evolves optimal configurations over time
type ConfigurationEvolver struct {
	population            []*Configuration
	fitnessFunction       FitnessFunction
	mutationRate          float64
	crossoverRate         float64
	elitismRate           float64
	generationCount       int64
	bestConfiguration     *Configuration
	evolutionHistory      []*Generation
	diversityMaintainer   *DiversityMaintainer
	nicheFinder           *NicheFinder
}

// AnomalyLearner learns to detect and handle new types of anomalies
type AnomalyLearner struct {
	anomalyDetector       *OnlineAnomalyDetector
	clusteringEngine      *ClusteringEngine
	outlierDetector       *OutlierDetector
	anomalyDatabase       *AnomalyDatabase
	adaptiveThresholds    map[string]float64
	learningWindow        time.Duration
	falsePositiveRate     float64
	detectionAccuracy     float64
	newAnomalyTypes       []*AnomalyType
}

// WorkloadPredictor predicts future workload patterns
type WorkloadPredictor struct {
	timeSeriesForecaster  *TimeSeriesForecaster
	seasonalDecomposer    *SeasonalDecomposer
	trendAnalyzer         *TrendAnalyzer
	workloadClassifier    *WorkloadClassifier
	predictiveModels      map[string]*PredictiveModel
	predictionHorizon     time.Duration
	predictionAccuracy    float64
	confidenceIntervals   map[string][2]float64
}

// AdaptiveAlgorithms switches between algorithms based on conditions
type AdaptiveAlgorithms struct {
	algorithmLibrary      map[string]ConsensusAlgorithm
	performanceProfiles   map[string]*AlgorithmProfile
	contextAnalyzer       *ContextAnalyzer
	algorithmSelector     *AlgorithmSelector
	transitionManager     *TransitionManager
	hybridExecutor        *HybridExecutor
	currentAlgorithm      string
	algorithmHistory      []*AlgorithmSwitch
}

// KnowledgeBase stores learned knowledge persistently
type KnowledgeBase struct {
	facts                 map[string]*Fact
	rules                 map[string]*Rule
	heuristics            map[string]*Heuristic
	bestPractices         map[string]*BestPractice
	performanceModels     map[string]*Model
	causalRelationships   map[string]*CausalRelation
	knowledgeGraph        *KnowledgeGraph
	reasoningEngine       *ReasoningEngine
	lastUpdated           time.Time
	version               int64
}

// Core learning structures
type SuccessPattern struct {
	patternID             string
	conditions            map[string]interface{}
	actions               []Action
	outcomes              []Outcome
	successRate           float64
	occurrences           int64
	lastSeen              time.Time
	confidence            float64
}

type FailurePattern struct {
	patternID             string
	triggerConditions     map[string]interface{}
	failureMode           string
	mitigation            []Action
	preventionStrategy    *Strategy
	occurrences           int64
	lastSeen              time.Time
	severity              float64
}

type Configuration struct {
	configID              string
	parameters            map[string]interface{}
	fitness               float64
	generation            int64
	parents               []string
	mutations             []Mutation
	performance           *PerformanceProfile
	stabilityScore        float64
	adoptionTime          time.Time
}

type OptimizationAttempt struct {
	timestamp             time.Time
	targetMetric          string
	previousValue         float64
	newValue              float64
	improvement           float64
	technique             string
	parameters            map[string]interface{}
	successful            bool
	rollbackRequired      bool
}

type AnomalyType struct {
	typeID                string
	signature             []float64
	detectionMethod       string
	severity              float64
	frequency             float64
	mitigationActions     []Action
	learningExamples      []Example
	detectionAccuracy     float64
}

type PredictiveModel struct {
	modelType             string
	features              []string
	weights               []float64
	accuracy              float64
	lastTraining          time.Time
	trainingData          []DataPoint
	validationMetrics     *ValidationMetrics
	updateFrequency       time.Duration
}

// NewSelfImprovingSystem creates a system that gets better autonomously
func NewSelfImprovingSystem() *SelfImprovingSystem {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &SelfImprovingSystem{
		ctx:    ctx,
		cancel: cancel,
		experienceLearner: &ExperienceLearner{
			experienceBuffer:   NewCircularBuffer(100000), // Store 100k experiences
			patternRecognizer:  NewPatternRecognizer(),
			outcomePredictor:   NewOutcomePredictor(),
			decisionTree:       NewDecisionTree(),
			experienceReplay:   NewExperienceReplay(),
			learningRate:       0.01,
			explorationRate:    0.1, // 10% exploration
			successfulPatterns: make(map[string]*SuccessPattern),
			failurePatterns:    make(map[string]*FailurePattern),
		},
		performanceOptimizer: &AutomaticPerformanceOptimizer{
			performanceHistory:   NewTimeSeriesDB(),
			optimizationTargets: map[string]float64{
				"latency_ms":     10.0,
				"throughput_tps": 10000.0,
				"success_rate":   0.999,
			},
			currentSettings:      make(map[string]interface{}),
			optimizationSpace:    NewOptimizationSpace(),
			gradientOptimizer:    NewGradientOptimizer(),
			simulatedAnnealing:   NewSimulatedAnnealing(),
			particleSwarm:        NewParticleSwarmOptimizer(),
			improvementThreshold: 0.01, // 1% improvement threshold
			autoTuning:           true,
		},
		configurationEvolver: &ConfigurationEvolver{
			population:          make([]*Configuration, 100), // 100 configurations
			mutationRate:        0.05,
			crossoverRate:       0.7,
			elitismRate:         0.1,
			generationCount:     0,
			evolutionHistory:    []*Generation{},
			diversityMaintainer: NewDiversityMaintainer(),
			nicheFinder:         NewNicheFinder(),
		},
		anomalyLearner: &AnomalyLearner{
			anomalyDetector:    NewOnlineAnomalyDetector(),
			clusteringEngine:   NewClusteringEngine(),
			outlierDetector:    NewOutlierDetector(),
			anomalyDatabase:    NewAnomalyDatabase(),
			adaptiveThresholds: make(map[string]float64),
			learningWindow:     time.Hour * 24, // Learn over 24 hours
			falsePositiveRate:  0.01,
			detectionAccuracy:  0.95,
			newAnomalyTypes:    []*AnomalyType{},
		},
		workloadPredictor: &WorkloadPredictor{
			timeSeriesForecaster: NewTimeSeriesForecaster(),
			seasonalDecomposer:   NewSeasonalDecomposer(),
			trendAnalyzer:        NewTrendAnalyzer(),
			workloadClassifier:   NewWorkloadClassifier(),
			predictiveModels:     make(map[string]*PredictiveModel),
			predictionHorizon:    time.Hour, // Predict 1 hour ahead
			predictionAccuracy:   0.85,
		},
		adaptiveAlgorithms: &AdaptiveAlgorithms{
			algorithmLibrary:    make(map[string]ConsensusAlgorithm),
			performanceProfiles: make(map[string]*AlgorithmProfile),
			contextAnalyzer:     NewContextAnalyzer(),
			algorithmSelector:   NewAlgorithmSelector(),
			transitionManager:   NewTransitionManager(),
			hybridExecutor:      NewHybridExecutor(),
			currentAlgorithm:    "hybrid_default",
			algorithmHistory:    []*AlgorithmSwitch{},
		},
		knowledgeBase: &KnowledgeBase{
			facts:               make(map[string]*Fact),
			rules:               make(map[string]*Rule),
			heuristics:          make(map[string]*Heuristic),
			bestPractices:       make(map[string]*BestPractice),
			performanceModels:   make(map[string]*Model),
			causalRelationships: make(map[string]*CausalRelation),
			knowledgeGraph:      NewKnowledgeGraph(),
			reasoningEngine:     NewReasoningEngine(),
			lastUpdated:         time.Now(),
			version:             1,
		},
		experimentRunner:   NewExperimentRunner(),
		feedbackProcessor:  NewFeedbackProcessor(),
		improvementMetrics: NewImprovementMetrics(),
	}
}

// StartSelfImprovement begins the autonomous improvement process
func (sis *SelfImprovingSystem) StartSelfImprovement() error {
	fmt.Println("🧠 Starting Self-Improving System")
	fmt.Println("📈 System will autonomously optimize over time...")

	// Start learning loops
	go sis.experienceLearningLoop()
	go sis.performanceOptimizationLoop()
	go sis.configurationEvolutionLoop()
	go sis.anomalyLearningLoop()
	go sis.workloadPredictionLoop()
	go sis.experimentationLoop()
	go sis.knowledgeConsolidationLoop()

	// Start improvement monitoring
	go sis.monitorImprovements()

	return nil
}

// LearnFromExperience processes a consensus round and learns from it
func (sis *SelfImprovingSystem) LearnFromExperience(experience *ConsensusExperience) {
	sis.mu.Lock()
	defer sis.mu.Unlock()

	// Store experience
	sis.experienceLearner.experienceBuffer.Add(experience)
	atomic.AddInt64(&sis.experienceLearner.totalExperiences, 1)

	// Pattern recognition
	patterns := sis.experienceLearner.patternRecognizer.FindPatterns(experience)
	for _, pattern := range patterns {
		if experience.Successful {
			sis.recordSuccessPattern(pattern, experience)
		} else {
			sis.recordFailurePattern(pattern, experience)
		}
	}

	// Update outcome predictor
	sis.experienceLearner.outcomePredictor.Update(experience)

	// Update decision tree
	sis.experienceLearner.decisionTree.AddExample(experience)

	// Trigger experience replay for batch learning
	if sis.experienceLearner.totalExperiences%1000 == 0 {
		go sis.performExperienceReplay()
	}
}

// OptimizeAutomatically performs automatic performance optimization
func (sis *SelfImprovingSystem) OptimizeAutomatically() *OptimizationResult {
	sis.mu.Lock()
	defer sis.mu.Unlock()

	result := &OptimizationResult{
		Timestamp: time.Now(),
		Metrics:   make(map[string]float64),
	}

	// Get current performance metrics
	currentMetrics := sis.getCurrentMetrics()
	
	// Identify worst-performing metric
	worstMetric, worstValue := sis.identifyWorstMetric(currentMetrics)
	
	// Apply optimization based on metric type
	var improvement float64
	switch worstMetric {
	case "latency":
		improvement = sis.optimizeLatencyAutomatically()
	case "throughput":
		improvement = sis.optimizeThroughputAutomatically()
	case "error_rate":
		improvement = sis.optimizeReliabilityAutomatically()
	case "resource_usage":
		improvement = sis.optimizeResourcesAutomatically()
	}

	result.Metrics[worstMetric] = worstValue * (1 + improvement)
	result.Improvement = improvement

	// Record optimization attempt
	attempt := &OptimizationAttempt{
		timestamp:     time.Now(),
		targetMetric:  worstMetric,
		previousValue: worstValue,
		newValue:      result.Metrics[worstMetric],
		improvement:   improvement,
		successful:    improvement > sis.performanceOptimizer.improvementThreshold,
	}
	sis.performanceOptimizer.optimizationHistory = append(
		sis.performanceOptimizer.optimizationHistory, attempt)

	return result
}

// EvolveConfiguration evolves better configurations through genetic algorithms
func (sis *SelfImprovingSystem) EvolveConfiguration() *Configuration {
	sis.mu.Lock()
	defer sis.mu.Unlock()

	// Evaluate current population fitness
	for _, config := range sis.configurationEvolver.population {
		config.fitness = sis.evaluateConfiguration(config)
	}

	// Selection
	parents := sis.selectParents()

	// Crossover
	offspring := sis.performCrossover(parents)

	// Mutation
	sis.mutateOffspring(offspring)

	// Replace worst performers with offspring
	sis.replaceWorstPerformers(offspring)

	// Maintain diversity
	sis.configurationEvolver.diversityMaintainer.MaintainDiversity(
		sis.configurationEvolver.population)

	// Find niches for specialized scenarios
	niches := sis.configurationEvolver.nicheFinder.FindNiches(
		sis.configurationEvolver.population)
	
	// Update best configuration
	best := sis.findBestConfiguration()
	if best.fitness > sis.configurationEvolver.bestConfiguration.fitness {
		sis.configurationEvolver.bestConfiguration = best
		fmt.Printf("🧬 New best configuration found: fitness=%.4f\n", best.fitness)
	}

	sis.configurationEvolver.generationCount++

	return best
}

// PredictWorkload predicts future workload patterns
func (sis *SelfImprovingSystem) PredictWorkload() *WorkloadPrediction {
	sis.mu.RLock()
	defer sis.mu.RUnlock()

	prediction := &WorkloadPrediction{
		Timestamp:     time.Now(),
		Horizon:       sis.workloadPredictor.predictionHorizon,
		Predictions:   make(map[string]float64),
		Confidence:    make(map[string]float64),
		Seasonality:   make(map[string]float64),
		Trend:         make(map[string]float64),
	}

	// Time series forecasting
	forecast := sis.workloadPredictor.timeSeriesForecaster.Forecast(
		sis.workloadPredictor.predictionHorizon)
	prediction.Predictions["transaction_rate"] = forecast.Value
	prediction.Confidence["transaction_rate"] = forecast.Confidence

	// Seasonal decomposition
	seasonal := sis.workloadPredictor.seasonalDecomposer.Decompose()
	prediction.Seasonality["daily"] = seasonal.DailyPattern
	prediction.Seasonality["weekly"] = seasonal.WeeklyPattern

	// Trend analysis
	trend := sis.workloadPredictor.trendAnalyzer.AnalyzeTrend()
	prediction.Trend["growth_rate"] = trend.GrowthRate
	prediction.Trend["acceleration"] = trend.Acceleration

	// Workload classification
	workloadType := sis.workloadPredictor.workloadClassifier.Classify(forecast)
	prediction.WorkloadType = workloadType

	// Prepare system for predicted workload
	sis.prepareForWorkload(prediction)

	return prediction
}

// Implementation of learning loops

func (sis *SelfImprovingSystem) experienceLearningLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			sis.consolidateLearnedPatterns()
		}
	}
}

func (sis *SelfImprovingSystem) performanceOptimizationLoop() {
	ticker := time.NewTicker(time.Minute * 5)
	defer ticker.Stop()

	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			if sis.performanceOptimizer.autoTuning {
				result := sis.OptimizeAutomatically()
				if result.Improvement > 0 {
					fmt.Printf("⚡ Auto-optimization improved %s by %.1f%%\n", 
						result.OptimizedMetric, result.Improvement*100)
				}
			}
		}
	}
}

func (sis *SelfImprovingSystem) configurationEvolutionLoop() {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			best := sis.EvolveConfiguration()
			fmt.Printf("🧬 Configuration evolution: generation %d, best fitness %.4f\n",
				sis.configurationEvolver.generationCount, best.fitness)
		}
	}
}

func (sis *SelfImprovingSystem) anomalyLearningLoop() {
	ticker := time.NewTicker(time.Minute * 10)
	defer ticker.Stop()

	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			sis.updateAnomalyDetection()
		}
	}
}

func (sis *SelfImprovingSystem) workloadPredictionLoop() {
	ticker := time.NewTicker(time.Minute * 30)
	defer ticker.Stop()

	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			prediction := sis.PredictWorkload()
			fmt.Printf("📊 Workload prediction: %s pattern expected, confidence %.1f%%\n",
				prediction.WorkloadType, prediction.AverageConfidence()*100)
		}
	}
}

func (sis *SelfImprovingSystem) experimentationLoop() {
	ticker := time.NewTicker(time.Hour * 6) // Run experiments every 6 hours
	defer ticker.Stop()

	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			sis.runControlledExperiment()
		}
	}
}

func (sis *SelfImprovingSystem) knowledgeConsolidationLoop() {
	ticker := time.NewTicker(time.Hour * 24) // Daily consolidation
	defer ticker.Stop()

	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			sis.consolidateKnowledge()
		}
	}
}

func (sis *SelfImprovingSystem) monitorImprovements() {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()

	baseline := sis.establishBaseline()
	
	for {
		select {
		case <-sis.ctx.Done():
			return
		case <-ticker.C:
			current := sis.getCurrentMetrics()
			improvement := sis.calculateImprovement(baseline, current)
			
			fmt.Printf("📈 Self-improvement metrics:\n")
			fmt.Printf("  • Performance: +%.1f%%\n", improvement.Performance*100)
			fmt.Printf("  • Reliability: +%.1f%%\n", improvement.Reliability*100)
			fmt.Printf("  • Efficiency: +%.1f%%\n", improvement.Efficiency*100)
			fmt.Printf("  • Adaptability: +%.1f%%\n", improvement.Adaptability*100)
			
			// Update baseline periodically
			if improvement.Overall() > 0.1 { // 10% overall improvement
				baseline = current
				fmt.Println("  ✨ New performance baseline established!")
			}
		}
	}
}

// Helper methods for self-improvement

func (sis *SelfImprovingSystem) recordSuccessPattern(pattern Pattern, experience *ConsensusExperience) {
	patternID := pattern.Hash()
	
	if existing, ok := sis.experienceLearner.successfulPatterns[patternID]; ok {
		// Update existing pattern
		existing.occurrences++
		existing.successRate = (existing.successRate*float64(existing.occurrences-1) + 1.0) / 
			float64(existing.occurrences)
		existing.lastSeen = time.Now()
		existing.confidence = math.Min(0.99, existing.confidence*1.01) // Increase confidence
	} else {
		// Create new success pattern
		sis.experienceLearner.successfulPatterns[patternID] = &SuccessPattern{
			patternID:   patternID,
			conditions:  pattern.Conditions,
			actions:     pattern.Actions,
			outcomes:    []Outcome{experience.Outcome},
			successRate: 1.0,
			occurrences: 1,
			lastSeen:    time.Now(),
			confidence:  0.5, // Start with medium confidence
		}
	}
}

func (sis *SelfImprovingSystem) performExperienceReplay() {
	// Sample random batch of experiences
	batch := sis.experienceLearner.experienceBuffer.SampleBatch(1000)
	
	// Re-learn from past experiences
	for _, exp := range batch {
		// Update models with hindsight
		sis.experienceLearner.outcomePredictor.UpdateWithHindsight(exp)
		sis.experienceLearner.decisionTree.RefineWithExperience(exp)
	}
	
	fmt.Printf("🔄 Experience replay completed: re-learned from %d experiences\n", len(batch))
}

func (sis *SelfImprovingSystem) optimizeLatencyAutomatically() float64 {
	optimizer := sis.performanceOptimizer.gradientOptimizer
	
	// Define objective function (minimize latency)
	objective := func(params []float64) float64 {
		// Apply parameters and measure latency
		latency := sis.measureLatencyWithParams(params)
		return -latency // Negative because we minimize
	}
	
	// Run gradient optimization
	currentParams := sis.getCurrentParameters()
	optimizedParams := optimizer.Optimize(objective, currentParams)
	
	// Apply optimized parameters
	sis.applyParameters(optimizedParams)
	
	// Measure improvement
	oldLatency := sis.measureLatencyWithParams(currentParams)
	newLatency := sis.measureLatencyWithParams(optimizedParams)
	
	return (oldLatency - newLatency) / oldLatency
}

func (sis *SelfImprovingSystem) evaluateConfiguration(config *Configuration) float64 {
	// Multi-objective fitness function
	fitness := 0.0
	
	// Performance component
	perf := sis.measurePerformanceWithConfig(config)
	fitness += perf * 0.4
	
	// Reliability component
	reliability := sis.measureReliabilityWithConfig(config)
	fitness += reliability * 0.3
	
	// Efficiency component
	efficiency := sis.measureEfficiencyWithConfig(config)
	fitness += efficiency * 0.2
	
	// Adaptability component
	adaptability := sis.measureAdaptabilityWithConfig(config)
	fitness += adaptability * 0.1
	
	return fitness
}

func (sis *SelfImprovingSystem) consolidateKnowledge() {
	sis.mu.Lock()
	defer sis.mu.Unlock()
	
	// Extract rules from patterns
	newRules := sis.extractRulesFromPatterns()
	for id, rule := range newRules {
		sis.knowledgeBase.rules[id] = rule
	}
	
	// Update causal relationships
	causality := sis.inferCausalRelationships()
	for id, relation := range causality {
		sis.knowledgeBase.causalRelationships[id] = relation
	}
	
	// Derive heuristics from experience
	heuristics := sis.deriveHeuristics()
	for id, heuristic := range heuristics {
		sis.knowledgeBase.heuristics[id] = heuristic
	}
	
	// Update knowledge graph
	sis.knowledgeBase.knowledgeGraph.Update(
		sis.knowledgeBase.facts,
		sis.knowledgeBase.rules,
		sis.knowledgeBase.causalRelationships,
	)
	
	// Increment version
	sis.knowledgeBase.version++
	sis.knowledgeBase.lastUpdated = time.Now()
	
	fmt.Printf("📚 Knowledge consolidated: %d rules, %d heuristics, %d causal relations\n",
		len(sis.knowledgeBase.rules),
		len(sis.knowledgeBase.heuristics),
		len(sis.knowledgeBase.causalRelationships))
}

// Supporting structures for self-improvement
type ConsensusExperience struct {
	ID          string
	Timestamp   time.Time
	Context     map[string]interface{}
	Actions     []Action
	Outcome     Outcome
	Successful  bool
	Metrics     map[string]float64
	Anomalies   []Anomaly
}

type WorkloadPrediction struct {
	Timestamp    time.Time
	Horizon      time.Duration
	Predictions  map[string]float64
	Confidence   map[string]float64
	Seasonality  map[string]float64
	Trend        map[string]float64
	WorkloadType string
}

func (wp *WorkloadPrediction) AverageConfidence() float64 {
	if len(wp.Confidence) == 0 {
		return 0
	}
	
	sum := 0.0
	for _, conf := range wp.Confidence {
		sum += conf
	}
	return sum / float64(len(wp.Confidence))
}

type ImprovementMetrics struct {
	Performance  float64
	Reliability  float64
	Efficiency   float64
	Adaptability float64
}

func (im *ImprovementMetrics) Overall() float64 {
	return (im.Performance + im.Reliability + im.Efficiency + im.Adaptability) / 4.0
}

// This self-improving system makes the consensus genuinely better over time
// Through continuous learning, evolution, and adaptation