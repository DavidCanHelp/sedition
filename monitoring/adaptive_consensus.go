package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// AdaptiveConsensusMonitor implements real-time performance monitoring and adaptive consensus parameters
type AdaptiveConsensusMonitor struct {
	mu sync.RWMutex

	// Performance metrics
	metrics           *PerformanceMetrics
	metricsHistory    []MetricsSnapshot
	adaptationRules   []*AdaptationRule
	
	// Current system state
	currentParameters *ConsensusParameters
	baseParameters    *ConsensusParameters
	adaptationEnabled bool
	
	// Monitoring configuration
	monitoringInterval time.Duration
	historyWindow      time.Duration
	adaptationCooldown time.Duration
	lastAdaptation     time.Time
	
	// Alert system
	alertThresholds   map[string]*AlertThreshold
	activeAlerts      map[string]*Alert
	alertCallbacks    []AlertCallback
	
	// Machine learning optimization
	optimizer         *ConsensusOptimizer
	adaptationLog     []AdaptationEvent
	
	// Network health tracking
	networkHealth     *NetworkHealthTracker
	nodeStates        map[string]*NodeState
	
	// Stop channel for graceful shutdown
	stopCh            chan struct{}
	running           bool
}

// PerformanceMetrics tracks real-time consensus performance
type PerformanceMetrics struct {
	// Throughput metrics
	TPS                  float64   `json:"transactions_per_second"`
	BlocksPerMinute      float64   `json:"blocks_per_minute"`
	CommitsPerBlock      float64   `json:"commits_per_block"`
	
	// Latency metrics
	BlockLatency         time.Duration `json:"block_latency"`
	ConsensusLatency     time.Duration `json:"consensus_latency"`
	PropagationLatency   time.Duration `json:"propagation_latency"`
	FinalizationLatency  time.Duration `json:"finalization_latency"`
	
	// Quality metrics
	AverageQuality       float64   `json:"average_quality_score"`
	QualityVariance      float64   `json:"quality_variance"`
	MLAccuracy           float64   `json:"ml_accuracy"`
	SecurityScore        float64   `json:"security_score"`
	
	// Network metrics
	NetworkUtilization   float64   `json:"network_utilization"`
	ValidatorParticipation float64 `json:"validator_participation"`
	ByzantineNodes       int       `json:"byzantine_nodes"`
	NetworkPartitions    int       `json:"network_partitions"`
	
	// Resource metrics
	CPUUsage             float64   `json:"cpu_usage"`
	MemoryUsage          float64   `json:"memory_usage"`
	DiskUsage            float64   `json:"disk_usage"`
	NetworkBandwidth     float64   `json:"network_bandwidth"`
	
	// Consensus health
	ForkCount            int       `json:"fork_count"`
	OrphanBlocks         int       `json:"orphan_blocks"`
	RoundDuration        time.Duration `json:"round_duration"`
	CommitteeEfficiency  float64   `json:"committee_efficiency"`
	
	Timestamp            time.Time `json:"timestamp"`
}

// MetricsSnapshot captures metrics at a specific point in time
type MetricsSnapshot struct {
	Metrics   PerformanceMetrics `json:"metrics"`
	Timestamp time.Time          `json:"timestamp"`
}

// ConsensusParameters represents adaptive consensus configuration
type ConsensusParameters struct {
	BlockTime            time.Duration `json:"block_time"`
	CommitteeSize        int           `json:"committee_size"`
	MinStakeRequired     int64         `json:"min_stake_required"`
	QualityThreshold     float64       `json:"quality_threshold"`
	ReputationWeight     float64       `json:"reputation_weight"`
	VRFDifficulty        int           `json:"vrf_difficulty"`
	MaxBlockSize         int           `json:"max_block_size"`
	EpochLength          int64         `json:"epoch_length"`
	SlashingPenalty      float64       `json:"slashing_penalty"`
	RewardDistribution   []float64     `json:"reward_distribution"`
}

// AdaptationRule defines rules for parameter adaptation
type AdaptationRule struct {
	Name              string                 `json:"name"`
	TriggerCondition  string                 `json:"trigger_condition"`
	TargetParameter   string                 `json:"target_parameter"`
	AdaptationFunction func(current float64, metrics *PerformanceMetrics) float64
	MinValue          float64                `json:"min_value"`
	MaxValue          float64                `json:"max_value"`
	AdaptationRate    float64                `json:"adaptation_rate"`
	Enabled           bool                   `json:"enabled"`
	Priority          int                    `json:"priority"`
}

// AlertThreshold defines performance alert criteria
type AlertThreshold struct {
	MetricName   string        `json:"metric_name"`
	MinValue     *float64      `json:"min_value,omitempty"`
	MaxValue     *float64      `json:"max_value,omitempty"`
	Duration     time.Duration `json:"duration"`
	Severity     AlertSeverity `json:"severity"`
	Description  string        `json:"description"`
}

// Alert represents an active performance alert
type Alert struct {
	ID          string        `json:"id"`
	Threshold   *AlertThreshold `json:"threshold"`
	TriggeredAt time.Time     `json:"triggered_at"`
	Value       float64       `json:"value"`
	Severity    AlertSeverity `json:"severity"`
	Message     string        `json:"message"`
	Resolved    bool          `json:"resolved"`
	ResolvedAt  *time.Time    `json:"resolved_at,omitempty"`
}

type AlertSeverity int

const (
	AlertSeverityInfo AlertSeverity = iota
	AlertSeverityWarning
	AlertSeverityCritical
	AlertSeverityEmergency
)

// AlertCallback function type for alert notifications
type AlertCallback func(alert *Alert)

// AdaptationEvent records parameter adaptations
type AdaptationEvent struct {
	Timestamp     time.Time              `json:"timestamp"`
	Rule          string                 `json:"rule"`
	Parameter     string                 `json:"parameter"`
	OldValue      float64                `json:"old_value"`
	NewValue      float64                `json:"new_value"`
	TriggerMetric string                 `json:"trigger_metric"`
	MetricValue   float64                `json:"metric_value"`
	Reason        string                 `json:"reason"`
}

// ConsensusOptimizer uses machine learning to optimize consensus parameters
type ConsensusOptimizer struct {
	model            *OptimizationModel
	trainingData     []OptimizationSample
	predictionCache  map[string]OptimizationResult
	learningRate     float64
	optimizationGoal OptimizationGoal
}

// OptimizationModel represents the ML model for parameter optimization
type OptimizationModel struct {
	Weights          [][]float64 `json:"weights"`
	Biases           []float64   `json:"biases"`
	Architecture     []int       `json:"architecture"`
	TrainingMetrics  map[string]float64 `json:"training_metrics"`
	LastUpdated      time.Time   `json:"last_updated"`
}

// OptimizationSample represents training data for the optimizer
type OptimizationSample struct {
	Parameters  ConsensusParameters `json:"parameters"`
	Metrics     PerformanceMetrics  `json:"metrics"`
	Objective   float64             `json:"objective"`
	Timestamp   time.Time           `json:"timestamp"`
}

// OptimizationResult represents optimization recommendations
type OptimizationResult struct {
	RecommendedParameters ConsensusParameters `json:"recommended_parameters"`
	ExpectedImprovement   float64             `json:"expected_improvement"`
	Confidence            float64             `json:"confidence"`
	Reasoning             []string            `json:"reasoning"`
}

type OptimizationGoal int

const (
	OptimizeLatency OptimizationGoal = iota
	OptimizeThroughput
	OptimizeQuality
	OptimizeBalance
)

// NetworkHealthTracker monitors network-wide health metrics
type NetworkHealthTracker struct {
	nodes              map[string]*NodeState
	networkTopology    *NetworkTopology
	partitionDetector  *PartitionDetector
	healingStrategies  []HealingStrategy
}

// NodeState represents the state of a validator node
type NodeState struct {
	NodeID           string        `json:"node_id"`
	LastSeen         time.Time     `json:"last_seen"`
	Status           NodeStatus    `json:"status"`
	PerformanceScore float64       `json:"performance_score"`
	ReputationScore  float64       `json:"reputation_score"`
	Latency          time.Duration `json:"latency"`
	NetworkPosition  NetworkPosition `json:"network_position"`
	FaultHistory     []FaultEvent  `json:"fault_history"`
}

type NodeStatus int

const (
	NodeStatusOnline NodeStatus = iota
	NodeStatusOffline
	NodeStatusDegraded
	NodeStatusSuspected
	NodeStatusByzantine
)

type NetworkPosition struct {
	RegionID    string  `json:"region_id"`
	Coordinates []float64 `json:"coordinates"`
	Connections []string `json:"connections"`
}

type FaultEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	FaultType   string    `json:"fault_type"`
	Severity    int       `json:"severity"`
	Description string    `json:"description"`
}

// NetworkTopology represents the network structure
type NetworkTopology struct {
	Nodes       map[string]*NodeState
	Connections map[string][]string
	Regions     map[string]*Region
}

type Region struct {
	ID              string            `json:"id"`
	Name            string            `json:"name"`
	NodeCount       int               `json:"node_count"`
	AverageLatency  time.Duration     `json:"average_latency"`
	ConnectedRegions []string          `json:"connected_regions"`
}

// PartitionDetector identifies network partitions
type PartitionDetector struct {
	partitionHistory []PartitionEvent
	detectionConfig  *PartitionDetectionConfig
}

type PartitionEvent struct {
	Timestamp     time.Time `json:"timestamp"`
	PartitionType string    `json:"partition_type"`
	AffectedNodes []string  `json:"affected_nodes"`
	Duration      time.Duration `json:"duration"`
	Resolved      bool      `json:"resolved"`
}

type PartitionDetectionConfig struct {
	TimeoutThreshold      time.Duration `json:"timeout_threshold"`
	MinPartitionSize      int           `json:"min_partition_size"`
	ConsensusRequirement  float64       `json:"consensus_requirement"`
}

// HealingStrategy defines network healing approaches
type HealingStrategy struct {
	Name          string                 `json:"name"`
	Trigger       PartitionTrigger       `json:"trigger"`
	Action        func(*PartitionEvent) error
	Priority      int                    `json:"priority"`
	Enabled       bool                   `json:"enabled"`
}

type PartitionTrigger struct {
	PartitionType     string        `json:"partition_type"`
	MinDuration       time.Duration `json:"min_duration"`
	AffectedNodeRatio float64       `json:"affected_node_ratio"`
}

// NewAdaptiveConsensusMonitor creates a new adaptive consensus monitor
func NewAdaptiveConsensusMonitor(baseParams *ConsensusParameters) *AdaptiveConsensusMonitor {
	monitor := &AdaptiveConsensusMonitor{
		metrics:            &PerformanceMetrics{},
		metricsHistory:     make([]MetricsSnapshot, 0),
		adaptationRules:    make([]*AdaptationRule, 0),
		currentParameters:  baseParams,
		baseParameters:     copyParameters(baseParams),
		adaptationEnabled:  true,
		monitoringInterval: 5 * time.Second,
		historyWindow:      1 * time.Hour,
		adaptationCooldown: 30 * time.Second,
		alertThresholds:    make(map[string]*AlertThreshold),
		activeAlerts:       make(map[string]*Alert),
		alertCallbacks:     make([]AlertCallback, 0),
		adaptationLog:      make([]AdaptationEvent, 0),
		nodeStates:         make(map[string]*NodeState),
		stopCh:            make(chan struct{}),
	}

	// Initialize default adaptation rules
	monitor.initializeDefaultRules()
	
	// Initialize default alert thresholds
	monitor.initializeDefaultAlerts()
	
	// Initialize optimizer
	monitor.optimizer = NewConsensusOptimizer()
	
	// Initialize network health tracker
	monitor.networkHealth = NewNetworkHealthTracker()

	return monitor
}

// Start begins the adaptive monitoring process
func (acm *AdaptiveConsensusMonitor) Start(ctx context.Context) error {
	acm.mu.Lock()
	if acm.running {
		acm.mu.Unlock()
		return fmt.Errorf("monitor is already running")
	}
	acm.running = true
	acm.mu.Unlock()

	// Start monitoring goroutine
	go acm.monitoringLoop(ctx)
	
	// Start adaptation goroutine
	go acm.adaptationLoop(ctx)
	
	// Start alert processing goroutine
	go acm.alertProcessingLoop(ctx)
	
	// Start network health monitoring
	go acm.networkHealthLoop(ctx)

	return nil
}

// Stop gracefully shuts down the monitor
func (acm *AdaptiveConsensusMonitor) Stop() {
	acm.mu.Lock()
	defer acm.mu.Unlock()
	
	if !acm.running {
		return
	}
	
	close(acm.stopCh)
	acm.running = false
}

// monitoringLoop continuously collects performance metrics
func (acm *AdaptiveConsensusMonitor) monitoringLoop(ctx context.Context) {
	ticker := time.NewTicker(acm.monitoringInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-acm.stopCh:
			return
		case <-ticker.C:
			acm.collectMetrics()
			acm.updateMetricsHistory()
			acm.checkAlertThresholds()
		}
	}
}

// collectMetrics gathers current performance data
func (acm *AdaptiveConsensusMonitor) collectMetrics() {
	acm.mu.Lock()
	defer acm.mu.Unlock()

	// Collect system metrics (simplified implementation)
	now := time.Now()
	
	// Update metrics (in real implementation, these would come from system monitoring)
	acm.metrics.Timestamp = now
	
	// Simulate metrics collection
	acm.metrics.TPS = acm.calculateTPS()
	acm.metrics.BlocksPerMinute = acm.calculateBlocksPerMinute()
	acm.metrics.BlockLatency = acm.calculateBlockLatency()
	acm.metrics.ConsensusLatency = acm.calculateConsensusLatency()
	acm.metrics.AverageQuality = acm.calculateAverageQuality()
	acm.metrics.ValidatorParticipation = acm.calculateValidatorParticipation()
	acm.metrics.NetworkUtilization = acm.calculateNetworkUtilization()
	acm.metrics.CPUUsage = acm.getCPUUsage()
	acm.metrics.MemoryUsage = acm.getMemoryUsage()
	acm.metrics.CommitteeEfficiency = acm.calculateCommitteeEfficiency()
}

// adaptationLoop handles parameter adaptation based on performance
func (acm *AdaptiveConsensusMonitor) adaptationLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second) // Check for adaptations every 10 seconds
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-acm.stopCh:
			return
		case <-ticker.C:
			if acm.adaptationEnabled && acm.shouldAdapt() {
				acm.performAdaptation()
			}
		}
	}
}

// shouldAdapt determines if parameters should be adapted
func (acm *AdaptiveConsensusMonitor) shouldAdapt() bool {
	acm.mu.RLock()
	defer acm.mu.RUnlock()
	
	// Check cooldown period
	if time.Since(acm.lastAdaptation) < acm.adaptationCooldown {
		return false
	}
	
	// Check if any adaptation rules are triggered
	for _, rule := range acm.adaptationRules {
		if rule.Enabled && acm.isRuleTriggered(rule) {
			return true
		}
	}
	
	return false
}

// performAdaptation adapts consensus parameters based on current performance
func (acm *AdaptiveConsensusMonitor) performAdaptation() {
	acm.mu.Lock()
	defer acm.mu.Unlock()
	
	// Sort rules by priority
	rules := make([]*AdaptationRule, len(acm.adaptationRules))
	copy(rules, acm.adaptationRules)
	
	// Apply triggered rules
	for _, rule := range rules {
		if rule.Enabled && acm.isRuleTriggered(rule) {
			oldValue := acm.getParameterValue(rule.TargetParameter)
			newValue := rule.AdaptationFunction(oldValue, acm.metrics)
			
			// Clamp to bounds
			newValue = math.Max(rule.MinValue, math.Min(rule.MaxValue, newValue))
			
			if math.Abs(newValue-oldValue) > 0.001 { // Avoid tiny changes
				acm.setParameterValue(rule.TargetParameter, newValue)
				
				// Log adaptation
				event := AdaptationEvent{
					Timestamp:     time.Now(),
					Rule:          rule.Name,
					Parameter:     rule.TargetParameter,
					OldValue:      oldValue,
					NewValue:      newValue,
					TriggerMetric: rule.TriggerCondition,
					MetricValue:   acm.getMetricValue(rule.TriggerCondition),
					Reason:        fmt.Sprintf("Rule '%s' triggered", rule.Name),
				}
				acm.adaptationLog = append(acm.adaptationLog, event)
				
				acm.lastAdaptation = time.Now()
				
				// Notify about adaptation
				fmt.Printf("Adapted %s from %.3f to %.3f (rule: %s)\n", 
					rule.TargetParameter, oldValue, newValue, rule.Name)
			}
		}
	}
}

// ML-based optimization methods
func (acm *AdaptiveConsensusMonitor) OptimizeParameters() *OptimizationResult {
	acm.mu.RLock()
	defer acm.mu.RUnlock()
	
	if acm.optimizer == nil {
		return nil
	}
	
	// Use current metrics and historical data to optimize
	return acm.optimizer.Optimize(acm.currentParameters, acm.metrics, acm.metricsHistory)
}

// Alert processing methods
func (acm *AdaptiveConsensusMonitor) alertProcessingLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-acm.stopCh:
			return
		case <-ticker.C:
			acm.processAlerts()
		}
	}
}

func (acm *AdaptiveConsensusMonitor) processAlerts() {
	acm.mu.Lock()
	defer acm.mu.Unlock()
	
	// Check for resolved alerts
	for id, alert := range acm.activeAlerts {
		if !alert.Resolved && acm.isAlertResolved(alert) {
			alert.Resolved = true
			now := time.Now()
			alert.ResolvedAt = &now
			
			// Notify callbacks
			for _, callback := range acm.alertCallbacks {
				go callback(alert)
			}
			
			fmt.Printf("Alert resolved: %s\n", alert.Message)
			delete(acm.activeAlerts, id)
		}
	}
}

// Network health monitoring
func (acm *AdaptiveConsensusMonitor) networkHealthLoop(ctx context.Context) {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-acm.stopCh:
			return
		case <-ticker.C:
			acm.updateNetworkHealth()
		}
	}
}

func (acm *AdaptiveConsensusMonitor) updateNetworkHealth() {
	if acm.networkHealth == nil {
		return
	}
	
	acm.mu.Lock()
	defer acm.mu.Unlock()
	
	// Update node states
	for nodeID, node := range acm.nodeStates {
		// Check node health (simplified)
		if time.Since(node.LastSeen) > 30*time.Second {
			if node.Status == NodeStatusOnline {
				node.Status = NodeStatusOffline
				fmt.Printf("Node %s marked as offline\n", nodeID)
			}
		}
	}
	
	// Detect network partitions
	acm.networkHealth.DetectPartitions()
}

// Utility methods for parameter access
func (acm *AdaptiveConsensusMonitor) getParameterValue(paramName string) float64 {
	switch paramName {
	case "block_time":
		return float64(acm.currentParameters.BlockTime.Milliseconds())
	case "committee_size":
		return float64(acm.currentParameters.CommitteeSize)
	case "quality_threshold":
		return acm.currentParameters.QualityThreshold
	case "reputation_weight":
		return acm.currentParameters.ReputationWeight
	default:
		return 0
	}
}

func (acm *AdaptiveConsensusMonitor) setParameterValue(paramName string, value float64) {
	switch paramName {
	case "block_time":
		acm.currentParameters.BlockTime = time.Duration(value) * time.Millisecond
	case "committee_size":
		acm.currentParameters.CommitteeSize = int(value)
	case "quality_threshold":
		acm.currentParameters.QualityThreshold = value
	case "reputation_weight":
		acm.currentParameters.ReputationWeight = value
	}
}

// Helper methods for metrics calculation (simplified implementations)
func (acm *AdaptiveConsensusMonitor) calculateTPS() float64 {
	// Simplified TPS calculation
	return 100.0 + float64(time.Now().Unix()%50)
}

func (acm *AdaptiveConsensusMonitor) calculateBlocksPerMinute() float64 {
	blockTimeSeconds := acm.currentParameters.BlockTime.Seconds()
	return 60.0 / blockTimeSeconds
}

func (acm *AdaptiveConsensusMonitor) calculateBlockLatency() time.Duration {
	return acm.currentParameters.BlockTime + time.Duration(50+time.Now().Unix()%100)*time.Millisecond
}

func (acm *AdaptiveConsensusMonitor) calculateConsensusLatency() time.Duration {
	return time.Duration(200+time.Now().Unix()%300) * time.Millisecond
}

func (acm *AdaptiveConsensusMonitor) calculateAverageQuality() float64 {
	return 0.75 + 0.2*math.Sin(float64(time.Now().Unix())/100.0)
}

func (acm *AdaptiveConsensusMonitor) calculateValidatorParticipation() float64 {
	return 0.85 + 0.1*math.Cos(float64(time.Now().Unix())/50.0)
}

func (acm *AdaptiveConsensusMonitor) calculateNetworkUtilization() float64 {
	return 0.6 + 0.3*math.Sin(float64(time.Now().Unix())/200.0)
}

func (acm *AdaptiveConsensusMonitor) getCPUUsage() float64 {
	return 0.3 + 0.2*math.Sin(float64(time.Now().Unix())/150.0)
}

func (acm *AdaptiveConsensusMonitor) getMemoryUsage() float64 {
	return 0.4 + 0.1*math.Sin(float64(time.Now().Unix())/300.0)
}

func (acm *AdaptiveConsensusMonitor) calculateCommitteeEfficiency() float64 {
	return 0.9 + 0.08*math.Cos(float64(time.Now().Unix())/100.0)
}

// Initialization methods
func (acm *AdaptiveConsensusMonitor) initializeDefaultRules() {
	// Rule 1: Adapt block time based on latency
	acm.adaptationRules = append(acm.adaptationRules, &AdaptationRule{
		Name:             "LatencyBasedBlockTime",
		TriggerCondition: "block_latency",
		TargetParameter:  "block_time",
		AdaptationFunction: func(current float64, metrics *PerformanceMetrics) float64 {
			latencyMs := float64(metrics.BlockLatency.Milliseconds())
			if latencyMs > current*1.5 {
				return current * 1.1 // Increase block time by 10%
			} else if latencyMs < current*0.7 {
				return current * 0.95 // Decrease block time by 5%
			}
			return current
		},
		MinValue:       1000.0, // 1 second
		MaxValue:       30000.0, // 30 seconds
		AdaptationRate: 0.1,
		Enabled:        true,
		Priority:       1,
	})

	// Rule 2: Adapt committee size based on participation
	acm.adaptationRules = append(acm.adaptationRules, &AdaptationRule{
		Name:             "ParticipationBasedCommittee",
		TriggerCondition: "validator_participation",
		TargetParameter:  "committee_size",
		AdaptationFunction: func(current float64, metrics *PerformanceMetrics) float64 {
			if metrics.ValidatorParticipation < 0.7 {
				return current * 0.9 // Reduce committee size
			} else if metrics.ValidatorParticipation > 0.95 {
				return current * 1.05 // Increase committee size
			}
			return current
		},
		MinValue:       5.0,
		MaxValue:       50.0,
		AdaptationRate: 0.05,
		Enabled:        true,
		Priority:       2,
	})

	// Rule 3: Adapt quality threshold based on average quality
	acm.adaptationRules = append(acm.adaptationRules, &AdaptationRule{
		Name:             "QualityAdaptiveThreshold",
		TriggerCondition: "average_quality",
		TargetParameter:  "quality_threshold",
		AdaptationFunction: func(current float64, metrics *PerformanceMetrics) float64 {
			if metrics.AverageQuality > current+0.1 {
				return current + 0.02 // Raise standards gradually
			} else if metrics.AverageQuality < current-0.2 {
				return current - 0.05 // Lower standards if quality drops
			}
			return current
		},
		MinValue:       0.3,
		MaxValue:       0.95,
		AdaptationRate: 0.02,
		Enabled:        true,
		Priority:       3,
	})
}

func (acm *AdaptiveConsensusMonitor) initializeDefaultAlerts() {
	// Low TPS alert
	lowTPS := 50.0
	acm.alertThresholds["low_tps"] = &AlertThreshold{
		MetricName:  "transactions_per_second",
		MinValue:    &lowTPS,
		Duration:    30 * time.Second,
		Severity:    AlertSeverityWarning,
		Description: "Transaction throughput is below minimum threshold",
	}

	// High latency alert
	highLatency := 10000.0 // 10 seconds in milliseconds
	acm.alertThresholds["high_latency"] = &AlertThreshold{
		MetricName:  "block_latency",
		MaxValue:    &highLatency,
		Duration:    60 * time.Second,
		Severity:    AlertSeverityCritical,
		Description: "Block latency is unacceptably high",
	}

	// Low participation alert
	lowParticipation := 0.6
	acm.alertThresholds["low_participation"] = &AlertThreshold{
		MetricName:  "validator_participation",
		MinValue:    &lowParticipation,
		Duration:    120 * time.Second,
		Severity:    AlertSeverityWarning,
		Description: "Validator participation is below safe threshold",
	}
}

// Additional utility functions
func copyParameters(params *ConsensusParameters) *ConsensusParameters {
	paramsCopy := *params
	if params.RewardDistribution != nil {
		paramsCopy.RewardDistribution = make([]float64, len(params.RewardDistribution))
		copy(paramsCopy.RewardDistribution, params.RewardDistribution)
	}
	return &paramsCopy
}

func (acm *AdaptiveConsensusMonitor) updateMetricsHistory() {
	acm.mu.Lock()
	defer acm.mu.Unlock()
	
	snapshot := MetricsSnapshot{
		Metrics:   *acm.metrics,
		Timestamp: time.Now(),
	}
	
	acm.metricsHistory = append(acm.metricsHistory, snapshot)
	
	// Limit history size
	cutoff := time.Now().Add(-acm.historyWindow)
	var filtered []MetricsSnapshot
	for _, snap := range acm.metricsHistory {
		if snap.Timestamp.After(cutoff) {
			filtered = append(filtered, snap)
		}
	}
	acm.metricsHistory = filtered
}

func (acm *AdaptiveConsensusMonitor) checkAlertThresholds() {
	for id, threshold := range acm.alertThresholds {
		value := acm.getMetricValue(threshold.MetricName)
		
		shouldAlert := false
		if threshold.MinValue != nil && value < *threshold.MinValue {
			shouldAlert = true
		}
		if threshold.MaxValue != nil && value > *threshold.MaxValue {
			shouldAlert = true
		}
		
		if shouldAlert && acm.activeAlerts[id] == nil {
			alert := &Alert{
				ID:          id,
				Threshold:   threshold,
				TriggeredAt: time.Now(),
				Value:       value,
				Severity:    threshold.Severity,
				Message:     fmt.Sprintf("%s: %.2f", threshold.Description, value),
			}
			
			acm.activeAlerts[id] = alert
			
			// Notify callbacks
			for _, callback := range acm.alertCallbacks {
				go callback(alert)
			}
		}
	}
}

func (acm *AdaptiveConsensusMonitor) getMetricValue(metricName string) float64 {
	switch metricName {
	case "transactions_per_second":
		return acm.metrics.TPS
	case "block_latency":
		return float64(acm.metrics.BlockLatency.Milliseconds())
	case "validator_participation":
		return acm.metrics.ValidatorParticipation
	case "average_quality":
		return acm.metrics.AverageQuality
	default:
		return 0
	}
}

func (acm *AdaptiveConsensusMonitor) isRuleTriggered(rule *AdaptationRule) bool {
	// Simple rule evaluation (in practice, this would be more sophisticated)
	_ = acm.getMetricValue(rule.TriggerCondition)
	currentParam := acm.getParameterValue(rule.TargetParameter)
	
	// Calculate if adaptation would make a significant change
	newValue := rule.AdaptationFunction(currentParam, acm.metrics)
	return math.Abs(newValue-currentParam) > 0.01
}

func (acm *AdaptiveConsensusMonitor) isAlertResolved(alert *Alert) bool {
	value := acm.getMetricValue(alert.Threshold.MetricName)
	
	if alert.Threshold.MinValue != nil && value >= *alert.Threshold.MinValue {
		return true
	}
	if alert.Threshold.MaxValue != nil && value <= *alert.Threshold.MaxValue {
		return true
	}
	return false
}

// Public API methods
func (acm *AdaptiveConsensusMonitor) GetCurrentMetrics() *PerformanceMetrics {
	acm.mu.RLock()
	defer acm.mu.RUnlock()
	
	copy := *acm.metrics
	return &copy
}

func (acm *AdaptiveConsensusMonitor) GetCurrentParameters() *ConsensusParameters {
	acm.mu.RLock()
	defer acm.mu.RUnlock()
	
	return copyParameters(acm.currentParameters)
}

func (acm *AdaptiveConsensusMonitor) GetAdaptationHistory() []AdaptationEvent {
	acm.mu.RLock()
	defer acm.mu.RUnlock()
	
	history := make([]AdaptationEvent, len(acm.adaptationLog))
	copy(history, acm.adaptationLog)
	return history
}

func (acm *AdaptiveConsensusMonitor) GetActiveAlerts() []*Alert {
	acm.mu.RLock()
	defer acm.mu.RUnlock()
	
	alerts := make([]*Alert, 0, len(acm.activeAlerts))
	for _, alert := range acm.activeAlerts {
		alerts = append(alerts, alert)
	}
	return alerts
}

func (acm *AdaptiveConsensusMonitor) RegisterAlertCallback(callback AlertCallback) {
	acm.mu.Lock()
	defer acm.mu.Unlock()
	
	acm.alertCallbacks = append(acm.alertCallbacks, callback)
}

func (acm *AdaptiveConsensusMonitor) ExportMetrics() ([]byte, error) {
	acm.mu.RLock()
	defer acm.mu.RUnlock()
	
	export := struct {
		Metrics           *PerformanceMetrics  `json:"current_metrics"`
		Parameters        *ConsensusParameters `json:"current_parameters"`
		AdaptationHistory []AdaptationEvent    `json:"adaptation_history"`
		ActiveAlerts      []*Alert             `json:"active_alerts"`
		Timestamp         time.Time            `json:"timestamp"`
	}{
		Metrics:           acm.metrics,
		Parameters:        acm.currentParameters,
		AdaptationHistory: acm.adaptationLog,
		ActiveAlerts:      acm.GetActiveAlerts(),
		Timestamp:         time.Now(),
	}
	
	return json.MarshalIndent(export, "", "  ")
}

// Placeholder implementations for referenced types
func NewConsensusOptimizer() *ConsensusOptimizer {
	return &ConsensusOptimizer{
		learningRate:     0.01,
		optimizationGoal: OptimizeBalance,
		predictionCache:  make(map[string]OptimizationResult),
	}
}

func (co *ConsensusOptimizer) Optimize(params *ConsensusParameters, metrics *PerformanceMetrics, history []MetricsSnapshot) *OptimizationResult {
	// Placeholder optimization logic
	return &OptimizationResult{
		RecommendedParameters: *params,
		ExpectedImprovement:   0.05,
		Confidence:           0.7,
		Reasoning:            []string{"Current parameters are near optimal"},
	}
}

func NewNetworkHealthTracker() *NetworkHealthTracker {
	return &NetworkHealthTracker{
		nodes:              make(map[string]*NodeState),
		networkTopology:    &NetworkTopology{},
		partitionDetector:  &PartitionDetector{},
		healingStrategies:  make([]HealingStrategy, 0),
	}
}

func (nht *NetworkHealthTracker) DetectPartitions() {
	// Placeholder partition detection logic
}