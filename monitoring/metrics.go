package monitoring

import (
	"context"
	"fmt"
	"net/http"
	"runtime"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/client_golang/prometheus/push"
)

// MetricsCollector collects and exposes metrics
type MetricsCollector struct {
	mu sync.RWMutex

	// Blockchain metrics
	BlockHeight         prometheus.Gauge
	BlockProcessingTime prometheus.Histogram
	BlockSize           prometheus.Histogram
	TransactionCount    prometheus.Counter
	TransactionPool     prometheus.Gauge
	BlockValidationTime prometheus.Histogram

	// Consensus metrics
	ConsensusRounds          prometheus.Counter
	ConsensusFailures        prometheus.Counter
	ValidatorCount           prometheus.Gauge
	ValidatorStake           prometheus.Gauge
	ProposerSelectionTime    prometheus.Histogram
	VotingRoundDuration      prometheus.Histogram
	ByzantineFaults          prometheus.Counter

	// Network metrics
	PeerCount            prometheus.Gauge
	MessagesReceived     *prometheus.CounterVec
	MessagesSent         *prometheus.CounterVec
	BytesReceived        prometheus.Counter
	BytesSent            prometheus.Counter
	ConnectionErrors     prometheus.Counter
	PeerLatency          *prometheus.HistogramVec
	NetworkPartitions    prometheus.Counter

	// Performance metrics
	CPUUsage            prometheus.Gauge
	MemoryUsage         prometheus.Gauge
	DiskUsage           prometheus.Gauge
	GoroutineCount      prometheus.Gauge
	GCPauseTime         prometheus.Histogram
	DatabaseQueries     *prometheus.CounterVec
	DatabaseQueryTime   *prometheus.HistogramVec
	CacheHitRate        prometheus.Gauge
	CacheMissCount      prometheus.Counter

	// API metrics
	APIRequests         *prometheus.CounterVec
	APIRequestDuration  *prometheus.HistogramVec
	APIErrors           *prometheus.CounterVec
	WebSocketClients    prometheus.Gauge
	WebSocketMessages   prometheus.Counter

	// Security metrics
	AuthenticationAttempts *prometheus.CounterVec
	AuthorizationFailures  prometheus.Counter
	RateLimitHits          prometheus.Counter
	SuspiciousActivities   prometheus.Counter

	// Business metrics
	ContributionScore      *prometheus.GaugeVec
	ValidatorReputation    *prometheus.GaugeVec
	RewardDistribution     prometheus.Counter
	SlashingEvents         prometheus.Counter

	// Custom metrics registry
	customMetrics map[string]prometheus.Collector
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(namespace string) *MetricsCollector {
	m := &MetricsCollector{
		customMetrics: make(map[string]prometheus.Collector),

		// Blockchain metrics
		BlockHeight: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "blockchain",
			Name:      "height",
			Help:      "Current blockchain height",
		}),
		BlockProcessingTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "blockchain",
			Name:      "block_processing_seconds",
			Help:      "Time taken to process a block",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 10),
		}),
		BlockSize: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "blockchain",
			Name:      "block_size_bytes",
			Help:      "Size of blocks in bytes",
			Buckets:   prometheus.ExponentialBuckets(1024, 2, 10),
		}),
		TransactionCount: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "blockchain",
			Name:      "transactions_total",
			Help:      "Total number of processed transactions",
		}),
		TransactionPool: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "blockchain",
			Name:      "pending_transactions",
			Help:      "Number of transactions in the pool",
		}),
		BlockValidationTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "blockchain",
			Name:      "validation_seconds",
			Help:      "Time taken to validate a block",
			Buckets:   prometheus.ExponentialBuckets(0.0001, 2, 10),
		}),

		// Consensus metrics
		ConsensusRounds: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "consensus",
			Name:      "rounds_total",
			Help:      "Total consensus rounds",
		}),
		ConsensusFailures: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "consensus",
			Name:      "failures_total",
			Help:      "Total consensus failures",
		}),
		ValidatorCount: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "consensus",
			Name:      "validator_count",
			Help:      "Number of active validators",
		}),
		ValidatorStake: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "consensus",
			Name:      "total_stake",
			Help:      "Total staked amount",
		}),
		ProposerSelectionTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "consensus",
			Name:      "proposer_selection_seconds",
			Help:      "Time to select block proposer",
			Buckets:   prometheus.ExponentialBuckets(0.00001, 2, 10),
		}),
		VotingRoundDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "consensus",
			Name:      "voting_round_seconds",
			Help:      "Duration of voting rounds",
			Buckets:   prometheus.ExponentialBuckets(0.1, 2, 10),
		}),
		ByzantineFaults: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "consensus",
			Name:      "byzantine_faults_total",
			Help:      "Total detected Byzantine faults",
		}),

		// Network metrics
		PeerCount: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "peer_count",
			Help:      "Number of connected peers",
		}),
		MessagesReceived: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "messages_received_total",
			Help:      "Total messages received by type",
		}, []string{"type"}),
		MessagesSent: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "messages_sent_total",
			Help:      "Total messages sent by type",
		}, []string{"type"}),
		BytesReceived: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "bytes_received_total",
			Help:      "Total bytes received",
		}),
		BytesSent: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "bytes_sent_total",
			Help:      "Total bytes sent",
		}),
		ConnectionErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "connection_errors_total",
			Help:      "Total connection errors",
		}),
		PeerLatency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "peer_latency_seconds",
			Help:      "Latency to peers",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 10),
		}, []string{"peer"}),
		NetworkPartitions: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "network",
			Name:      "partitions_detected_total",
			Help:      "Total network partitions detected",
		}),

		// Performance metrics
		CPUUsage: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "cpu_usage_percent",
			Help:      "CPU usage percentage",
		}),
		MemoryUsage: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "memory_usage_bytes",
			Help:      "Memory usage in bytes",
		}),
		DiskUsage: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "disk_usage_bytes",
			Help:      "Disk usage in bytes",
		}),
		GoroutineCount: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "goroutines",
			Help:      "Number of goroutines",
		}),
		GCPauseTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "gc_pause_seconds",
			Help:      "GC pause time",
			Buckets:   prometheus.ExponentialBuckets(0.00001, 2, 10),
		}),
		DatabaseQueries: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "database",
			Name:      "queries_total",
			Help:      "Total database queries",
		}, []string{"operation"}),
		DatabaseQueryTime: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "database",
			Name:      "query_duration_seconds",
			Help:      "Database query duration",
			Buckets:   prometheus.ExponentialBuckets(0.0001, 2, 10),
		}, []string{"operation"}),
		CacheHitRate: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "cache",
			Name:      "hit_rate",
			Help:      "Cache hit rate",
		}),
		CacheMissCount: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "cache",
			Name:      "misses_total",
			Help:      "Total cache misses",
		}),

		// API metrics
		APIRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "api",
			Name:      "requests_total",
			Help:      "Total API requests",
		}, []string{"method", "endpoint", "status"}),
		APIRequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "api",
			Name:      "request_duration_seconds",
			Help:      "API request duration",
			Buckets:   prometheus.DefBuckets,
		}, []string{"method", "endpoint"}),
		APIErrors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "api",
			Name:      "errors_total",
			Help:      "Total API errors",
		}, []string{"method", "endpoint", "error"}),
		WebSocketClients: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "websocket",
			Name:      "clients",
			Help:      "Number of WebSocket clients",
		}),
		WebSocketMessages: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "websocket",
			Name:      "messages_total",
			Help:      "Total WebSocket messages",
		}),

		// Security metrics
		AuthenticationAttempts: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "security",
			Name:      "auth_attempts_total",
			Help:      "Total authentication attempts",
		}, []string{"result"}),
		AuthorizationFailures: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "security",
			Name:      "authz_failures_total",
			Help:      "Total authorization failures",
		}),
		RateLimitHits: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "security",
			Name:      "rate_limit_hits_total",
			Help:      "Total rate limit hits",
		}),
		SuspiciousActivities: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "security",
			Name:      "suspicious_activities_total",
			Help:      "Total suspicious activities detected",
		}),

		// Business metrics
		ContributionScore: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "business",
			Name:      "contribution_score",
			Help:      "Validator contribution scores",
		}, []string{"validator"}),
		ValidatorReputation: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "business",
			Name:      "validator_reputation",
			Help:      "Validator reputation scores",
		}, []string{"validator"}),
		RewardDistribution: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "business",
			Name:      "rewards_distributed_total",
			Help:      "Total rewards distributed",
		}),
		SlashingEvents: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "business",
			Name:      "slashing_events_total",
			Help:      "Total slashing events",
		}),
	}

	// Register all metrics
	m.registerAll()

	return m
}

// registerAll registers all metrics with Prometheus
func (m *MetricsCollector) registerAll() {
	// Blockchain
	prometheus.MustRegister(m.BlockHeight)
	prometheus.MustRegister(m.BlockProcessingTime)
	prometheus.MustRegister(m.BlockSize)
	prometheus.MustRegister(m.TransactionCount)
	prometheus.MustRegister(m.TransactionPool)
	prometheus.MustRegister(m.BlockValidationTime)

	// Consensus
	prometheus.MustRegister(m.ConsensusRounds)
	prometheus.MustRegister(m.ConsensusFailures)
	prometheus.MustRegister(m.ValidatorCount)
	prometheus.MustRegister(m.ValidatorStake)
	prometheus.MustRegister(m.ProposerSelectionTime)
	prometheus.MustRegister(m.VotingRoundDuration)
	prometheus.MustRegister(m.ByzantineFaults)

	// Network
	prometheus.MustRegister(m.PeerCount)
	prometheus.MustRegister(m.MessagesReceived)
	prometheus.MustRegister(m.MessagesSent)
	prometheus.MustRegister(m.BytesReceived)
	prometheus.MustRegister(m.BytesSent)
	prometheus.MustRegister(m.ConnectionErrors)
	prometheus.MustRegister(m.PeerLatency)
	prometheus.MustRegister(m.NetworkPartitions)

	// Performance
	prometheus.MustRegister(m.CPUUsage)
	prometheus.MustRegister(m.MemoryUsage)
	prometheus.MustRegister(m.DiskUsage)
	prometheus.MustRegister(m.GoroutineCount)
	prometheus.MustRegister(m.GCPauseTime)
	prometheus.MustRegister(m.DatabaseQueries)
	prometheus.MustRegister(m.DatabaseQueryTime)
	prometheus.MustRegister(m.CacheHitRate)
	prometheus.MustRegister(m.CacheMissCount)

	// API
	prometheus.MustRegister(m.APIRequests)
	prometheus.MustRegister(m.APIRequestDuration)
	prometheus.MustRegister(m.APIErrors)
	prometheus.MustRegister(m.WebSocketClients)
	prometheus.MustRegister(m.WebSocketMessages)

	// Security
	prometheus.MustRegister(m.AuthenticationAttempts)
	prometheus.MustRegister(m.AuthorizationFailures)
	prometheus.MustRegister(m.RateLimitHits)
	prometheus.MustRegister(m.SuspiciousActivities)

	// Business
	prometheus.MustRegister(m.ContributionScore)
	prometheus.MustRegister(m.ValidatorReputation)
	prometheus.MustRegister(m.RewardDistribution)
	prometheus.MustRegister(m.SlashingEvents)
}

// StartSystemMetricsCollection starts collecting system metrics
func (m *MetricsCollector) StartSystemMetricsCollection(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		for {
			select {
			case <-ctx.Done():
				ticker.Stop()
				return
			case <-ticker.C:
				m.collectSystemMetrics()
			}
		}
	}()
}

// collectSystemMetrics collects system-level metrics
func (m *MetricsCollector) collectSystemMetrics() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Memory metrics
	m.MemoryUsage.Set(float64(memStats.Alloc))
	m.GoroutineCount.Set(float64(runtime.NumGoroutine()))

	// GC metrics
	if len(memStats.PauseNs) > 0 {
		m.GCPauseTime.Observe(float64(memStats.PauseNs[0]) / 1e9)
	}
}

// RecordBlockProcessing records block processing metrics
func (m *MetricsCollector) RecordBlockProcessing(height uint64, processingTime time.Duration, size int) {
	m.BlockHeight.Set(float64(height))
	m.BlockProcessingTime.Observe(processingTime.Seconds())
	m.BlockSize.Observe(float64(size))
}

// RecordTransaction records transaction metrics
func (m *MetricsCollector) RecordTransaction() {
	m.TransactionCount.Inc()
}

// RecordConsensusRound records consensus round metrics
func (m *MetricsCollector) RecordConsensusRound(duration time.Duration, success bool) {
	m.ConsensusRounds.Inc()
	m.VotingRoundDuration.Observe(duration.Seconds())
	if !success {
		m.ConsensusFailures.Inc()
	}
}

// RecordAPIRequest records API request metrics
func (m *MetricsCollector) RecordAPIRequest(method, endpoint, status string, duration time.Duration) {
	m.APIRequests.WithLabelValues(method, endpoint, status).Inc()
	m.APIRequestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
}

// RecordNetworkMessage records network message metrics
func (m *MetricsCollector) RecordNetworkMessage(messageType string, sent bool, size int) {
	if sent {
		m.MessagesSent.WithLabelValues(messageType).Inc()
		m.BytesSent.Add(float64(size))
	} else {
		m.MessagesReceived.WithLabelValues(messageType).Inc()
		m.BytesReceived.Add(float64(size))
	}
}

// HTTPHandler returns the Prometheus HTTP handler
func (m *MetricsCollector) HTTPHandler() http.Handler {
	return promhttp.Handler()
}

// PushGateway support for batch job metrics
type PushGateway struct {
	pusher *push.Pusher
}

func NewPushGateway(url, job string) *PushGateway {
	return &PushGateway{
		pusher: push.New(url, job),
	}
}

func (p *PushGateway) Push() error {
	return p.pusher.Push()
}

// HealthCheck provides health check metrics
type HealthCheck struct {
	mu         sync.RWMutex
	components map[string]ComponentHealth
}

type ComponentHealth struct {
	Name        string    `json:"name"`
	Status      string    `json:"status"` // "healthy", "degraded", "unhealthy"
	LastChecked time.Time `json:"last_checked"`
	Details     string    `json:"details,omitempty"`
}

func NewHealthCheck() *HealthCheck {
	return &HealthCheck{
		components: make(map[string]ComponentHealth),
	}
}

func (h *HealthCheck) UpdateComponent(name, status, details string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.components[name] = ComponentHealth{
		Name:        name,
		Status:      status,
		LastChecked: time.Now(),
		Details:     details,
	}
}

func (h *HealthCheck) GetStatus() map[string]ComponentHealth {
	h.mu.RLock()
	defer h.mu.RUnlock()

	result := make(map[string]ComponentHealth)
	for k, v := range h.components {
		result[k] = v
	}
	return result
}

func (h *HealthCheck) IsHealthy() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	for _, component := range h.components {
		if component.Status == "unhealthy" {
			return false
		}
	}
	return true
}

// AlertManager handles metric-based alerting
type AlertManager struct {
	mu        sync.RWMutex
	alerts    []Alert
	rules     []AlertRule
	callbacks map[string]AlertCallback
}

type Alert struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Severity    string                 `json:"severity"` // "info", "warning", "critical"
	Message     string                 `json:"message"`
	Timestamp   time.Time              `json:"timestamp"`
	Labels      map[string]string      `json:"labels"`
	Annotations map[string]string      `json:"annotations"`
}

type AlertRule struct {
	Name      string
	Query     string
	Duration  time.Duration
	Severity  string
	Threshold float64
}

type AlertCallback func(alert Alert)

func NewAlertManager() *AlertManager {
	return &AlertManager{
		alerts:    make([]Alert, 0),
		rules:     make([]AlertRule, 0),
		callbacks: make(map[string]AlertCallback),
	}
}

func (a *AlertManager) AddRule(rule AlertRule) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.rules = append(a.rules, rule)
}

func (a *AlertManager) RegisterCallback(name string, callback AlertCallback) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.callbacks[name] = callback
}

func (a *AlertManager) TriggerAlert(alert Alert) {
	a.mu.Lock()
	a.alerts = append(a.alerts, alert)
	callbacks := make([]AlertCallback, 0, len(a.callbacks))
	for _, cb := range a.callbacks {
		callbacks = append(callbacks, cb)
	}
	a.mu.Unlock()

	// Execute callbacks
	for _, cb := range callbacks {
		go cb(alert)
	}
}

func (a *AlertManager) GetActiveAlerts() []Alert {
	a.mu.RLock()
	defer a.mu.RUnlock()

	// Return alerts from last hour
	cutoff := time.Now().Add(-1 * time.Hour)
	active := make([]Alert, 0)
	for _, alert := range a.alerts {
		if alert.Timestamp.After(cutoff) {
			active = append(active, alert)
		}
	}
	return active
}

// Dashboard provides metrics dashboard data
type Dashboard struct {
	metrics *MetricsCollector
	health  *HealthCheck
	alerts  *AlertManager
}

func NewDashboard(metrics *MetricsCollector, health *HealthCheck, alerts *AlertManager) *Dashboard {
	return &Dashboard{
		metrics: metrics,
		health:  health,
		alerts:  alerts,
	}
}

func (d *Dashboard) GetDashboardData() map[string]interface{} {
	return map[string]interface{}{
		"timestamp": time.Now(),
		"health":    d.health.GetStatus(),
		"alerts":    d.alerts.GetActiveAlerts(),
		"metrics": map[string]interface{}{
			"blockchain": map[string]float64{
				"height": getGaugeValue(d.metrics.BlockHeight),
				"peers":  getGaugeValue(d.metrics.PeerCount),
			},
			"system": map[string]float64{
				"memory":     getGaugeValue(d.metrics.MemoryUsage),
				"goroutines": getGaugeValue(d.metrics.GoroutineCount),
			},
		},
	}
}

func getGaugeValue(gauge prometheus.Gauge) float64 {
	// For simplicity, returning 0 as getting actual value requires prometheus internal package
	// In production, you would use prometheus client to query the metric
	return 0
}

// CreateMetricsServer creates an HTTP server for metrics
func CreateMetricsServer(addr string, collector *MetricsCollector) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", collector.HTTPHandler())
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "OK")
	})

	return &http.Server{
		Addr:    addr,
		Handler: mux,
	}
}