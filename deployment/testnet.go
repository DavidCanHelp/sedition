// Package deployment provides testnet deployment configuration and management
package deployment

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/crypto"
	"github.com/davidcanhelp/sedition/network"
	"github.com/davidcanhelp/sedition/storage"
	poc "github.com/davidcanhelp/sedition"
)

// TestnetConfig defines the testnet configuration
type TestnetConfig struct {
	// Network configuration
	NetworkName      string        `json:"network_name"`
	ChainID          string        `json:"chain_id"`
	GenesisTime      time.Time     `json:"genesis_time"`
	
	// Consensus parameters
	MinStake         string        `json:"min_stake"`
	BlockTime        string        `json:"block_time"`
	EpochLength      int64         `json:"epoch_length"`
	CommitteeSize    int           `json:"committee_size"`
	
	// Network parameters
	MaxPeers         int           `json:"max_peers"`
	BootstrapNodes   []string      `json:"bootstrap_nodes"`
	P2PPort          int           `json:"p2p_port"`
	RPCPort          int           `json:"rpc_port"`
	MetricsPort      int           `json:"metrics_port"`
	
	// Quality analysis parameters
	MinQualityScore  float64       `json:"min_quality_score"`
	QualityWeights   QualityWeights `json:"quality_weights"`
	
	// Slashing parameters
	SlashingRates    SlashingRates `json:"slashing_rates"`
	
	// Economic parameters
	RewardRate       float64       `json:"reward_rate"`
	InflationRate    float64       `json:"inflation_rate"`
}

// QualityWeights defines weights for quality score calculation
type QualityWeights struct {
	Complexity    float64 `json:"complexity"`
	Coverage      float64 `json:"coverage"`
	Documentation float64 `json:"documentation"`
	Security      float64 `json:"security"`
}

// SlashingRates defines slashing penalties for different violations
type SlashingRates struct {
	MaliciousCode     float64 `json:"malicious_code"`
	FalseContribution float64 `json:"false_contribution"`
	DoubleProposal    float64 `json:"double_proposal"`
	NetworkAttack     float64 `json:"network_attack"`
	QualityViolation  float64 `json:"quality_violation"`
}

// ValidatorConfig defines configuration for a testnet validator
type ValidatorConfig struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	Stake            string    `json:"stake"`
	Host             string    `json:"host"`
	P2PPort          int       `json:"p2p_port"`
	RPCPort          int       `json:"rpc_port"`
	MetricsPort      int       `json:"metrics_port"`
	
	// Validator keys
	ValidatorKey     string    `json:"validator_key"`
	VRFKey           string    `json:"vrf_key"`
	
	// Configuration
	DataDir          string    `json:"data_dir"`
	LogLevel         string    `json:"log_level"`
	EnableMetrics    bool      `json:"enable_metrics"`
	EnableRPC        bool      `json:"enable_rpc"`
}

// TestnetDeployment manages a complete testnet deployment
type TestnetDeployment struct {
	config         *TestnetConfig
	validators     map[string]*TestnetValidator
	consensus      *poc.EnhancedConsensusEngine
	genesisState   *poc.GenesisState
	
	// Monitoring
	metrics        *DeploymentMetrics
	healthChecker  *HealthChecker
	
	// Control
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
}

// TestnetValidator represents a validator node in the testnet
type TestnetValidator struct {
	config       *ValidatorConfig
	consensus    *poc.EnhancedConsensusEngine
	p2pNode      *network.P2PNode
	database     *storage.BlockchainDB
	signer       *crypto.Signer
	vrf          *crypto.VRF
	
	// Runtime state
	isRunning    bool
	startTime    time.Time
	metrics      *ValidatorMetrics
	
	// Control channels
	stopChan     chan struct{}
	healthChan   chan HealthStatus
}

// ValidatorMetrics tracks validator performance
type ValidatorMetrics struct {
	BlocksProposed     int64         `json:"blocks_proposed"`
	BlocksValidated    int64         `json:"blocks_validated"`
	TransactionsProcessed int64      `json:"transactions_processed"`
	PeerCount          int           `json:"peer_count"`
	Uptime             time.Duration `json:"uptime"`
	LastBlockTime      time.Time     `json:"last_block_time"`
	QualityScore       float64       `json:"quality_score"`
	Reputation         float64       `json:"reputation"`
	
	// Network metrics
	MessagesSent       int64         `json:"messages_sent"`
	MessagesReceived   int64         `json:"messages_received"`
	BytesSent          int64         `json:"bytes_sent"`
	BytesReceived      int64         `json:"bytes_received"`
}

// DeploymentMetrics tracks overall testnet health
type DeploymentMetrics struct {
	TotalValidators    int           `json:"total_validators"`
	ActiveValidators   int           `json:"active_validators"`
	ChainHeight        int64         `json:"chain_height"`
	TotalTransactions  int64         `json:"total_transactions"`
	AverageBlockTime   time.Duration `json:"average_block_time"`
	NetworkThroughput  float64       `json:"network_throughput"`
	ConsensusHealth    float64       `json:"consensus_health"`
	
	// Quality metrics
	AverageQuality     float64       `json:"average_quality"`
	QualityDistribution map[string]int `json:"quality_distribution"`
	
	LastUpdated        time.Time     `json:"last_updated"`
}

// HealthStatus represents the health status of a component
type HealthStatus struct {
	Component    string    `json:"component"`
	Status       string    `json:"status"` // healthy, degraded, unhealthy
	Message      string    `json:"message"`
	Timestamp    time.Time `json:"timestamp"`
	Metrics      map[string]interface{} `json:"metrics"`
}

// HealthChecker monitors testnet health
type HealthChecker struct {
	deployment   *TestnetDeployment
	checkInterval time.Duration
	alerts       []Alert
	
	// Health thresholds
	maxBlockTime      time.Duration
	minActiveValidators int
	maxConsensusLatency time.Duration
}

// Alert represents a health alert
type Alert struct {
	Level      string    `json:"level"`     // info, warning, error, critical
	Component  string    `json:"component"`
	Message    string    `json:"message"`
	Timestamp  time.Time `json:"timestamp"`
	Resolved   bool      `json:"resolved"`
}

// NewTestnetConfig creates a default testnet configuration
func NewTestnetConfig(networkName string) *TestnetConfig {
	return &TestnetConfig{
		NetworkName:     networkName,
		ChainID:         fmt.Sprintf("%s-testnet", networkName),
		GenesisTime:     time.Now().Add(5 * time.Minute), // Start in 5 minutes
		MinStake:        "1000000",  // 1M tokens
		BlockTime:       "5s",
		EpochLength:     100,
		CommitteeSize:   21,
		MaxPeers:        50,
		P2PPort:         9000,
		RPCPort:         8000,
		MetricsPort:     9090,
		MinQualityScore: 30.0,
		QualityWeights: QualityWeights{
			Complexity:    0.25,
			Coverage:      0.30,
			Documentation: 0.20,
			Security:      0.25,
		},
		SlashingRates: SlashingRates{
			MaliciousCode:     0.50,
			FalseContribution: 0.30,
			DoubleProposal:    0.40,
			NetworkAttack:     0.70,
			QualityViolation:  0.20,
		},
		RewardRate:    0.05, // 5% annual reward rate
		InflationRate: 0.02, // 2% annual inflation
	}
}

// GenerateValidatorConfigs creates validator configurations for the testnet
func (cfg *TestnetConfig) GenerateValidatorConfigs(numValidators int, baseDir string) ([]*ValidatorConfig, error) {
	validators := make([]*ValidatorConfig, numValidators)
	
	for i := 0; i < numValidators; i++ {
		// Create cryptographic keys
		signer, err := crypto.NewSigner()
		if err != nil {
			return nil, fmt.Errorf("failed to create signer for validator %d: %w", i, err)
		}
		
		vrf, err := crypto.NewVRF()
		if err != nil {
			return nil, fmt.Errorf("failed to create VRF for validator %d: %w", i, err)
		}
		
		// Calculate stake (vary stakes for realistic distribution)
		baseStake, _ := big.NewInt(0).SetString(cfg.MinStake, 10)
		multiplier := int64(1 + (i % 5)) // 1x to 5x base stake
		stake := new(big.Int).Mul(baseStake, big.NewInt(multiplier))
		
		validators[i] = &ValidatorConfig{
			ID:           fmt.Sprintf("validator-%03d", i),
			Name:         fmt.Sprintf("Testnet Validator %d", i),
			Stake:        stake.String(),
			Host:         "localhost", // Change for distributed deployment
			P2PPort:      cfg.P2PPort + i,
			RPCPort:      cfg.RPCPort + i,
			MetricsPort:  cfg.MetricsPort + i,
			ValidatorKey: fmt.Sprintf("%x", signer.GetPublicKey()),
			VRFKey:       fmt.Sprintf("%x", vrf.GetPublicKey()),
			DataDir:      filepath.Join(baseDir, fmt.Sprintf("validator-%03d", i)),
			LogLevel:     "info",
			EnableMetrics: true,
			EnableRPC:    true,
		}
	}
	
	return validators, nil
}

// NewTestnetDeployment creates a new testnet deployment
func NewTestnetDeployment(config *TestnetConfig) *TestnetDeployment {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &TestnetDeployment{
		config:     config,
		validators: make(map[string]*TestnetValidator),
		metrics: &DeploymentMetrics{
			QualityDistribution: make(map[string]int),
		},
		healthChecker: &HealthChecker{
			checkInterval:       30 * time.Second,
			maxBlockTime:        15 * time.Second,
			minActiveValidators: 3,
			maxConsensusLatency: 10 * time.Second,
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// DeployTestnet deploys the complete testnet
func (d *TestnetDeployment) DeployTestnet(validatorConfigs []*ValidatorConfig) error {
	log.Printf("Starting testnet deployment: %s", d.config.NetworkName)
	
	// Create genesis state
	if err := d.createGenesisState(validatorConfigs); err != nil {
		return fmt.Errorf("failed to create genesis state: %w", err)
	}
	
	// Deploy validators
	for _, config := range validatorConfigs {
		validator, err := d.createValidator(config)
		if err != nil {
			return fmt.Errorf("failed to create validator %s: %w", config.ID, err)
		}
		d.validators[config.ID] = validator
	}
	
	// Start validators
	for _, validator := range d.validators {
		if err := d.startValidator(validator); err != nil {
			log.Printf("Failed to start validator %s: %v", validator.config.ID, err)
		}
	}
	
	// Wait for network to establish
	time.Sleep(5 * time.Second)
	
	// Connect validators
	if err := d.establishPeerConnections(); err != nil {
		return fmt.Errorf("failed to establish peer connections: %w", err)
	}
	
	// Start monitoring
	d.startHealthChecker()
	d.startMetricsCollection()
	
	log.Printf("Testnet deployment complete. %d validators running.", len(d.validators))
	return nil
}

// createGenesisState creates the initial blockchain state
func (d *TestnetDeployment) createGenesisState(validatorConfigs []*ValidatorConfig) error {
	minStake, ok := big.NewInt(0).SetString(d.config.MinStake, 10)
	if !ok {
		return fmt.Errorf("invalid min stake: %s", d.config.MinStake)
	}
	
	blockTime, err := time.ParseDuration(d.config.BlockTime)
	if err != nil {
		return fmt.Errorf("invalid block time: %s", d.config.BlockTime)
	}
	
	// Create consensus engine
	d.consensus = poc.NewEnhancedConsensusEngine(minStake, blockTime)
	
	// Register genesis validators
	genesisValidators := make([]poc.GenesisValidator, len(validatorConfigs))
	for i, config := range validatorConfigs {
		stake, _ := big.NewInt(0).SetString(config.Stake, 10)
		
		genesisValidators[i] = poc.GenesisValidator{
			Address:         config.ID,
			PublicKey:       config.ValidatorKey,
			TokenStake:      config.Stake,
			ReputationScore: 5.0, // Start with neutral reputation
		}
		
		// Register with consensus engine
		seed := []byte(config.ValidatorKey)
		err := d.consensus.RegisterValidator(config.ID, stake, seed)
		if err != nil {
			return fmt.Errorf("failed to register genesis validator %s: %w", config.ID, err)
		}
	}
	
	// Create genesis state
	d.genesisState = &poc.GenesisState{
		Validators:  genesisValidators,
		EpochLength: d.config.EpochLength,
		BlockTime:   blockTime,
		MinStake:    d.config.MinStake,
		GenesisTime: d.config.GenesisTime,
	}
	
	return nil
}

// createValidator creates a single validator instance
func (d *TestnetDeployment) createValidator(config *ValidatorConfig) (*TestnetValidator, error) {
	// Create data directory
	if err := os.MkdirAll(config.DataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}
	
	// Create database
	database, err := storage.NewBlockchainDB(config.DataDir)
	if err != nil {
		return nil, fmt.Errorf("failed to create database: %w", err)
	}
	
	// Create cryptographic keys
	seed := []byte(config.ValidatorKey)
	signer, err := crypto.NewSignerFromSeed(seed)
	if err != nil {
		return nil, fmt.Errorf("failed to create signer: %w", err)
	}
	
	vrf, err := crypto.NewVRFFromSeed(seed)
	if err != nil {
		return nil, fmt.Errorf("failed to create VRF: %w", err)
	}
	
	// Create P2P node
	listenAddr := fmt.Sprintf("%s:%d", config.Host, config.P2PPort)
	p2pNode, err := network.NewP2PNode(listenAddr, signer, d.config.BootstrapNodes)
	if err != nil {
		return nil, fmt.Errorf("failed to create P2P node: %w", err)
	}
	
	// Create consensus engine copy for this validator
	minStake, _ := big.NewInt(0).SetString(d.config.MinStake, 10)
	blockTime, _ := time.ParseDuration(d.config.BlockTime)
	consensus := poc.NewEnhancedConsensusEngine(minStake, blockTime)
	
	// Register all genesis validators
	for _, genesisValidator := range d.genesisState.Validators {
		stake, _ := big.NewInt(0).SetString(genesisValidator.TokenStake, 10)
		seed := []byte(genesisValidator.PublicKey)
		consensus.RegisterValidator(genesisValidator.Address, stake, seed)
	}
	
	validator := &TestnetValidator{
		config:     config,
		consensus:  consensus,
		p2pNode:    p2pNode,
		database:   database,
		signer:     signer,
		vrf:        vrf,
		stopChan:   make(chan struct{}),
		healthChan: make(chan HealthStatus, 10),
		metrics: &ValidatorMetrics{
			LastBlockTime: time.Now(),
		},
	}
	
	return validator, nil
}

// startValidator starts a validator node
func (d *TestnetDeployment) startValidator(validator *TestnetValidator) error {
	log.Printf("Starting validator: %s", validator.config.ID)
	
	// Start P2P node
	if err := validator.p2pNode.Start(); err != nil {
		return fmt.Errorf("failed to start P2P node: %w", err)
	}
	
	// Start validator goroutine
	d.wg.Add(1)
	go d.runValidator(validator)
	
	validator.isRunning = true
	validator.startTime = time.Now()
	
	return nil
}

// runValidator runs the validator consensus loop
func (d *TestnetDeployment) runValidator(validator *TestnetValidator) {
	defer d.wg.Done()
	
	ticker := time.NewTicker(time.Second) // Check for consensus opportunities every second
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-validator.stopChan:
			return
		case <-ticker.C:
			d.processConsensusRound(validator)
		}
	}
}

// processConsensusRound processes a single consensus round for a validator
func (d *TestnetDeployment) processConsensusRound(validator *TestnetValidator) {
	// Try to be selected as leader
	leader, proof, err := validator.consensus.SelectBlockProposer()
	if err != nil {
		return
	}
	
	// If we're the leader, propose a block
	if leader == validator.config.ID {
		commits := d.generateTestCommits(validator.config.ID)
		block, err := validator.consensus.ProposeBlock(leader, commits, proof)
		if err != nil {
			log.Printf("Validator %s failed to propose block: %v", validator.config.ID, err)
			return
		}
		
		// Broadcast block to network
		validator.p2pNode.BroadcastBlock(block)
		
		validator.metrics.BlocksProposed++
		validator.metrics.LastBlockTime = time.Now()
		
		log.Printf("Validator %s proposed block at height %d", validator.config.ID, block.Height)
	}
	
	// Always participate in validation
	// (In a real implementation, this would be triggered by receiving block proposals)
	validator.metrics.BlocksValidated++
}

// generateTestCommits generates test commits for block proposals
func (d *TestnetDeployment) generateTestCommits(validatorID string) []poc.Commit {
	numCommits := 1 + (time.Now().UnixNano() % 5) // 1-5 commits
	commits := make([]poc.Commit, numCommits)
	
	for i := int64(0); i < numCommits; i++ {
		quality := 60.0 + float64((time.Now().UnixNano()+i)%40) // 60-100 quality
		
		commits[i] = poc.Commit{
			ID:            fmt.Sprintf("%s_%d_%d", validatorID, time.Now().UnixNano(), i),
			Author:        validatorID,
			Hash:          make([]byte, 32),
			Timestamp:     time.Now(),
			Message:       fmt.Sprintf("Test commit %d from %s", i, validatorID),
			FilesChanged:  []string{fmt.Sprintf("file_%d.go", i)},
			LinesAdded:    int(10 + (time.Now().UnixNano() % 200)),
			LinesModified: int(5 + (time.Now().UnixNano() % 100)),
			QualityScore:  quality,
		}
		
		// Generate random hash
		copy(commits[i].Hash, fmt.Sprintf("%032d", time.Now().UnixNano()+i))
	}
	
	return commits
}

// establishPeerConnections connects validators to each other
func (d *TestnetDeployment) establishPeerConnections() error {
	validators := make([]*TestnetValidator, 0, len(d.validators))
	for _, v := range d.validators {
		validators = append(validators, v)
	}
	
	// Connect each validator to several others
	for i, validator := range validators {
		connectTo := 3 // Connect to 3 peers
		for j := 0; j < connectTo && j < len(validators)-1; j++ {
			peerIndex := (i + j + 1) % len(validators)
			peer := validators[peerIndex]
			
			peerAddr := fmt.Sprintf("%s:%d", peer.config.Host, peer.config.P2PPort)
			err := validator.p2pNode.ConnectToPeer(peerAddr)
			if err != nil {
				log.Printf("Failed to connect %s to %s: %v", validator.config.ID, peer.config.ID, err)
			}
		}
	}
	
	return nil
}

// startHealthChecker starts the health monitoring system
func (d *TestnetDeployment) startHealthChecker() {
	d.healthChecker.deployment = d
	
	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		
		ticker := time.NewTicker(d.healthChecker.checkInterval)
		defer ticker.Stop()
		
		for {
			select {
			case <-d.ctx.Done():
				return
			case <-ticker.C:
				d.performHealthCheck()
			}
		}
	}()
}

// performHealthCheck checks the health of all testnet components
func (d *TestnetDeployment) performHealthCheck() {
	// Check validator health
	activeValidators := 0
	for _, validator := range d.validators {
		if validator.isRunning {
			activeValidators++
			
			// Check if validator is producing blocks
			timeSinceLastBlock := time.Since(validator.metrics.LastBlockTime)
			if timeSinceLastBlock > d.healthChecker.maxBlockTime*2 {
				alert := Alert{
					Level:     "warning",
					Component: fmt.Sprintf("validator-%s", validator.config.ID),
					Message:   fmt.Sprintf("No blocks produced for %v", timeSinceLastBlock),
					Timestamp: time.Now(),
				}
				d.healthChecker.alerts = append(d.healthChecker.alerts, alert)
			}
		}
	}
	
	// Check minimum validators
	if activeValidators < d.healthChecker.minActiveValidators {
		alert := Alert{
			Level:     "critical",
			Component: "consensus",
			Message:   fmt.Sprintf("Only %d active validators (minimum: %d)", activeValidators, d.healthChecker.minActiveValidators),
			Timestamp: time.Now(),
		}
		d.healthChecker.alerts = append(d.healthChecker.alerts, alert)
	}
	
	// Update deployment metrics
	d.metrics.TotalValidators = len(d.validators)
	d.metrics.ActiveValidators = activeValidators
	d.metrics.LastUpdated = time.Now()
}

// startMetricsCollection starts collecting deployment metrics
func (d *TestnetDeployment) startMetricsCollection() {
	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		
		for {
			select {
			case <-d.ctx.Done():
				return
			case <-ticker.C:
				d.collectMetrics()
			}
		}
	}()
}

// collectMetrics collects metrics from all validators
func (d *TestnetDeployment) collectMetrics() {
	var totalQuality float64
	qualityCount := 0
	
	for _, validator := range d.validators {
		if validator.isRunning {
			// Update validator uptime
			validator.metrics.Uptime = time.Since(validator.startTime)
			
			// Get peer count
			peers := validator.p2pNode.GetPeers()
			validator.metrics.PeerCount = len(peers)
			
			// Aggregate quality scores
			if validator.metrics.QualityScore > 0 {
				totalQuality += validator.metrics.QualityScore
				qualityCount++
			}
		}
	}
	
	// Calculate average quality
	if qualityCount > 0 {
		d.metrics.AverageQuality = totalQuality / float64(qualityCount)
	}
	
	// Get chain height (from any validator)
	for _, validator := range d.validators {
		if validator.isRunning {
			d.metrics.ChainHeight = validator.consensus.GetChainHeight()
			break
		}
	}
}

// ExportMetrics exports testnet metrics to a file
func (d *TestnetDeployment) ExportMetrics(filename string) error {
	data, err := json.MarshalIndent(d.metrics, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metrics: %w", err)
	}
	
	return ioutil.WriteFile(filename, data, 0644)
}

// StopTestnet gracefully stops the testnet
func (d *TestnetDeployment) StopTestnet() error {
	log.Printf("Stopping testnet: %s", d.config.NetworkName)
	
	// Cancel context to stop all goroutines
	d.cancel()
	
	// Stop all validators
	for _, validator := range d.validators {
		close(validator.stopChan)
		validator.p2pNode.Stop()
		validator.database.Close()
		validator.isRunning = false
	}
	
	// Wait for all goroutines to finish
	d.wg.Wait()
	
	log.Printf("Testnet stopped")
	return nil
}

// GetStatus returns the current testnet status
func (d *TestnetDeployment) GetStatus() *TestnetStatus {
	return &TestnetStatus{
		NetworkName:     d.config.NetworkName,
		ChainID:         d.config.ChainID,
		Status:          d.determineOverallStatus(),
		Validators:      len(d.validators),
		ActiveValidators: d.metrics.ActiveValidators,
		ChainHeight:     d.metrics.ChainHeight,
		AverageQuality:  d.metrics.AverageQuality,
		Alerts:          len(d.healthChecker.alerts),
		LastUpdated:     time.Now(),
	}
}

// TestnetStatus represents the overall status of the testnet
type TestnetStatus struct {
	NetworkName      string    `json:"network_name"`
	ChainID          string    `json:"chain_id"`
	Status           string    `json:"status"` // running, degraded, stopped
	Validators       int       `json:"validators"`
	ActiveValidators int       `json:"active_validators"`
	ChainHeight      int64     `json:"chain_height"`
	AverageQuality   float64   `json:"average_quality"`
	Alerts           int       `json:"alerts"`
	LastUpdated      time.Time `json:"last_updated"`
}

// determineOverallStatus determines the overall testnet status
func (d *TestnetDeployment) determineOverallStatus() string {
	criticalAlerts := 0
	for _, alert := range d.healthChecker.alerts {
		if alert.Level == "critical" && !alert.Resolved {
			criticalAlerts++
		}
	}
	
	if criticalAlerts > 0 {
		return "degraded"
	}
	
	if d.metrics.ActiveValidators >= d.healthChecker.minActiveValidators {
		return "running"
	}
	
	return "stopped"
}

// CreateDefaultTestnet creates a default testnet configuration and deployment
func CreateDefaultTestnet(numValidators int, baseDir string) (*TestnetDeployment, error) {
	// Create config
	config := NewTestnetConfig("sedition")
	
	// Generate validator configs
	validatorConfigs, err := config.GenerateValidatorConfigs(numValidators, baseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to generate validator configs: %w", err)
	}
	
	// Create deployment
	deployment := NewTestnetDeployment(config)
	
	// Deploy testnet
	if err := deployment.DeployTestnet(validatorConfigs); err != nil {
		return nil, fmt.Errorf("failed to deploy testnet: %w", err)
	}
	
	return deployment, nil
}

// SaveTestnetConfig saves testnet configuration to file
func (cfg *TestnetConfig) SaveConfig(filename string) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}
	
	return ioutil.WriteFile(filename, data, 0644)
}

// LoadTestnetConfig loads testnet configuration from file  
func LoadTestnetConfig(filename string) (*TestnetConfig, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	
	var config TestnetConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}
	
	return &config, nil
}