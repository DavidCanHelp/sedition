package config

import (
	"math/big"
	"time"
)

// ConsensusConfig holds consensus engine configuration
type ConsensusConfig struct {
	// Block and epoch settings
	BlockTime   time.Duration `json:"block_time"`
	EpochLength int64         `json:"epoch_length"`

	// Staking parameters
	MinStakeRequired *big.Int `json:"min_stake_required"`
	SlashingRate     float64  `json:"slashing_rate"`

	// Reputation parameters
	InitialReputation       float64 `json:"initial_reputation"`
	ReputationDecayRate     float64 `json:"reputation_decay_rate"`
	MinReputationMultiplier float64 `json:"min_reputation_multiplier"`
	MaxReputationMultiplier float64 `json:"max_reputation_multiplier"`

	// Contribution scoring
	ContributionWindow      time.Duration `json:"contribution_window"`
	MinContributionBonus    float64       `json:"min_contribution_bonus"`
	MaxContributionBonus    float64       `json:"max_contribution_bonus"`
	QualityThreshold        float64       `json:"quality_threshold"`
	InactivityPenalty       float64       `json:"inactivity_penalty"`
	NoRecentActivityPenalty float64       `json:"no_recent_activity_penalty"`

	// Proposer selection
	ProposerHistorySize     int `json:"proposer_history_size"`
	MaxProposerFrequency    int `json:"max_proposer_frequency"`
	ProposerFrequencyWindow int `json:"proposer_frequency_window"`
}

// ValidationConfig holds validation framework configuration
type ValidationConfig struct {
	// Thermal noise parameters
	ThermalNoiseLevel     float64 `json:"thermal_noise_level"`
	ThermalErrorTolerance float64 `json:"thermal_error_tolerance"`

	// Quantum parameters
	T1Time                   time.Duration `json:"t1_time"`
	T2Time                   time.Duration `json:"t2_time"`
	GateErrorRate            float64       `json:"gate_error_rate"`
	EnergyRelaxationRate     float64       `json:"energy_relaxation_rate"`
	QuantumFidelityThreshold float64       `json:"quantum_fidelity_threshold"`

	// Biological parameters
	PhenotypicNoise              float64 `json:"phenotypic_noise"`
	BiologicalConsensusThreshold float64 `json:"biological_consensus_threshold"`

	// Environment coupling
	EnvironmentCoupling float64 `json:"environment_coupling"`

	// Performance thresholds
	ConsensusLatencyThreshold int     `json:"consensus_latency_threshold"`
	MinThroughputRatio        float64 `json:"min_throughput_ratio"`

	// Scalability testing
	NodeCountTests   []int `json:"node_count_tests"`
	ScalabilityLimit int   `json:"scalability_limit"`

	// Hybrid mode
	HybridBiologicalPercent    float64 `json:"hybrid_biological_percent"`
	HybridReliabilityThreshold float64 `json:"hybrid_reliability_threshold"`
}

// ZKPConfig holds zero-knowledge proof configuration
type ZKPConfig struct {
	RangeSize      int `json:"range_size"`
	Precision      int `json:"precision"`
	ValueBitLength int `json:"value_bit_length"`
}

// NetworkConfig holds network configuration
type NetworkConfig struct {
	MaxPeers          int           `json:"max_peers"`
	ConnectionTimeout time.Duration `json:"connection_timeout"`
	HeartbeatInterval time.Duration `json:"heartbeat_interval"`
	MessageBufferSize int           `json:"message_buffer_size"`
}

// DefaultConsensusConfig returns default consensus configuration
func DefaultConsensusConfig() *ConsensusConfig {
	return &ConsensusConfig{
		BlockTime:               5 * time.Second,
		EpochLength:             100,
		MinStakeRequired:        big.NewInt(1000),
		SlashingRate:            0.1,
		InitialReputation:       5.0,
		ReputationDecayRate:     0.01,
		MinReputationMultiplier: 0.1,
		MaxReputationMultiplier: 2.0,
		ContributionWindow:      7 * 24 * time.Hour,
		MinContributionBonus:    0.8,
		MaxContributionBonus:    1.5,
		QualityThreshold:        75.0,
		InactivityPenalty:       0.8,
		NoRecentActivityPenalty: 0.9,
		ProposerHistorySize:     20,
		MaxProposerFrequency:    2,
		ProposerFrequencyWindow: 3,
	}
}

// DefaultValidationConfig returns default validation configuration
func DefaultValidationConfig() *ValidationConfig {
	return &ValidationConfig{
		ThermalNoiseLevel:            0.3,
		ThermalErrorTolerance:        0.5,
		T1Time:                       100 * time.Microsecond,
		T2Time:                       50 * time.Microsecond,
		GateErrorRate:                0.001,
		EnergyRelaxationRate:         1e5,
		QuantumFidelityThreshold:     0.5,
		PhenotypicNoise:              0.2,
		BiologicalConsensusThreshold: 0.7,
		EnvironmentCoupling:          0.1,
		ConsensusLatencyThreshold:    1000,
		MinThroughputRatio:           0.1,
		NodeCountTests:               []int{10, 100, 1000, 10000},
		ScalabilityLimit:             1000,
		HybridBiologicalPercent:      0.1,
		HybridReliabilityThreshold:   0.8,
	}
}

// DefaultZKPConfig returns default ZKP configuration
func DefaultZKPConfig() *ZKPConfig {
	return &ZKPConfig{
		RangeSize:      100,
		Precision:      100,
		ValueBitLength: 8,
	}
}

// DefaultNetworkConfig returns default network configuration
func DefaultNetworkConfig() *NetworkConfig {
	return &NetworkConfig{
		MaxPeers:          50,
		ConnectionTimeout: 30 * time.Second,
		HeartbeatInterval: 10 * time.Second,
		MessageBufferSize: 1000,
	}
}
