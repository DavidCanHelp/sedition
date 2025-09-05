package config

import (
	"math/big"
	"testing"
	"time"
)

func TestDefaultConsensusConfig(t *testing.T) {
	cfg := DefaultConsensusConfig()
	
	if cfg == nil {
		t.Fatal("DefaultConsensusConfig returned nil")
	}
	
	// Test default values
	if cfg.BlockTime != 5*time.Second {
		t.Errorf("Expected BlockTime to be 5s, got %v", cfg.BlockTime)
	}
	
	if cfg.EpochLength != 100 {
		t.Errorf("Expected EpochLength to be 100, got %d", cfg.EpochLength)
	}
	
	expectedMinStake := big.NewInt(1000)
	if cfg.MinStakeRequired.Cmp(expectedMinStake) != 0 {
		t.Errorf("Expected MinStakeRequired to be %v, got %v", expectedMinStake, cfg.MinStakeRequired)
	}
	
	if cfg.SlashingRate != 0.1 {
		t.Errorf("Expected SlashingRate to be 0.1, got %f", cfg.SlashingRate)
	}
}

func TestDefaultNetworkConfig(t *testing.T) {
	cfg := DefaultNetworkConfig()
	
	if cfg == nil {
		t.Fatal("DefaultNetworkConfig returned nil")
	}
	
	// Test default values
	if cfg.MaxPeers != 50 {
		t.Errorf("Expected MaxPeers to be 50, got %d", cfg.MaxPeers)
	}
	
	if cfg.ConnectionTimeout != 30*time.Second {
		t.Errorf("Expected ConnectionTimeout to be 30s, got %v", cfg.ConnectionTimeout)
	}
	
	if cfg.HeartbeatInterval != 10*time.Second {
		t.Errorf("Expected HeartbeatInterval to be 10s, got %v", cfg.HeartbeatInterval)
	}
	
	if cfg.MessageBufferSize != 1000 {
		t.Errorf("Expected MessageBufferSize to be 1000, got %d", cfg.MessageBufferSize)
	}
}

func TestDefaultValidationConfig(t *testing.T) {
	cfg := DefaultValidationConfig()
	
	if cfg == nil {
		t.Fatal("DefaultValidationConfig returned nil")
	}
	
	// Test thermal noise parameters
	if cfg.ThermalNoiseLevel != 0.3 {
		t.Errorf("Expected ThermalNoiseLevel to be 0.3, got %f", cfg.ThermalNoiseLevel)
	}
	
	if cfg.ThermalErrorTolerance != 0.5 {
		t.Errorf("Expected ThermalErrorTolerance to be 0.5, got %f", cfg.ThermalErrorTolerance)
	}
	
	// Test quantum parameters
	if cfg.T1Time != 100*time.Microsecond {
		t.Errorf("Expected T1Time to be 100μs, got %v", cfg.T1Time)
	}
	
	if cfg.T2Time != 50*time.Microsecond {
		t.Errorf("Expected T2Time to be 50μs, got %v", cfg.T2Time)
	}
	
	if cfg.GateErrorRate != 0.001 {
		t.Errorf("Expected GateErrorRate to be 0.001, got %f", cfg.GateErrorRate)
	}
}

func TestDefaultZKPConfig(t *testing.T) {
	cfg := DefaultZKPConfig()
	
	if cfg == nil {
		t.Fatal("DefaultZKPConfig returned nil")
	}
	
	// Test ZKP settings
	if cfg.RangeSize != 100 {
		t.Errorf("Expected RangeSize to be 100, got %d", cfg.RangeSize)
	}
	
	if cfg.Precision != 100 {
		t.Errorf("Expected Precision to be 100, got %d", cfg.Precision)
	}
	
	if cfg.ValueBitLength != 8 {
		t.Errorf("Expected ValueBitLength to be 8, got %d", cfg.ValueBitLength)
	}
}

// Test that configs have reasonable default values
func TestConfigSanity(t *testing.T) {
	t.Run("ConsensusConfig", func(t *testing.T) {
		cfg := DefaultConsensusConfig()
		
		// Block time should be reasonable
		if cfg.BlockTime < time.Second || cfg.BlockTime > time.Minute {
			t.Errorf("BlockTime seems unreasonable: %v", cfg.BlockTime)
		}
		
		// Epoch length should be positive
		if cfg.EpochLength <= 0 {
			t.Errorf("EpochLength should be positive, got %d", cfg.EpochLength)
		}
		
		// Slashing rate should be between 0 and 1
		if cfg.SlashingRate < 0 || cfg.SlashingRate > 1 {
			t.Errorf("SlashingRate should be between 0 and 1, got %f", cfg.SlashingRate)
		}
	})
	
	t.Run("NetworkConfig", func(t *testing.T) {
		cfg := DefaultNetworkConfig()
		
		// Max peers should be reasonable
		if cfg.MaxPeers <= 0 || cfg.MaxPeers > 10000 {
			t.Errorf("MaxPeers seems unreasonable: %d", cfg.MaxPeers)
		}
		
		// Connection timeout should be reasonable
		if cfg.ConnectionTimeout <= 0 || cfg.ConnectionTimeout > time.Hour {
			t.Errorf("ConnectionTimeout seems unreasonable: %v", cfg.ConnectionTimeout)
		}
	})
	
	t.Run("ValidationConfig", func(t *testing.T) {
		cfg := DefaultValidationConfig()
		
		// Thermal noise should be small
		if cfg.ThermalNoiseLevel < 0 || cfg.ThermalNoiseLevel > 1 {
			t.Errorf("ThermalNoiseLevel should be between 0 and 1, got %f", cfg.ThermalNoiseLevel)
		}
		
		// Error rates should be reasonable
		if cfg.GateErrorRate < 0 || cfg.GateErrorRate > 0.1 {
			t.Errorf("GateErrorRate should be between 0 and 0.1, got %f", cfg.GateErrorRate)
		}
	})
}

// Benchmark tests
func BenchmarkDefaultConsensusConfig(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DefaultConsensusConfig()
	}
}

func BenchmarkDefaultNetworkConfig(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DefaultNetworkConfig()
	}
}

func BenchmarkDefaultValidationConfig(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DefaultValidationConfig()
	}
}

func BenchmarkDefaultZKPConfig(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DefaultZKPConfig()
	}
}