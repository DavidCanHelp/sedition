package errors

import (
	"errors"
	"strings"
	"testing"
)

func TestNewConsensusError(t *testing.T) {
	err := NewConsensusError(ErrInsufficientStake, "stake too low")
	
	if err == nil {
		t.Fatal("Expected non-nil error")
	}
	
	if err.Code != ErrInsufficientStake {
		t.Errorf("Expected code %s, got %s", ErrInsufficientStake, err.Code)
	}
	
	if err.Message != "stake too low" {
		t.Errorf("Expected message 'stake too low', got %s", err.Message)
	}
}

func TestConsensusErrorError(t *testing.T) {
	err := NewConsensusError(ErrValidatorNotFound, "validator not found")
	
	errStr := err.Error()
	if !strings.Contains(errStr, "VALIDATOR_NOT_FOUND") {
		t.Errorf("Error string should contain error code, got: %s", errStr)
	}
	
	if !strings.Contains(errStr, "validator not found") {
		t.Errorf("Error string should contain message, got: %s", errStr)
	}
}

func TestNewNetworkError(t *testing.T) {
	cause := errors.New("connection timeout")
	err := NewNetworkError(ErrConnectionFailed, "failed to connect", cause)
	
	if err == nil {
		t.Fatal("Expected non-nil error")
	}
	
	if err.Code != ErrConnectionFailed {
		t.Errorf("Expected code %s, got %s", ErrConnectionFailed, err.Code)
	}
	
	if err.Message != "failed to connect" {
		t.Errorf("Expected message 'failed to connect', got %s", err.Message)
	}
	
	if err.Err == nil || err.Err.Error() != "connection timeout" {
		t.Errorf("Expected cause 'connection timeout', got %v", err.Err)
	}
}

func TestNewStorageError(t *testing.T) {
	cause := errors.New("disk full")
	err := NewStorageError(ErrStorageWrite, "write failed", "test.db", cause)
	
	if err == nil {
		t.Fatal("Expected non-nil error")
	}
	
	if err.Code != ErrStorageWrite {
		t.Errorf("Expected code %s, got %s", ErrStorageWrite, err.Code)
	}
	
	if err.Message != "write failed" {
		t.Errorf("Expected message 'write failed', got %s", err.Message)
	}
	
	if err.Key != "test.db" {
		t.Errorf("Expected key 'test.db', got %s", err.Key)
	}
	
	if err.Err == nil || err.Err.Error() != "disk full" {
		t.Errorf("Expected cause 'disk full', got %v", err.Err)
	}
}

func TestNewValidationError(t *testing.T) {
	err := NewValidationError(ErrInvalidSignature, "invalid input", "amount", -1)
	
	if err == nil {
		t.Fatal("Expected non-nil error")
	}
	
	if err.Code != ErrInvalidSignature {
		t.Errorf("Expected code %s, got %s", ErrInvalidSignature, err.Code)
	}
	
	if err.Message != "invalid input" {
		t.Errorf("Expected message 'invalid input', got %s", err.Message)
	}
	
	if err.Field != "amount" {
		t.Errorf("Expected field 'amount', got %s", err.Field)
	}
	
	if err.Value != -1 {
		t.Errorf("Expected value -1, got %v", err.Value)
	}
}

func TestErrorCodeString(t *testing.T) {
	tests := []struct {
		code     ErrorCode
		expected string
	}{
		{ErrInsufficientStake, "INSUFFICIENT_STAKE"},
		{ErrValidatorNotFound, "VALIDATOR_NOT_FOUND"},
		{ErrNoActiveValidators, "NO_ACTIVE_VALIDATORS"},
		{ErrProposerSelection, "PROPOSER_SELECTION_FAILED"},
		{ErrInvalidContribution, "INVALID_CONTRIBUTION"},
		{ErrConnectionFailed, "CONNECTION_FAILED"},
		{ErrMessageSendFailed, "MESSAGE_SEND_FAILED"},
		{ErrPeerNotFound, "PEER_NOT_FOUND"},
		{ErrNetworkTimeout, "NETWORK_TIMEOUT"},
		{ErrStorageRead, "STORAGE_READ_FAILED"},
		{ErrStorageWrite, "STORAGE_WRITE_FAILED"},
		{ErrBlockNotFound, "BLOCK_NOT_FOUND"},
		{ErrCorruptedData, "CORRUPTED_DATA"},
		{ErrInvalidSignature, "INVALID_SIGNATURE"},
		{ErrInvalidProof, "INVALID_PROOF"},
		{ErrQualityBelowThreshold, "QUALITY_BELOW_THRESHOLD"},
		{ErrInvalidConfig, "INVALID_CONFIGURATION"},
		{ErrMissingConfig, "MISSING_CONFIGURATION"},
	}
	
	for _, tt := range tests {
		t.Run(string(tt.code), func(t *testing.T) {
			if string(tt.code) != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, string(tt.code))
			}
		})
	}
}

// Test error wrapping and unwrapping
func TestErrorWrapping(t *testing.T) {
	baseErr := errors.New("base error")
	netErr := NewNetworkError(ErrConnectionFailed, "connection failed", baseErr)
	
	// Check that we can access the underlying error
	if netErr.Err != baseErr {
		t.Error("Expected to find base error as Err field")
	}
	
	// Check error string contains both messages
	errStr := netErr.Error()
	if !strings.Contains(errStr, "CONNECTION_FAILED") {
		t.Error("Error string should contain error code")
	}
	if !strings.Contains(errStr, "connection failed") {
		t.Error("Error string should contain message")
	}
}

// Test error comparison
func TestErrorComparison(t *testing.T) {
	err1 := NewConsensusError(ErrInsufficientStake, "stake too low")
	err2 := NewConsensusError(ErrInsufficientStake, "stake too low")
	err3 := NewConsensusError(ErrValidatorNotFound, "not found")
	
	// Same error code should be comparable
	if err1.Code != err2.Code {
		t.Error("Same error codes should be equal")
	}
	
	// Different error codes should not be equal
	if err1.Code == err3.Code {
		t.Error("Different error codes should not be equal")
	}
}

// Test WithDetails for ConsensusError
func TestConsensusErrorWithDetails(t *testing.T) {
	err := NewConsensusError(ErrInsufficientStake, "stake too low")
	err.WithDetails("minStake", 1000)
	err.WithDetails("actualStake", 500)
	
	if err.Details["minStake"] != 1000 {
		t.Errorf("Expected minStake detail to be 1000, got %v", err.Details["minStake"])
	}
	
	if err.Details["actualStake"] != 500 {
		t.Errorf("Expected actualStake detail to be 500, got %v", err.Details["actualStake"])
	}
}

// Benchmark tests
func BenchmarkNewConsensusError(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewConsensusError(ErrInsufficientStake, "test")
	}
}

func BenchmarkNewNetworkError(b *testing.B) {
	cause := errors.New("test")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewNetworkError(ErrConnectionFailed, "test", cause)
	}
}

func BenchmarkNewStorageError(b *testing.B) {
	cause := errors.New("test")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewStorageError(ErrStorageWrite, "test", "key", cause)
	}
}

func BenchmarkNewValidationError(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewValidationError(ErrInvalidSignature, "test", "field", i)
	}
}

func BenchmarkErrorString(b *testing.B) {
	err := NewConsensusError(ErrInsufficientStake, "test")
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = err.Error()
	}
}