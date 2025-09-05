package errors

import (
	"fmt"
)

// ErrorCode represents a specific error type
type ErrorCode string

const (
	// Consensus errors
	ErrInsufficientStake   ErrorCode = "INSUFFICIENT_STAKE"
	ErrValidatorNotFound   ErrorCode = "VALIDATOR_NOT_FOUND"
	ErrNoActiveValidators  ErrorCode = "NO_ACTIVE_VALIDATORS"
	ErrProposerSelection   ErrorCode = "PROPOSER_SELECTION_FAILED"
	ErrInvalidContribution ErrorCode = "INVALID_CONTRIBUTION"

	// Network errors
	ErrConnectionFailed  ErrorCode = "CONNECTION_FAILED"
	ErrMessageSendFailed ErrorCode = "MESSAGE_SEND_FAILED"
	ErrPeerNotFound      ErrorCode = "PEER_NOT_FOUND"
	ErrNetworkTimeout    ErrorCode = "NETWORK_TIMEOUT"

	// Storage errors
	ErrStorageRead   ErrorCode = "STORAGE_READ_FAILED"
	ErrStorageWrite  ErrorCode = "STORAGE_WRITE_FAILED"
	ErrBlockNotFound ErrorCode = "BLOCK_NOT_FOUND"
	ErrCorruptedData ErrorCode = "CORRUPTED_DATA"

	// Validation errors
	ErrInvalidSignature      ErrorCode = "INVALID_SIGNATURE"
	ErrInvalidProof          ErrorCode = "INVALID_PROOF"
	ErrQualityBelowThreshold ErrorCode = "QUALITY_BELOW_THRESHOLD"

	// Configuration errors
	ErrInvalidConfig ErrorCode = "INVALID_CONFIGURATION"
	ErrMissingConfig ErrorCode = "MISSING_CONFIGURATION"
)

// ConsensusError represents a consensus-related error
type ConsensusError struct {
	Code    ErrorCode
	Message string
	Details map[string]interface{}
}

func (e *ConsensusError) Error() string {
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

// NewConsensusError creates a new consensus error
func NewConsensusError(code ErrorCode, message string) *ConsensusError {
	return &ConsensusError{
		Code:    code,
		Message: message,
		Details: make(map[string]interface{}),
	}
}

// WithDetails adds details to the error
func (e *ConsensusError) WithDetails(key string, value interface{}) *ConsensusError {
	e.Details[key] = value
	return e
}

// NetworkError represents a network-related error
type NetworkError struct {
	Code    ErrorCode
	Message string
	PeerID  string
	Err     error
}

func (e *NetworkError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

func (e *NetworkError) Unwrap() error {
	return e.Err
}

// NewNetworkError creates a new network error
func NewNetworkError(code ErrorCode, message string, err error) *NetworkError {
	return &NetworkError{
		Code:    code,
		Message: message,
		Err:     err,
	}
}

// StorageError represents a storage-related error
type StorageError struct {
	Code    ErrorCode
	Message string
	Key     string
	Err     error
}

func (e *StorageError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("[%s] %s (key: %s): %v", e.Code, e.Message, e.Key, e.Err)
	}
	return fmt.Sprintf("[%s] %s (key: %s)", e.Code, e.Message, e.Key)
}

func (e *StorageError) Unwrap() error {
	return e.Err
}

// NewStorageError creates a new storage error
func NewStorageError(code ErrorCode, message string, key string, err error) *StorageError {
	return &StorageError{
		Code:    code,
		Message: message,
		Key:     key,
		Err:     err,
	}
}

// ValidationError represents a validation-related error
type ValidationError struct {
	Code    ErrorCode
	Message string
	Field   string
	Value   interface{}
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("[%s] %s (field: %s, value: %v)", e.Code, e.Message, e.Field, e.Value)
}

// NewValidationError creates a new validation error
func NewValidationError(code ErrorCode, message string, field string, value interface{}) *ValidationError {
	return &ValidationError{
		Code:    code,
		Message: message,
		Field:   field,
		Value:   value,
	}
}

// IsErrorCode checks if an error has a specific error code
func IsErrorCode(err error, code ErrorCode) bool {
	switch e := err.(type) {
	case *ConsensusError:
		return e.Code == code
	case *NetworkError:
		return e.Code == code
	case *StorageError:
		return e.Code == code
	case *ValidationError:
		return e.Code == code
	default:
		return false
	}
}
