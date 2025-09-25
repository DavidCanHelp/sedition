package validator

import (
	"math/big"
	"sync"
)

// ValidatorSet manages a set of validators
type ValidatorSet struct {
	mu         sync.RWMutex
	validators map[string]*Validator
}

// NewValidatorSet creates a new validator set
func NewValidatorSet() *ValidatorSet {
	return &ValidatorSet{
		validators: make(map[string]*Validator),
	}
}

// AddValidator adds a validator to the set
func (vs *ValidatorSet) AddValidator(v *Validator) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.validators[v.Address] = v
}

// RemoveValidator removes a validator from the set
func (vs *ValidatorSet) RemoveValidator(address string) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	delete(vs.validators, address)
}

// GetValidator retrieves a validator by address
func (vs *ValidatorSet) GetValidator(address string) (*Validator, bool) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	v, exists := vs.validators[address]
	return v, exists
}

// GetValidators returns all validators in the set
func (vs *ValidatorSet) GetValidators() []*Validator {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	vals := make([]*Validator, 0, len(vs.validators))
	for _, v := range vs.validators {
		vals = append(vals, v)
	}
	return vals
}

// Size returns the number of validators in the set
func (vs *ValidatorSet) Size() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.validators)
}

// TotalStake returns the total stake of all validators
func (vs *ValidatorSet) TotalStake() *big.Int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	total := big.NewInt(0)
	for _, v := range vs.validators {
		if v.TokenStake != nil {
			total.Add(total, v.TokenStake)
		}
	}
	return total
}

// GetActiveValidators returns only active validators
func (vs *ValidatorSet) GetActiveValidators() []*Validator {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	active := make([]*Validator, 0)
	for _, v := range vs.validators {
		if v.IsActive {
			active = append(active, v)
		}
	}
	return active
}

// CreateValidator is a helper function to create a new validator with public key
func CreateValidator(address, publicKey string, stake *big.Int) *Validator {
	return &Validator{
		Address:         address,
		PublicKey:       publicKey,
		TokenStake:      stake,
		ReputationScore: 5.0, // Default reputation
		IsActive:        true,
		TotalStake:      new(big.Int).Set(stake),
	}
}