package vm

import (
	"math/big"
	"sync"
)

// StateDB represents the state database for contracts
type StateDB struct {
	mu sync.RWMutex

	// Account states
	balances map[string]*big.Int
	nonces   map[string]uint64
	code     map[string][]byte
	codeHash map[string][]byte

	// Contract storage
	storage map[string]map[string][]byte

	// Logs
	logs []*Log

	// Validator states
	stakes      map[string]*big.Int
	reputation  map[string]*big.Int
	contributions map[string]map[uint64]*big.Int

	// Marked for deletion
	deleted map[string]bool

	// Snapshot for rollback
	snapshot *StateSnapshot
}

// StateSnapshot represents a snapshot of the state
type StateSnapshot struct {
	balances      map[string]*big.Int
	nonces        map[string]uint64
	storage       map[string]map[string][]byte
	stakes        map[string]*big.Int
	reputation    map[string]*big.Int
	contributions map[string]map[uint64]*big.Int
	deleted       map[string]bool
}

// Log represents an event log
type Log struct {
	Address string
	Topics  [][]byte
	Data    []byte
}

// NewStateDB creates a new state database
func NewStateDB() *StateDB {
	return &StateDB{
		balances:      make(map[string]*big.Int),
		nonces:        make(map[string]uint64),
		code:          make(map[string][]byte),
		codeHash:      make(map[string][]byte),
		storage:       make(map[string]map[string][]byte),
		logs:          make([]*Log, 0),
		stakes:        make(map[string]*big.Int),
		reputation:    make(map[string]*big.Int),
		contributions: make(map[string]map[uint64]*big.Int),
		deleted:       make(map[string]bool),
	}
}

// Balance operations
func (s *StateDB) GetBalance(address string) *big.Int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if balance, exists := s.balances[address]; exists {
		return new(big.Int).Set(balance)
	}
	return new(big.Int)
}

func (s *StateDB) SetBalance(address string, amount *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.balances[address] = new(big.Int).Set(amount)
}

func (s *StateDB) AddBalance(address string, amount *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if balance, exists := s.balances[address]; exists {
		s.balances[address] = new(big.Int).Add(balance, amount)
	} else {
		s.balances[address] = new(big.Int).Set(amount)
	}
}

func (s *StateDB) SubBalance(address string, amount *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if balance, exists := s.balances[address]; exists {
		s.balances[address] = new(big.Int).Sub(balance, amount)
		if s.balances[address].Sign() < 0 {
			s.balances[address] = new(big.Int)
		}
	}
}

// Nonce operations
func (s *StateDB) GetNonce(address string) uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.nonces[address]
}

func (s *StateDB) SetNonce(address string, nonce uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.nonces[address] = nonce
}

func (s *StateDB) IncNonce(address string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.nonces[address]++
}

// Code operations
func (s *StateDB) GetCode(address string) []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if code, exists := s.code[address]; exists {
		result := make([]byte, len(code))
		copy(result, code)
		return result
	}
	return nil
}

func (s *StateDB) SetCode(address string, code []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.code[address] = make([]byte, len(code))
	copy(s.code[address], code)
}

func (s *StateDB) GetCodeHash(address string) []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if hash, exists := s.codeHash[address]; exists {
		result := make([]byte, len(hash))
		copy(result, hash)
		return result
	}
	return nil
}

func (s *StateDB) SetCodeHash(address string, hash []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.codeHash[address] = make([]byte, len(hash))
	copy(s.codeHash[address], hash)
}

// Storage operations
func (s *StateDB) GetState(address string, key []byte) []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()

	keyStr := string(key)
	if addressStorage, exists := s.storage[address]; exists {
		if value, exists := addressStorage[keyStr]; exists {
			result := make([]byte, len(value))
			copy(result, value)
			return result
		}
	}
	return nil
}

func (s *StateDB) SetState(address string, key []byte, value []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()

	keyStr := string(key)
	if s.storage[address] == nil {
		s.storage[address] = make(map[string][]byte)
	}

	if len(value) == 0 {
		delete(s.storage[address], keyStr)
	} else {
		s.storage[address][keyStr] = make([]byte, len(value))
		copy(s.storage[address][keyStr], value)
	}
}

// Log operations
func (s *StateDB) AddLog(log *Log) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.logs = append(s.logs, log)
}

func (s *StateDB) GetLogs() []*Log {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]*Log, len(s.logs))
	copy(result, s.logs)
	return result
}

func (s *StateDB) ClearLogs() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.logs = s.logs[:0]
}

// Stake operations
func (s *StateDB) GetStake(address string) *big.Int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if stake, exists := s.stakes[address]; exists {
		return new(big.Int).Set(stake)
	}
	return new(big.Int)
}

func (s *StateDB) AddStake(address string, amount *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if stake, exists := s.stakes[address]; exists {
		s.stakes[address] = new(big.Int).Add(stake, amount)
	} else {
		s.stakes[address] = new(big.Int).Set(amount)
	}
}

func (s *StateDB) SubStake(address string, amount *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if stake, exists := s.stakes[address]; exists {
		s.stakes[address] = new(big.Int).Sub(stake, amount)
		if s.stakes[address].Sign() < 0 {
			s.stakes[address] = new(big.Int)
		}
	}
}

// Reputation operations
func (s *StateDB) GetReputation(address string) *big.Int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if reputation, exists := s.reputation[address]; exists {
		return new(big.Int).Set(reputation)
	}
	return new(big.Int)
}

func (s *StateDB) SetReputation(address string, reputation *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.reputation[address] = new(big.Int).Set(reputation)
}

func (s *StateDB) AddReputation(address string, amount *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if reputation, exists := s.reputation[address]; exists {
		s.reputation[address] = new(big.Int).Add(reputation, amount)
	} else {
		s.reputation[address] = new(big.Int).Set(amount)
	}
}

// Contribution operations
func (s *StateDB) AddContribution(address string, contributionType uint64, amount *big.Int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.contributions[address] == nil {
		s.contributions[address] = make(map[uint64]*big.Int)
	}

	if existing, exists := s.contributions[address][contributionType]; exists {
		s.contributions[address][contributionType] = new(big.Int).Add(existing, amount)
	} else {
		s.contributions[address][contributionType] = new(big.Int).Set(amount)
	}
}

func (s *StateDB) GetContribution(address string, contributionType uint64) *big.Int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if contributions, exists := s.contributions[address]; exists {
		if amount, exists := contributions[contributionType]; exists {
			return new(big.Int).Set(amount)
		}
	}
	return new(big.Int)
}

func (s *StateDB) GetTotalContributions(address string) *big.Int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := new(big.Int)
	if contributions, exists := s.contributions[address]; exists {
		for _, amount := range contributions {
			total = total.Add(total, amount)
		}
	}
	return total
}

// Deletion operations
func (s *StateDB) MarkForDeletion(address string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.deleted[address] = true
}

func (s *StateDB) IsDeleted(address string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.deleted[address]
}

// Validation operations
func (s *StateDB) ValidateBlock(height uint64, hash []byte) bool {
	// Simplified validation - always return true for now
	// In production, this would check against stored block data
	return true
}

// Snapshot operations
func (s *StateDB) Snapshot() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.snapshot = &StateSnapshot{
		balances:      make(map[string]*big.Int),
		nonces:        make(map[string]uint64),
		storage:       make(map[string]map[string][]byte),
		stakes:        make(map[string]*big.Int),
		reputation:    make(map[string]*big.Int),
		contributions: make(map[string]map[uint64]*big.Int),
		deleted:       make(map[string]bool),
	}

	// Copy balances
	for addr, balance := range s.balances {
		s.snapshot.balances[addr] = new(big.Int).Set(balance)
	}

	// Copy nonces
	for addr, nonce := range s.nonces {
		s.snapshot.nonces[addr] = nonce
	}

	// Copy storage
	for addr, addrStorage := range s.storage {
		s.snapshot.storage[addr] = make(map[string][]byte)
		for key, value := range addrStorage {
			s.snapshot.storage[addr][key] = make([]byte, len(value))
			copy(s.snapshot.storage[addr][key], value)
		}
	}

	// Copy stakes
	for addr, stake := range s.stakes {
		s.snapshot.stakes[addr] = new(big.Int).Set(stake)
	}

	// Copy reputation
	for addr, rep := range s.reputation {
		s.snapshot.reputation[addr] = new(big.Int).Set(rep)
	}

	// Copy contributions
	for addr, contribs := range s.contributions {
		s.snapshot.contributions[addr] = make(map[uint64]*big.Int)
		for cType, amount := range contribs {
			s.snapshot.contributions[addr][cType] = new(big.Int).Set(amount)
		}
	}

	// Copy deleted
	for addr, deleted := range s.deleted {
		s.snapshot.deleted[addr] = deleted
	}
}

func (s *StateDB) Rollback() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.snapshot == nil {
		return
	}

	// Restore from snapshot
	s.balances = s.snapshot.balances
	s.nonces = s.snapshot.nonces
	s.storage = s.snapshot.storage
	s.stakes = s.snapshot.stakes
	s.reputation = s.snapshot.reputation
	s.contributions = s.snapshot.contributions
	s.deleted = s.snapshot.deleted

	s.snapshot = nil
}

func (s *StateDB) CommitSnapshot() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.snapshot = nil
}

// Account existence
func (s *StateDB) Exists(address string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.deleted[address] {
		return false
	}

	// Account exists if it has balance, nonce, code, or storage
	if _, exists := s.balances[address]; exists {
		return true
	}
	if _, exists := s.nonces[address]; exists && s.nonces[address] > 0 {
		return true
	}
	if _, exists := s.code[address]; exists {
		return true
	}
	if _, exists := s.storage[address]; exists {
		return true
	}

	return false
}

// Empty returns true if the account has no balance, nonce, or code
func (s *StateDB) Empty(address string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.deleted[address] {
		return true
	}

	balance := s.balances[address]
	nonce := s.nonces[address]
	code := s.code[address]

	return (balance == nil || balance.Sign() == 0) &&
		nonce == 0 &&
		len(code) == 0
}

// Copy creates a deep copy of the state database
func (s *StateDB) Copy() *StateDB {
	s.mu.RLock()
	defer s.mu.RUnlock()

	newState := NewStateDB()

	// Copy balances
	for addr, balance := range s.balances {
		newState.balances[addr] = new(big.Int).Set(balance)
	}

	// Copy nonces
	for addr, nonce := range s.nonces {
		newState.nonces[addr] = nonce
	}

	// Copy code
	for addr, code := range s.code {
		newState.code[addr] = make([]byte, len(code))
		copy(newState.code[addr], code)
	}

	// Copy code hashes
	for addr, hash := range s.codeHash {
		newState.codeHash[addr] = make([]byte, len(hash))
		copy(newState.codeHash[addr], hash)
	}

	// Copy storage
	for addr, addrStorage := range s.storage {
		newState.storage[addr] = make(map[string][]byte)
		for key, value := range addrStorage {
			newState.storage[addr][key] = make([]byte, len(value))
			copy(newState.storage[addr][key], value)
		}
	}

	// Copy stakes
	for addr, stake := range s.stakes {
		newState.stakes[addr] = new(big.Int).Set(stake)
	}

	// Copy reputation
	for addr, rep := range s.reputation {
		newState.reputation[addr] = new(big.Int).Set(rep)
	}

	// Copy contributions
	for addr, contribs := range s.contributions {
		newState.contributions[addr] = make(map[uint64]*big.Int)
		for cType, amount := range contribs {
			newState.contributions[addr][cType] = new(big.Int).Set(amount)
		}
	}

	// Copy deleted
	for addr, deleted := range s.deleted {
		newState.deleted[addr] = deleted
	}

	// Copy logs
	newState.logs = make([]*Log, len(s.logs))
	copy(newState.logs, s.logs)

	return newState
}