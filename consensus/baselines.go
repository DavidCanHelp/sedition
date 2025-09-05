// Package consensus implements baseline consensus algorithms for comparison with PoC
package consensus

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"
	"sort"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/crypto"
)

// ConsensusAlgorithm defines the interface for consensus algorithms
type ConsensusAlgorithm interface {
	// Core consensus operations
	SelectLeader() (string, error)
	ProposeBlock(proposer string, transactions []Transaction) (*Block, error)
	ValidateBlock(block *Block) error
	FinalizeBlock(block *Block) error

	// State management
	AddValidator(id string, stake *big.Int) error
	RemoveValidator(id string) error
	GetValidators() map[string]*Validator

	// Metrics
	GetMetrics() *ConsensusMetrics
	Reset() error
}

// Block represents a block in any consensus algorithm
type Block struct {
	Height       int64
	PrevHash     []byte
	Timestamp    time.Time
	Proposer     string
	Transactions []Transaction
	Nonce        uint64   // For PoW
	Difficulty   *big.Int // For PoW
	VRFProof     []byte   // For PoS/PoC
	Hash         []byte
	Signatures   map[string][]byte
}

// Transaction represents a transaction/contribution
type Transaction struct {
	ID        string
	From      string
	Data      []byte
	Timestamp time.Time
	Size      int
}

// Validator represents a validator in any consensus system
type Validator struct {
	ID           string
	Stake        *big.Int
	PublicKey    []byte
	Reputation   float64 // For PoC
	Contribution float64 // For PoC
	LastActive   time.Time
	IsActive     bool
}

// ConsensusMetrics contains performance metrics
type ConsensusMetrics struct {
	Algorithm             string
	BlocksProduced        int64
	TotalTransactions     int64
	AverageBlockTime      time.Duration
	ThroughputTPS         float64
	EnergyConsumption     float64 // Relative units
	DecentralizationIndex float64
	FinalityTime          time.Duration
	NetworkOverhead       int64 // Messages sent
}

// =============================================================================
// Proof of Work Implementation
// =============================================================================

// ProofOfWork implements Bitcoin-style Proof of Work consensus
type ProofOfWork struct {
	mu sync.RWMutex

	validators      map[string]*Validator
	blockchain      []*Block
	targetBlockTime time.Duration
	difficulty      *big.Int
	maxDifficulty   *big.Int

	// Mining state
	currentTarget  *big.Int
	lastAdjustment time.Time

	// Metrics
	startTime      time.Time
	totalHashrate  float64
	energyConsumed float64
	blocksProduced int64
}

// NewProofOfWork creates a new PoW consensus engine
func NewProofOfWork(targetBlockTime time.Duration) *ProofOfWork {
	// Initial difficulty (4 leading zeros)
	difficulty := big.NewInt(1)
	difficulty.Lsh(difficulty, 252) // 2^252

	return &ProofOfWork{
		validators:      make(map[string]*Validator),
		blockchain:      make([]*Block, 0),
		targetBlockTime: targetBlockTime,
		difficulty:      difficulty,
		maxDifficulty:   new(big.Int).Lsh(big.NewInt(1), 248), // 2^248
		currentTarget:   difficulty,
		lastAdjustment:  time.Now(),
		startTime:       time.Now(),
	}
}

// SelectLeader in PoW, anyone can mine (return random validator)
func (pow *ProofOfWork) SelectLeader() (string, error) {
	pow.mu.RLock()
	defer pow.mu.RUnlock()

	if len(pow.validators) == 0 {
		return "", errors.New("no validators available")
	}

	// In PoW, any miner can propose - select random for simulation
	validators := make([]string, 0, len(pow.validators))
	for id := range pow.validators {
		validators = append(validators, id)
	}

	// Simple random selection for simulation
	randomBytes := make([]byte, 8)
	rand.Read(randomBytes)
	index := binary.BigEndian.Uint64(randomBytes) % uint64(len(validators))

	return validators[index], nil
}

// ProposeBlock creates a new PoW block
func (pow *ProofOfWork) ProposeBlock(proposer string, transactions []Transaction) (*Block, error) {
	pow.mu.Lock()
	defer pow.mu.Unlock()

	validator, exists := pow.validators[proposer]
	if !exists {
		return nil, errors.New("proposer not found")
	}

	// Get previous block hash
	var prevHash []byte
	height := int64(0)
	if len(pow.blockchain) > 0 {
		prevBlock := pow.blockchain[len(pow.blockchain)-1]
		prevHash = prevBlock.Hash
		height = prevBlock.Height + 1
	}

	block := &Block{
		Height:       height,
		PrevHash:     prevHash,
		Timestamp:    time.Now(),
		Proposer:     proposer,
		Transactions: transactions,
		Difficulty:   new(big.Int).Set(pow.currentTarget),
		Signatures:   make(map[string][]byte),
	}

	// Mine the block (find nonce that satisfies difficulty)
	if err := pow.mineBlock(block); err != nil {
		return nil, err
	}

	validator.LastActive = time.Now()
	return block, nil
}

// mineBlock performs proof of work mining
func (pow *ProofOfWork) mineBlock(block *Block) error {
	target := block.Difficulty
	maxNonce := uint64(1<<32) - 1 // 32-bit nonce space

	startTime := time.Now()

	for nonce := uint64(0); nonce <= maxNonce; nonce++ {
		block.Nonce = nonce
		hash := pow.calculateBlockHash(block)

		// Check if hash meets difficulty target
		hashInt := new(big.Int).SetBytes(hash)
		if hashInt.Cmp(target) <= 0 {
			block.Hash = hash

			// Update energy consumption (simplified model)
			miningTime := time.Since(startTime)
			pow.energyConsumed += float64(nonce) * 1e-6 // Simplified energy model
			pow.totalHashrate += float64(nonce) / miningTime.Seconds()

			return nil
		}

		// Timeout check (prevent infinite mining)
		if time.Since(startTime) > pow.targetBlockTime*10 {
			return errors.New("mining timeout")
		}
	}

	return errors.New("failed to find valid nonce")
}

// ValidateBlock validates a PoW block
func (pow *ProofOfWork) ValidateBlock(block *Block) error {
	// Verify hash meets difficulty
	hashInt := new(big.Int).SetBytes(block.Hash)
	if hashInt.Cmp(block.Difficulty) > 0 {
		return errors.New("block hash does not meet difficulty target")
	}

	// Verify hash is correct
	expectedHash := pow.calculateBlockHash(block)
	if !equalBytes(expectedHash, block.Hash) {
		return errors.New("invalid block hash")
	}

	// Verify previous block hash
	if len(pow.blockchain) > 0 {
		prevBlock := pow.blockchain[len(pow.blockchain)-1]
		if !equalBytes(block.PrevHash, prevBlock.Hash) {
			return errors.New("invalid previous block hash")
		}
	}

	return nil
}

// FinalizeBlock adds block to PoW chain
func (pow *ProofOfWork) FinalizeBlock(block *Block) error {
	pow.mu.Lock()
	defer pow.mu.Unlock()

	pow.blockchain = append(pow.blockchain, block)
	pow.blocksProduced++

	// Adjust difficulty every 10 blocks
	if len(pow.blockchain)%10 == 0 {
		pow.adjustDifficulty()
	}

	return nil
}

// adjustDifficulty adjusts mining difficulty based on block times
func (pow *ProofOfWork) adjustDifficulty() {
	if len(pow.blockchain) < 10 {
		return
	}

	// Calculate average block time over last 10 blocks
	recent := pow.blockchain[len(pow.blockchain)-10:]
	totalTime := recent[9].Timestamp.Sub(recent[0].Timestamp)
	avgBlockTime := totalTime / 9

	// Adjust difficulty
	targetTime := pow.targetBlockTime * 9
	if avgBlockTime < targetTime {
		// Blocks too fast, increase difficulty
		pow.currentTarget.Mul(pow.currentTarget, big.NewInt(9))
		pow.currentTarget.Div(pow.currentTarget, big.NewInt(10))
	} else {
		// Blocks too slow, decrease difficulty
		pow.currentTarget.Mul(pow.currentTarget, big.NewInt(11))
		pow.currentTarget.Div(pow.currentTarget, big.NewInt(10))
	}

	// Clamp difficulty
	if pow.currentTarget.Cmp(pow.maxDifficulty) > 0 {
		pow.currentTarget.Set(pow.maxDifficulty)
	}
	if pow.currentTarget.Cmp(big.NewInt(1)) < 0 {
		pow.currentTarget.Set(big.NewInt(1))
	}
}

// calculateBlockHash computes the hash of a PoW block
func (pow *ProofOfWork) calculateBlockHash(block *Block) []byte {
	data := make([]byte, 0)

	// Height
	heightBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(heightBytes, uint64(block.Height))
	data = append(data, heightBytes...)

	// Previous hash
	data = append(data, block.PrevHash...)

	// Timestamp
	timeBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(timeBytes, uint64(block.Timestamp.Unix()))
	data = append(data, timeBytes...)

	// Transactions (simplified)
	for _, tx := range block.Transactions {
		data = append(data, []byte(tx.ID)...)
	}

	// Nonce
	nonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(nonceBytes, block.Nonce)
	data = append(data, nonceBytes...)

	hash := sha256.Sum256(data)
	return hash[:]
}

// AddValidator adds a PoW miner
func (pow *ProofOfWork) AddValidator(id string, stake *big.Int) error {
	pow.mu.Lock()
	defer pow.mu.Unlock()

	pow.validators[id] = &Validator{
		ID:         id,
		Stake:      new(big.Int).Set(stake),
		LastActive: time.Now(),
		IsActive:   true,
	}

	return nil
}

// RemoveValidator removes a PoW miner
func (pow *ProofOfWork) RemoveValidator(id string) error {
	pow.mu.Lock()
	defer pow.mu.Unlock()

	delete(pow.validators, id)
	return nil
}

// GetValidators returns PoW validators
func (pow *ProofOfWork) GetValidators() map[string]*Validator {
	pow.mu.RLock()
	defer pow.mu.RUnlock()

	validators := make(map[string]*Validator)
	for k, v := range pow.validators {
		validators[k] = v
	}
	return validators
}

// GetMetrics returns PoW metrics
func (pow *ProofOfWork) GetMetrics() *ConsensusMetrics {
	pow.mu.RLock()
	defer pow.mu.RUnlock()

	uptime := time.Since(pow.startTime)
	avgBlockTime := time.Duration(0)
	if pow.blocksProduced > 0 {
		avgBlockTime = uptime / time.Duration(pow.blocksProduced)
	}

	return &ConsensusMetrics{
		Algorithm:             "Proof of Work",
		BlocksProduced:        pow.blocksProduced,
		TotalTransactions:     pow.getTotalTransactions(),
		AverageBlockTime:      avgBlockTime,
		ThroughputTPS:         float64(pow.getTotalTransactions()) / uptime.Seconds(),
		EnergyConsumption:     pow.energyConsumed,
		DecentralizationIndex: pow.calculateDecentralization(),
		FinalityTime:          avgBlockTime * 6, // 6 confirmations
		NetworkOverhead:       0,                // Minimal in PoW
	}
}

// Reset resets PoW state
func (pow *ProofOfWork) Reset() error {
	pow.mu.Lock()
	defer pow.mu.Unlock()

	pow.blockchain = pow.blockchain[:0]
	pow.blocksProduced = 0
	pow.energyConsumed = 0
	pow.startTime = time.Now()

	return nil
}

// =============================================================================
// Proof of Stake Implementation
// =============================================================================

// ProofOfStake implements Ethereum-style Proof of Stake consensus
type ProofOfStake struct {
	mu sync.RWMutex

	validators map[string]*Validator
	blockchain []*Block
	totalStake *big.Int
	minStake   *big.Int

	// Epoch management
	currentEpoch  uint64
	epochLength   uint64
	slotsPerEpoch uint64

	// VRF for randomness
	vrfs      map[string]*crypto.VRF
	epochSeed []byte

	// Metrics
	startTime      time.Time
	blocksProduced int64
	slashingEvents int64
}

// NewProofOfStake creates a new PoS consensus engine
func NewProofOfStake(minStake *big.Int, epochLength uint64) *ProofOfStake {
	seed := make([]byte, 32)
	rand.Read(seed)

	return &ProofOfStake{
		validators:    make(map[string]*Validator),
		blockchain:    make([]*Block, 0),
		totalStake:    big.NewInt(0),
		minStake:      new(big.Int).Set(minStake),
		epochLength:   epochLength,
		slotsPerEpoch: epochLength,
		vrfs:          make(map[string]*crypto.VRF),
		epochSeed:     seed,
		startTime:     time.Now(),
	}
}

// SelectLeader uses VRF-based selection weighted by stake
func (pos *ProofOfStake) SelectLeader() (string, error) {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	if len(pos.validators) == 0 {
		return "", errors.New("no validators available")
	}

	// Use VRF-based sortition
	slot := uint64(len(pos.blockchain))

	var bestValidator string
	var bestValue *big.Int

	for id, validator := range pos.validators {
		if !validator.IsActive {
			continue
		}

		vrf, exists := pos.vrfs[id]
		if !exists {
			continue
		}

		// Create sortition message
		message := make([]byte, 0, 32+8+8)
		message = append(message, pos.epochSeed...)

		epochBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(epochBytes, pos.currentEpoch)
		message = append(message, epochBytes...)

		slotBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(slotBytes, slot)
		message = append(message, slotBytes...)

		// Generate VRF proof
		proof, err := crypto.Sortition(vrf, pos.epochSeed, slot, "proposer",
			validator.Stake, pos.totalStake, 1)
		if err != nil {
			continue
		}

		// Check if selected
		if proof.J > 0 {
			value := proof.VRFOutput.GetRandomness()
			if bestValidator == "" || value.Cmp(bestValue) < 0 {
				bestValidator = id
				bestValue = value
			}
		}
	}

	if bestValidator == "" {
		return "", errors.New("no validator selected")
	}

	return bestValidator, nil
}

// ProposeBlock creates a new PoS block
func (pos *ProofOfStake) ProposeBlock(proposer string, transactions []Transaction) (*Block, error) {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	validator, exists := pos.validators[proposer]
	if !exists {
		return nil, errors.New("proposer not found")
	}

	// Get previous block hash
	var prevHash []byte
	height := int64(0)
	if len(pos.blockchain) > 0 {
		prevBlock := pos.blockchain[len(pos.blockchain)-1]
		prevHash = prevBlock.Hash
		height = prevBlock.Height + 1
	}

	// Generate VRF proof
	vrf := pos.vrfs[proposer]
	message := append(pos.epochSeed, byte(height))
	vrfOutput, err := vrf.Prove(message)
	if err != nil {
		return nil, fmt.Errorf("VRF proof generation failed: %w", err)
	}

	block := &Block{
		Height:       height,
		PrevHash:     prevHash,
		Timestamp:    time.Now(),
		Proposer:     proposer,
		Transactions: transactions,
		VRFProof:     vrfOutput.Proof,
		Signatures:   make(map[string][]byte),
	}

	block.Hash = pos.calculateBlockHash(block)
	validator.LastActive = time.Now()

	return block, nil
}

// ValidateBlock validates a PoS block
func (pos *ProofOfStake) ValidateBlock(block *Block) error {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	// Verify proposer exists and has stake
	validator, exists := pos.validators[block.Proposer]
	if !exists {
		return errors.New("proposer not found")
	}

	if validator.Stake.Cmp(pos.minStake) < 0 {
		return errors.New("proposer has insufficient stake")
	}

	// Verify VRF proof (simplified)
	if len(block.VRFProof) == 0 {
		return errors.New("missing VRF proof")
	}

	// Verify block hash
	expectedHash := pos.calculateBlockHash(block)
	if !equalBytes(expectedHash, block.Hash) {
		return errors.New("invalid block hash")
	}

	return nil
}

// FinalizeBlock adds block to PoS chain
func (pos *ProofOfStake) FinalizeBlock(block *Block) error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	pos.blockchain = append(pos.blockchain, block)
	pos.blocksProduced++

	// Update epoch if needed
	if uint64(len(pos.blockchain))%pos.epochLength == 0 {
		pos.transitionEpoch()
	}

	return nil
}

// transitionEpoch handles epoch transitions
func (pos *ProofOfStake) transitionEpoch() {
	pos.currentEpoch++

	// Generate new epoch seed
	hash := sha256.Sum256(append(pos.epochSeed, byte(pos.currentEpoch)))
	pos.epochSeed = hash[:]
}

// calculateBlockHash computes the hash of a PoS block
func (pos *ProofOfStake) calculateBlockHash(block *Block) []byte {
	data := make([]byte, 0)

	// Height
	heightBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(heightBytes, uint64(block.Height))
	data = append(data, heightBytes...)

	// Previous hash
	data = append(data, block.PrevHash...)

	// Timestamp
	timeBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(timeBytes, uint64(block.Timestamp.Unix()))
	data = append(data, timeBytes...)

	// Proposer
	data = append(data, []byte(block.Proposer)...)

	// VRF proof
	data = append(data, block.VRFProof...)

	// Transactions
	for _, tx := range block.Transactions {
		data = append(data, []byte(tx.ID)...)
	}

	hash := sha256.Sum256(data)
	return hash[:]
}

// AddValidator adds a PoS validator
func (pos *ProofOfStake) AddValidator(id string, stake *big.Int) error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	if stake.Cmp(pos.minStake) < 0 {
		return errors.New("insufficient stake")
	}

	// Create VRF for this validator
	seed := make([]byte, 32)
	copy(seed, id) // Simplified seed from ID
	vrf, err := crypto.NewVRFFromSeed(seed)
	if err != nil {
		return err
	}

	pos.validators[id] = &Validator{
		ID:         id,
		Stake:      new(big.Int).Set(stake),
		LastActive: time.Now(),
		IsActive:   true,
	}

	pos.vrfs[id] = vrf
	pos.totalStake.Add(pos.totalStake, stake)

	return nil
}

// RemoveValidator removes a PoS validator
func (pos *ProofOfStake) RemoveValidator(id string) error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	validator, exists := pos.validators[id]
	if !exists {
		return errors.New("validator not found")
	}

	pos.totalStake.Sub(pos.totalStake, validator.Stake)
	delete(pos.validators, id)
	delete(pos.vrfs, id)

	return nil
}

// GetValidators returns PoS validators
func (pos *ProofOfStake) GetValidators() map[string]*Validator {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	validators := make(map[string]*Validator)
	for k, v := range pos.validators {
		validators[k] = v
	}
	return validators
}

// GetMetrics returns PoS metrics
func (pos *ProofOfStake) GetMetrics() *ConsensusMetrics {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	uptime := time.Since(pos.startTime)
	avgBlockTime := time.Duration(0)
	if pos.blocksProduced > 0 {
		avgBlockTime = uptime / time.Duration(pos.blocksProduced)
	}

	return &ConsensusMetrics{
		Algorithm:             "Proof of Stake",
		BlocksProduced:        pos.blocksProduced,
		TotalTransactions:     pos.getTotalTransactions(),
		AverageBlockTime:      avgBlockTime,
		ThroughputTPS:         float64(pos.getTotalTransactions()) / uptime.Seconds(),
		EnergyConsumption:     0.001, // Very low energy consumption
		DecentralizationIndex: pos.calculateDecentralization(),
		FinalityTime:          avgBlockTime * 2,                // 2 epochs for finality
		NetworkOverhead:       int64(len(pos.validators)) * 10, // VRF + signatures
	}
}

// Reset resets PoS state
func (pos *ProofOfStake) Reset() error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	pos.blockchain = pos.blockchain[:0]
	pos.blocksProduced = 0
	pos.currentEpoch = 0
	pos.startTime = time.Now()

	return nil
}

// =============================================================================
// Practical Byzantine Fault Tolerance (PBFT)
// =============================================================================

// PBFT implements simplified PBFT consensus
type PBFT struct {
	mu sync.RWMutex

	validators map[string]*Validator
	blockchain []*Block
	view       uint64
	round      uint64

	// PBFT phases
	prepareMsgs map[string]map[string]bool // block hash -> validator -> prepared
	commitMsgs  map[string]map[string]bool // block hash -> validator -> committed

	// Metrics
	startTime      time.Time
	blocksProduced int64
	messagesSent   int64
	byzantineCount int
}

// NewPBFT creates a new PBFT consensus engine
func NewPBFT() *PBFT {
	return &PBFT{
		validators:  make(map[string]*Validator),
		blockchain:  make([]*Block, 0),
		prepareMsgs: make(map[string]map[string]bool),
		commitMsgs:  make(map[string]map[string]bool),
		startTime:   time.Now(),
	}
}

// SelectLeader in PBFT, leader is determined by round-robin
func (pbft *PBFT) SelectLeader() (string, error) {
	pbft.mu.RLock()
	defer pbft.mu.RUnlock()

	if len(pbft.validators) == 0 {
		return "", errors.New("no validators available")
	}

	// Round-robin leader selection
	validators := make([]string, 0, len(pbft.validators))
	for id, validator := range pbft.validators {
		if validator.IsActive {
			validators = append(validators, id)
		}
	}

	if len(validators) == 0 {
		return "", errors.New("no active validators")
	}

	sort.Strings(validators) // Deterministic ordering
	leaderIndex := pbft.view % uint64(len(validators))

	return validators[leaderIndex], nil
}

// ProposeBlock creates a new PBFT block
func (pbft *PBFT) ProposeBlock(proposer string, transactions []Transaction) (*Block, error) {
	pbft.mu.Lock()
	defer pbft.mu.Unlock()

	validator, exists := pbft.validators[proposer]
	if !exists {
		return nil, errors.New("proposer not found")
	}

	// Get previous block hash
	var prevHash []byte
	height := int64(0)
	if len(pbft.blockchain) > 0 {
		prevBlock := pbft.blockchain[len(pbft.blockchain)-1]
		prevHash = prevBlock.Hash
		height = prevBlock.Height + 1
	}

	block := &Block{
		Height:       height,
		PrevHash:     prevHash,
		Timestamp:    time.Now(),
		Proposer:     proposer,
		Transactions: transactions,
		Signatures:   make(map[string][]byte),
	}

	block.Hash = pbft.calculateBlockHash(block)
	validator.LastActive = time.Now()

	// Initialize message tracking for this block
	blockHashStr := string(block.Hash)
	pbft.prepareMsgs[blockHashStr] = make(map[string]bool)
	pbft.commitMsgs[blockHashStr] = make(map[string]bool)

	return block, nil
}

// ValidateBlock validates a PBFT block
func (pbft *PBFT) ValidateBlock(block *Block) error {
	// Basic validation
	expectedHash := pbft.calculateBlockHash(block)
	if !equalBytes(expectedHash, block.Hash) {
		return errors.New("invalid block hash")
	}

	// PBFT requires 2f+1 prepare messages and 2f+1 commit messages
	f := (len(pbft.validators) - 1) / 3
	threshold := 2*f + 1

	blockHashStr := string(block.Hash)
	prepareCount := len(pbft.prepareMsgs[blockHashStr])
	commitCount := len(pbft.commitMsgs[blockHashStr])

	if prepareCount < threshold || commitCount < threshold {
		return fmt.Errorf("insufficient consensus: prepare=%d, commit=%d, required=%d",
			prepareCount, commitCount, threshold)
	}

	return nil
}

// FinalizeBlock adds block to PBFT chain
func (pbft *PBFT) FinalizeBlock(block *Block) error {
	pbft.mu.Lock()
	defer pbft.mu.Unlock()

	pbft.blockchain = append(pbft.blockchain, block)
	pbft.blocksProduced++
	pbft.view++

	// Clean up message tracking
	blockHashStr := string(block.Hash)
	delete(pbft.prepareMsgs, blockHashStr)
	delete(pbft.commitMsgs, blockHashStr)

	return nil
}

// SimulatePBFTVoting simulates the PBFT voting process
func (pbft *PBFT) SimulatePBFTVoting(block *Block) error {
	pbft.mu.Lock()
	defer pbft.mu.Unlock()

	blockHashStr := string(block.Hash)
	f := (len(pbft.validators) - 1) / 3

	// Simulate prepare phase
	prepareCount := 0
	for id, validator := range pbft.validators {
		if validator.IsActive && prepareCount < 2*f+1 {
			pbft.prepareMsgs[blockHashStr][id] = true
			prepareCount++
			pbft.messagesSent += int64(len(pbft.validators)) // Broadcast to all
		}
	}

	// Simulate commit phase
	commitCount := 0
	for id, validator := range pbft.validators {
		if validator.IsActive && commitCount < 2*f+1 {
			pbft.commitMsgs[blockHashStr][id] = true
			commitCount++
			pbft.messagesSent += int64(len(pbft.validators)) // Broadcast to all
		}
	}

	return nil
}

// calculateBlockHash computes hash for PBFT block
func (pbft *PBFT) calculateBlockHash(block *Block) []byte {
	data := make([]byte, 0)

	// Height
	heightBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(heightBytes, uint64(block.Height))
	data = append(data, heightBytes...)

	// Previous hash
	data = append(data, block.PrevHash...)

	// Timestamp
	timeBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(timeBytes, uint64(block.Timestamp.Unix()))
	data = append(data, timeBytes...)

	// Proposer
	data = append(data, []byte(block.Proposer)...)

	// Transactions
	for _, tx := range block.Transactions {
		data = append(data, []byte(tx.ID)...)
	}

	hash := sha256.Sum256(data)
	return hash[:]
}

// AddValidator adds a PBFT validator
func (pbft *PBFT) AddValidator(id string, stake *big.Int) error {
	pbft.mu.Lock()
	defer pbft.mu.Unlock()

	pbft.validators[id] = &Validator{
		ID:         id,
		Stake:      new(big.Int).Set(stake),
		LastActive: time.Now(),
		IsActive:   true,
	}

	return nil
}

// RemoveValidator removes a PBFT validator
func (pbft *PBFT) RemoveValidator(id string) error {
	pbft.mu.Lock()
	defer pbft.mu.Unlock()

	delete(pbft.validators, id)
	return nil
}

// GetValidators returns PBFT validators
func (pbft *PBFT) GetValidators() map[string]*Validator {
	pbft.mu.RLock()
	defer pbft.mu.RUnlock()

	validators := make(map[string]*Validator)
	for k, v := range pbft.validators {
		validators[k] = v
	}
	return validators
}

// GetMetrics returns PBFT metrics
func (pbft *PBFT) GetMetrics() *ConsensusMetrics {
	pbft.mu.RLock()
	defer pbft.mu.RUnlock()

	uptime := time.Since(pbft.startTime)
	avgBlockTime := time.Duration(0)
	if pbft.blocksProduced > 0 {
		avgBlockTime = uptime / time.Duration(pbft.blocksProduced)
	}

	return &ConsensusMetrics{
		Algorithm:             "PBFT",
		BlocksProduced:        pbft.blocksProduced,
		TotalTransactions:     pbft.getTotalTransactions(),
		AverageBlockTime:      avgBlockTime,
		ThroughputTPS:         float64(pbft.getTotalTransactions()) / uptime.Seconds(),
		EnergyConsumption:     0.01, // Low energy, but more than PoS due to communication
		DecentralizationIndex: pbft.calculateDecentralization(),
		FinalityTime:          avgBlockTime, // Immediate finality
		NetworkOverhead:       pbft.messagesSent,
	}
}

// Reset resets PBFT state
func (pbft *PBFT) Reset() error {
	pbft.mu.Lock()
	defer pbft.mu.Unlock()

	pbft.blockchain = pbft.blockchain[:0]
	pbft.blocksProduced = 0
	pbft.messagesSent = 0
	pbft.view = 0
	pbft.round = 0
	pbft.prepareMsgs = make(map[string]map[string]bool)
	pbft.commitMsgs = make(map[string]map[string]bool)
	pbft.startTime = time.Now()

	return nil
}

// =============================================================================
// Helper functions
// =============================================================================

// getTotalTransactions counts total transactions across all algorithms
func (pow *ProofOfWork) getTotalTransactions() int64 {
	total := int64(0)
	for _, block := range pow.blockchain {
		total += int64(len(block.Transactions))
	}
	return total
}

func (pos *ProofOfStake) getTotalTransactions() int64 {
	total := int64(0)
	for _, block := range pos.blockchain {
		total += int64(len(block.Transactions))
	}
	return total
}

func (pbft *PBFT) getTotalTransactions() int64 {
	total := int64(0)
	for _, block := range pbft.blockchain {
		total += int64(len(block.Transactions))
	}
	return total
}

// calculateDecentralization computes decentralization index
func (pow *ProofOfWork) calculateDecentralization() float64 {
	if len(pow.validators) <= 1 {
		return 0.0
	}

	// Simple metric: 1 - (largest_stake / total_stake)
	// In PoW, all validators have equal weight
	return 1.0 - (1.0 / float64(len(pow.validators)))
}

func (pos *ProofOfStake) calculateDecentralization() float64 {
	if pos.totalStake.Cmp(big.NewInt(0)) <= 0 {
		return 0.0
	}

	// Find largest stake
	largestStake := big.NewInt(0)
	for _, validator := range pos.validators {
		if validator.Stake.Cmp(largestStake) > 0 {
			largestStake.Set(validator.Stake)
		}
	}

	// Calculate ratio
	ratio := new(big.Float).SetInt(largestStake)
	ratio.Quo(ratio, new(big.Float).SetInt(pos.totalStake))
	ratioFloat, _ := ratio.Float64()

	return 1.0 - ratioFloat
}

func (pbft *PBFT) calculateDecentralization() float64 {
	if len(pbft.validators) <= 1 {
		return 0.0
	}

	// PBFT has equal weight validators
	return 1.0 - (1.0 / float64(len(pbft.validators)))
}

// equalBytes compares two byte slices
func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
