// Package storage implements blockchain storage and management
package storage

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/errors"
	"github.com/davidcanhelp/sedition/validator"
	"github.com/syndtr/goleveldb/leveldb"
)

// Transaction represents a transaction in the blockchain
type Transaction struct {
	ID        string                 `json:"id"`
	From      string                 `json:"from"`
	To        string                 `json:"to"`
	Amount    *big.Int               `json:"amount"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Signature []byte                 `json:"signature"`
}

// BlockHeader contains the metadata of a block
type BlockHeader struct {
	Height       uint64    `json:"height"`
	PreviousHash string    `json:"previous_hash"`
	Timestamp    time.Time `json:"timestamp"`
	Proposer     string    `json:"proposer"`
	StateRoot    string    `json:"state_root"`
	TxRoot       string    `json:"tx_root"`
}

// BlockData represents the full block including transactions
type BlockData struct {
	Header       BlockHeader   `json:"header"`
	Transactions []Transaction `json:"transactions"`
	Hash         string        `json:"hash"`
	Signatures   []Signature   `json:"signatures"`
}

// Signature represents a validator's signature on a block
type Signature struct {
	ValidatorID string `json:"validator_id"`
	Signature   []byte `json:"signature"`
	Timestamp   time.Time `json:"timestamp"`
}

// ChainState represents the current state of the blockchain
type ChainState struct {
	Height           uint64                    `json:"height"`
	LastBlockHash    string                    `json:"last_block_hash"`
	LastBlockTime    time.Time                 `json:"last_block_time"`
	ValidatorSet     map[string]*validator.Validator `json:"validator_set"`
	AccountBalances  map[string]*big.Int       `json:"account_balances"`
	TotalSupply      *big.Int                  `json:"total_supply"`
}

// Blockchain manages the blockchain state and storage
type Blockchain struct {
	mu              sync.RWMutex
	db              *leveldb.DB
	consensusEngine *consensus.Engine
	currentState    *ChainState
	blocks          map[uint64]*BlockData // In-memory cache
	pendingTxs      []Transaction
	config          *BlockchainConfig
}

// BlockchainConfig holds blockchain configuration
type BlockchainConfig struct {
	DataDir          string
	MaxBlockSize     int
	MaxTxPerBlock    int
	BlockCacheSize   int
	StateCheckpoint  uint64
	PruningEnabled   bool
	PruningHeight    uint64
}

// DefaultBlockchainConfig returns default blockchain configuration
func DefaultBlockchainConfig() *BlockchainConfig {
	return &BlockchainConfig{
		DataDir:         "./data/blockchain",
		MaxBlockSize:    1024 * 1024, // 1MB
		MaxTxPerBlock:   1000,
		BlockCacheSize:  100,
		StateCheckpoint: 1000,
		PruningEnabled:  false,
		PruningHeight:   10000,
	}
}

// NewBlockchain creates a new blockchain instance
func NewBlockchain(engine *consensus.Engine, config *BlockchainConfig) (*Blockchain, error) {
	if config == nil {
		config = DefaultBlockchainConfig()
	}

	// Open or create database
	db, err := leveldb.OpenFile(config.DataDir, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	bc := &Blockchain{
		db:              db,
		consensusEngine: engine,
		blocks:          make(map[uint64]*BlockData),
		pendingTxs:      make([]Transaction, 0),
		config:          config,
	}

	// Initialize or load chain state
	if err := bc.initializeChainState(); err != nil {
		db.Close()
		return nil, err
	}

	return bc, nil
}

// Close closes the blockchain database
func (bc *Blockchain) Close() error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Save current state
	if err := bc.saveChainState(); err != nil {
		return err
	}

	return bc.db.Close()
}

// initializeChainState initializes or loads the chain state
func (bc *Blockchain) initializeChainState() error {
	// Try to load existing state
	stateData, err := bc.db.Get([]byte("chain_state"), nil)
	if err == nil {
		var state ChainState
		if err := json.Unmarshal(stateData, &state); err != nil {
			return fmt.Errorf("failed to unmarshal chain state: %w", err)
		}
		bc.currentState = &state
		return nil
	}

	// Create genesis state
	bc.currentState = &ChainState{
		Height:          0,
		LastBlockHash:   "",
		LastBlockTime:   time.Now(),
		ValidatorSet:    make(map[string]*validator.Validator),
		AccountBalances: make(map[string]*big.Int),
		TotalSupply:     big.NewInt(0),
	}

	// Create genesis block
	genesis := bc.createGenesisBlock()
	return bc.AddBlock(genesis)
}

// createGenesisBlock creates the genesis block
func (bc *Blockchain) createGenesisBlock() *BlockData {
	header := BlockHeader{
		Height:       0,
		PreviousHash: "0000000000000000000000000000000000000000000000000000000000000000",
		Timestamp:    time.Unix(1700000000, 0), // Fixed timestamp for genesis
		Proposer:     "genesis",
		StateRoot:    "",
		TxRoot:       "",
	}

	block := &BlockData{
		Header:       header,
		Transactions: []Transaction{},
		Signatures:   []Signature{},
	}

	block.Hash = bc.calculateBlockHash(block)
	return block
}

// AddTransaction adds a transaction to the pending pool
func (bc *Blockchain) AddTransaction(tx Transaction) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Validate transaction
	if err := bc.validateTransaction(tx); err != nil {
		return err
	}

	// Add to pending pool
	bc.pendingTxs = append(bc.pendingTxs, tx)

	// Check if we should create a new block
	if len(bc.pendingTxs) >= bc.config.MaxTxPerBlock {
		// Trigger block creation (in production, this would notify consensus)
	}

	return nil
}

// CreateBlock creates a new block with pending transactions
func (bc *Blockchain) CreateBlock(proposer string) (*BlockData, error) {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Get transactions for block
	txCount := len(bc.pendingTxs)
	if txCount > bc.config.MaxTxPerBlock {
		txCount = bc.config.MaxTxPerBlock
	}

	transactions := bc.pendingTxs[:txCount]
	bc.pendingTxs = bc.pendingTxs[txCount:]

	// Create block header
	header := BlockHeader{
		Height:       bc.currentState.Height + 1,
		PreviousHash: bc.currentState.LastBlockHash,
		Timestamp:    time.Now(),
		Proposer:     proposer,
		StateRoot:    bc.calculateStateRoot(),
		TxRoot:       bc.calculateTxRoot(transactions),
	}

	block := &BlockData{
		Header:       header,
		Transactions: transactions,
		Signatures:   []Signature{},
	}

	block.Hash = bc.calculateBlockHash(block)
	return block, nil
}

// AddBlock adds a new block to the blockchain
func (bc *Blockchain) AddBlock(block *BlockData) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Validate block
	if err := bc.validateBlock(block); err != nil {
		return err
	}

	// Store block in database
	blockKey := fmt.Sprintf("block_%d", block.Header.Height)
	blockData, err := json.Marshal(block)
	if err != nil {
		return fmt.Errorf("failed to marshal block: %w", err)
	}

	if err := bc.db.Put([]byte(blockKey), blockData, nil); err != nil {
		return fmt.Errorf("failed to store block: %w", err)
	}

	// Update chain state
	bc.currentState.Height = block.Header.Height
	bc.currentState.LastBlockHash = block.Hash
	bc.currentState.LastBlockTime = block.Header.Timestamp

	// Process transactions
	for _, tx := range block.Transactions {
		bc.applyTransaction(tx)
	}

	// Add to cache
	bc.blocks[block.Header.Height] = block

	// Prune old blocks from cache
	if len(bc.blocks) > bc.config.BlockCacheSize {
		minHeight := block.Header.Height - uint64(bc.config.BlockCacheSize)
		for height := range bc.blocks {
			if height < minHeight {
				delete(bc.blocks, height)
			}
		}
	}

	// Save state periodically
	if bc.config.StateCheckpoint > 0 && block.Header.Height%bc.config.StateCheckpoint == 0 {
		if err := bc.saveChainState(); err != nil {
			return fmt.Errorf("failed to save chain state: %w", err)
		}
	}

	return nil
}

// GetBlock retrieves a block by height
func (bc *Blockchain) GetBlock(height uint64) (*BlockData, error) {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	// Check cache first
	if block, exists := bc.blocks[height]; exists {
		return block, nil
	}

	// Load from database
	blockKey := fmt.Sprintf("block_%d", height)
	blockData, err := bc.db.Get([]byte(blockKey), nil)
	if err != nil {
		return nil, fmt.Errorf("block not found at height %d", height)
	}

	var block BlockData
	if err := json.Unmarshal(blockData, &block); err != nil {
		return nil, fmt.Errorf("failed to unmarshal block: %w", err)
	}

	// Add to cache
	bc.blocks[height] = &block

	return &block, nil
}

// GetLatestBlock returns the latest block
func (bc *Blockchain) GetLatestBlock() (*BlockData, error) {
	bc.mu.RLock()
	height := bc.currentState.Height
	bc.mu.RUnlock()

	return bc.GetBlock(height)
}

// GetChainState returns the current chain state
func (bc *Blockchain) GetChainState() *ChainState {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	// Return a copy to prevent external modification
	stateCopy := *bc.currentState
	return &stateCopy
}

// validateBlock validates a block before adding it
func (bc *Blockchain) validateBlock(block *BlockData) error {
	// Special case for genesis block
	if block.Header.Height == 0 && bc.currentState.Height == 0 {
		// Genesis block validation
		if block.Header.Proposer != "genesis" {
			return errors.NewConsensusError(
				errors.ErrInvalidProof,
				"invalid genesis block proposer",
			)
		}
		return nil
	}

	// Check block height
	if block.Header.Height != bc.currentState.Height+1 {
		return errors.NewConsensusError(
			errors.ErrInvalidProof,
			"invalid block height",
		).WithDetails("expected", bc.currentState.Height+1).
			WithDetails("got", block.Header.Height)
	}

	// Check previous hash
	if block.Header.PreviousHash != bc.currentState.LastBlockHash {
		return errors.NewConsensusError(
			errors.ErrInvalidProof,
			"invalid previous hash",
		)
	}

	// Check timestamp
	if block.Header.Timestamp.Before(bc.currentState.LastBlockTime) {
		return errors.NewConsensusError(
			errors.ErrInvalidProof,
			"block timestamp before last block",
		)
	}

	// Verify block hash
	calculatedHash := bc.calculateBlockHash(block)
	if calculatedHash != block.Hash {
		return errors.NewConsensusError(
			errors.ErrInvalidProof,
			"invalid block hash",
		)
	}

	// Validate all transactions
	for _, tx := range block.Transactions {
		if err := bc.validateTransaction(tx); err != nil {
			return err
		}
	}

	return nil
}

// validateTransaction validates a transaction
func (bc *Blockchain) validateTransaction(tx Transaction) error {
	// Check basic fields
	if tx.ID == "" || tx.From == "" || tx.To == "" {
		return fmt.Errorf("invalid transaction fields")
	}

	// Check amount
	if tx.Amount == nil || tx.Amount.Sign() < 0 {
		return fmt.Errorf("invalid transaction amount")
	}

	// Check balance
	balance, exists := bc.currentState.AccountBalances[tx.From]
	if !exists || balance == nil {
		return fmt.Errorf("account not found or zero balance")
	}
	if balance.Cmp(tx.Amount) < 0 {
		return fmt.Errorf("insufficient balance")
	}

	// TODO: Verify signature

	return nil
}

// applyTransaction applies a transaction to the state
func (bc *Blockchain) applyTransaction(tx Transaction) {
	// Update balances
	if fromBalance, exists := bc.currentState.AccountBalances[tx.From]; exists {
		newBalance := new(big.Int).Sub(fromBalance, tx.Amount)
		bc.currentState.AccountBalances[tx.From] = newBalance
	}

	if toBalance, exists := bc.currentState.AccountBalances[tx.To]; exists {
		newBalance := new(big.Int).Add(toBalance, tx.Amount)
		bc.currentState.AccountBalances[tx.To] = newBalance
	} else {
		bc.currentState.AccountBalances[tx.To] = new(big.Int).Set(tx.Amount)
	}
}

// calculateBlockHash calculates the hash of a block
func (bc *Blockchain) calculateBlockHash(block *BlockData) string {
	data := fmt.Sprintf("%d%s%d%s%s%s",
		block.Header.Height,
		block.Header.PreviousHash,
		block.Header.Timestamp.Unix(),
		block.Header.Proposer,
		block.Header.StateRoot,
		block.Header.TxRoot,
	)

	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// calculateStateRoot calculates the state root hash
func (bc *Blockchain) calculateStateRoot() string {
	// Simplified - in production, use Merkle Patricia Trie
	stateData, _ := json.Marshal(bc.currentState)
	hash := sha256.Sum256(stateData)
	return hex.EncodeToString(hash[:])
}

// calculateTxRoot calculates the transaction root hash
func (bc *Blockchain) calculateTxRoot(txs []Transaction) string {
	if len(txs) == 0 {
		return ""
	}

	// Simplified - in production, use Merkle tree
	var combinedData []byte
	for _, tx := range txs {
		txData, _ := json.Marshal(tx)
		combinedData = append(combinedData, txData...)
	}

	hash := sha256.Sum256(combinedData)
	return hex.EncodeToString(hash[:])
}

// saveChainState saves the current chain state to disk
func (bc *Blockchain) saveChainState() error {
	stateData, err := json.Marshal(bc.currentState)
	if err != nil {
		return fmt.Errorf("failed to marshal chain state: %w", err)
	}

	if err := bc.db.Put([]byte("chain_state"), stateData, nil); err != nil {
		return fmt.Errorf("failed to save chain state: %w", err)
	}

	return nil
}

// GetBalance returns the balance of an account
func (bc *Blockchain) GetBalance(address string) *big.Int {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	if balance, exists := bc.currentState.AccountBalances[address]; exists {
		return new(big.Int).Set(balance)
	}

	return big.NewInt(0)
}

// GetHeight returns the current blockchain height
func (bc *Blockchain) GetHeight() uint64 {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	return bc.currentState.Height
}

// GetPendingTransactions returns pending transactions
func (bc *Blockchain) GetPendingTransactions() []Transaction {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	txCopy := make([]Transaction, len(bc.pendingTxs))
	copy(txCopy, bc.pendingTxs)
	return txCopy
}