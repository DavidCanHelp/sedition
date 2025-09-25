package mining

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/mempool"
	"github.com/davidcanhelp/sedition/storage"
	"github.com/davidcanhelp/sedition/wallet"
)

var (
	ErrNotStarted       = errors.New("block producer not started")
	ErrAlreadyStarted   = errors.New("block producer already started")
	ErrNoTransactions   = errors.New("no transactions to include")
	ErrBlockCreationFailed = errors.New("failed to create block")
)

// BlockProducer manages block production
type BlockProducer struct {
	mu sync.RWMutex

	// Core components
	blockchain *storage.Blockchain
	txPool     *mempool.TxPool
	consensus  *consensus.Engine
	wallet     *wallet.Wallet

	// Configuration
	config ProducerConfig

	// State
	isRunning bool
	ctx       context.Context
	cancel    context.CancelFunc

	// Metrics
	blocksProduced   uint64
	lastBlockTime    time.Time
	totalRewards     *big.Int

	// Channels
	newBlockCh chan *storage.Block
	subscribers []chan<- *storage.Block
}

// ProducerConfig holds configuration for block producer
type ProducerConfig struct {
	BlockTime        time.Duration // Time between blocks
	MinTransactions  int           // Minimum transactions to create a block
	MaxTransactions  int           // Maximum transactions per block
	BlockGasLimit    uint64        // Maximum gas per block
	BlockReward      *big.Int      // Reward for producing a block
	EnableEmptyBlocks bool         // Whether to produce empty blocks
}

// DefaultProducerConfig returns default configuration
func DefaultProducerConfig() ProducerConfig {
	return ProducerConfig{
		BlockTime:         10 * time.Second,
		MinTransactions:   0, // Allow empty blocks by default
		MaxTransactions:   1000,
		BlockGasLimit:     10000000, // 10M gas
		BlockReward:       big.NewInt(2000000000000000000), // 2 SED
		EnableEmptyBlocks: true,
	}
}

// NewBlockProducer creates a new block producer
func NewBlockProducer(
	blockchain *storage.Blockchain,
	txPool *mempool.TxPool,
	consensus *consensus.Engine,
	wallet *wallet.Wallet,
	config ProducerConfig,
) *BlockProducer {
	return &BlockProducer{
		blockchain:   blockchain,
		txPool:       txPool,
		consensus:    consensus,
		wallet:       wallet,
		config:       config,
		totalRewards: big.NewInt(0),
		newBlockCh:   make(chan *storage.Block, 100),
		subscribers:  make([]chan<- *storage.Block, 0),
	}
}

// Start starts the block production loop
func (bp *BlockProducer) Start() error {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	if bp.isRunning {
		return ErrAlreadyStarted
	}

	bp.ctx, bp.cancel = context.WithCancel(context.Background())
	bp.isRunning = true

	go bp.productionLoop()

	return nil
}

// Stop stops the block production loop
func (bp *BlockProducer) Stop() error {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	if !bp.isRunning {
		return ErrNotStarted
	}

	bp.cancel()
	bp.isRunning = false

	return nil
}

// IsRunning returns whether the producer is running
func (bp *BlockProducer) IsRunning() bool {
	bp.mu.RLock()
	defer bp.mu.RUnlock()
	return bp.isRunning
}

// productionLoop is the main block production loop
func (bp *BlockProducer) productionLoop() {
	ticker := time.NewTicker(bp.config.BlockTime)
	defer ticker.Stop()

	// Produce initial block immediately if needed
	bp.tryProduceBlock()

	for {
		select {
		case <-bp.ctx.Done():
			return
		case <-ticker.C:
			bp.tryProduceBlock()
		}
	}
}

// tryProduceBlock attempts to produce a new block
func (bp *BlockProducer) tryProduceBlock() {
	// Check if we should produce a block
	if !bp.shouldProduceBlock() {
		return
	}

	// Create new block
	block, err := bp.createBlock()
	if err != nil {
		// Log error but continue
		fmt.Printf("Failed to create block: %v\n", err)
		return
	}

	// Validate block through consensus
	if err := bp.consensus.ValidateBlock(block); err != nil {
		fmt.Printf("Block validation failed: %v\n", err)
		return
	}

	// Add block to blockchain
	if err := bp.blockchain.AddBlock(block); err != nil {
		fmt.Printf("Failed to add block: %v\n", err)
		return
	}

	// Update metrics
	bp.mu.Lock()
	bp.blocksProduced++
	bp.lastBlockTime = time.Now()
	bp.totalRewards.Add(bp.totalRewards, bp.config.BlockReward)
	bp.mu.Unlock()

	// Remove included transactions from pool
	for _, tx := range block.Transactions {
		bp.txPool.RemoveTransaction(tx.Hash)
	}

	// Notify subscribers
	bp.notifySubscribers(block)

	fmt.Printf("Produced block #%d with %d transactions\n", block.Height, len(block.Transactions))
}

// shouldProduceBlock determines if a new block should be produced
func (bp *BlockProducer) shouldProduceBlock() bool {
	// Check if enough time has passed since last block
	bp.mu.RLock()
	timeSinceLastBlock := time.Since(bp.lastBlockTime)
	bp.mu.RUnlock()

	if timeSinceLastBlock < bp.config.BlockTime/2 {
		return false // Too soon
	}

	// Check transaction pool
	pendingCount := bp.txPool.PendingSize()

	if pendingCount >= bp.config.MinTransactions {
		return true
	}

	if bp.config.EnableEmptyBlocks && timeSinceLastBlock >= bp.config.BlockTime {
		return true
	}

	return false
}

// createBlock creates a new block with pending transactions
func (bp *BlockProducer) createBlock() (*storage.Block, error) {
	// Get current blockchain height
	currentHeight := uint64(0)
	lastBlock := bp.blockchain.GetLatestBlock()
	if lastBlock != nil {
		currentHeight = lastBlock.Height
	}

	// Select transactions from pool
	transactions := bp.selectTransactions()

	// Create coinbase transaction (miner reward)
	coinbaseTx := bp.createCoinbaseTransaction()
	transactions = append([]*storage.Transaction{coinbaseTx}, transactions...)

	// Create block
	block := &storage.Block{
		Height:       currentHeight + 1,
		PreviousHash: bp.blockchain.GetLatestBlockHash(),
		Timestamp:    time.Now(),
		Transactions: transactions,
		Proposer:     bp.wallet.GetAddress(),
	}

	// Calculate state root (simplified - in production, use Merkle Patricia Trie)
	block.StateRoot = bp.calculateStateRoot(block)

	// Calculate transaction root
	block.TxRoot = bp.calculateTxRoot(transactions)

	// Generate block hash
	block.Hash = bp.calculateBlockHash(block)

	return block, nil
}

// selectTransactions selects transactions from the pool for inclusion
func (bp *BlockProducer) selectTransactions() []*storage.Transaction {
	// Get sorted pending transactions (by gas price)
	pending := bp.txPool.GetSortedPending(bp.config.MaxTransactions)

	var selected []*storage.Transaction
	totalGasUsed := uint64(0)

	for _, tx := range pending {
		// Check gas limit
		if totalGasUsed+tx.GasLimit > bp.config.BlockGasLimit {
			continue // Skip transaction if it would exceed block gas limit
		}

		// Validate transaction one more time
		if !tx.VerifySignature() {
			continue
		}

		selected = append(selected, tx)
		totalGasUsed += tx.GasLimit

		// Check max transactions
		if len(selected) >= bp.config.MaxTransactions {
			break
		}
	}

	return selected
}

// createCoinbaseTransaction creates the miner reward transaction
func (bp *BlockProducer) createCoinbaseTransaction() *storage.Transaction {
	return &storage.Transaction{
		Hash:      fmt.Sprintf("coinbase_%d_%s", time.Now().Unix(), bp.wallet.GetAddress()),
		From:      "0x0000000000000000000000000000000000000000", // Null address
		To:        bp.wallet.GetAddress(),
		Value:     new(big.Int).Set(bp.config.BlockReward),
		Nonce:     0,
		GasLimit:  0,
		GasPrice:  big.NewInt(0),
		Timestamp: time.Now(),
	}
}

// calculateStateRoot calculates the state root for a block
func (bp *BlockProducer) calculateStateRoot(block *storage.Block) string {
	// Simplified state root calculation
	// In production, use Merkle Patricia Trie of account states
	data := fmt.Sprintf("%d%s%d", block.Height, block.PreviousHash, len(block.Transactions))
	return fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
}

// calculateTxRoot calculates the transaction root for a block
func (bp *BlockProducer) calculateTxRoot(transactions []*storage.Transaction) string {
	if len(transactions) == 0 {
		return ""
	}

	// Create Merkle tree of transaction hashes
	var hashes [][]byte
	for _, tx := range transactions {
		hash, _ := hex.DecodeString(tx.Hash)
		hashes = append(hashes, hash)
	}

	// Build Merkle tree (simplified)
	for len(hashes) > 1 {
		var newLevel [][]byte
		for i := 0; i < len(hashes); i += 2 {
			var combined []byte
			combined = append(combined, hashes[i]...)
			if i+1 < len(hashes) {
				combined = append(combined, hashes[i+1]...)
			} else {
				combined = append(combined, hashes[i]...) // Duplicate last hash if odd number
			}
			hash := sha256.Sum256(combined)
			newLevel = append(newLevel, hash[:])
		}
		hashes = newLevel
	}

	return hex.EncodeToString(hashes[0])
}

// calculateBlockHash calculates the hash of a block
func (bp *BlockProducer) calculateBlockHash(block *storage.Block) string {
	// Must match the format used in blockchain.go's calculateBlockHash
	data := fmt.Sprintf("%d%s%d%s%s%s",
		block.Height,
		block.PreviousHash,
		block.Timestamp.Unix(),
		block.Proposer,
		block.StateRoot,
		block.TxRoot,
	)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// ProduceBlockNow forces immediate block production
func (bp *BlockProducer) ProduceBlockNow() error {
	if !bp.isRunning {
		return ErrNotStarted
	}

	bp.tryProduceBlock()
	return nil
}

// GetStats returns producer statistics
func (bp *BlockProducer) GetStats() map[string]interface{} {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	return map[string]interface{}{
		"blocks_produced": bp.blocksProduced,
		"last_block_time": bp.lastBlockTime,
		"total_rewards":   bp.totalRewards.String(),
		"is_running":      bp.isRunning,
	}
}

// Subscribe adds a channel to receive new block notifications
func (bp *BlockProducer) Subscribe(ch chan<- *storage.Block) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	bp.subscribers = append(bp.subscribers, ch)
}

// notifySubscribers sends new block to all subscribers
func (bp *BlockProducer) notifySubscribers(block *storage.Block) {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	for _, ch := range bp.subscribers {
		select {
		case ch <- block:
		default:
			// Don't block if subscriber is not ready
		}
	}
}

// SetBlockTime updates the block time
func (bp *BlockProducer) SetBlockTime(duration time.Duration) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	bp.config.BlockTime = duration
}

// SetBlockReward updates the block reward
func (bp *BlockProducer) SetBlockReward(reward *big.Int) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	bp.config.BlockReward = new(big.Int).Set(reward)
}