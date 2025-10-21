package mempool

import (
	"errors"
	"math/big"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/storage"
)

var (
	ErrTransactionExists     = errors.New("transaction already exists")
	ErrInsufficientBalance   = errors.New("insufficient balance")
	ErrInvalidNonce          = errors.New("invalid nonce")
	ErrTransactionTooOld     = errors.New("transaction too old")
	ErrPoolFull              = errors.New("transaction pool is full")
	ErrInvalidSignature      = errors.New("invalid transaction signature")
	ErrGasPriceTooLow        = errors.New("gas price too low")
)

// TxPool manages pending transactions
type TxPool struct {
	mu sync.RWMutex

	// Pending transactions ready to be included in blocks
	pending map[string]map[uint64]*storage.Transaction // address -> nonce -> transaction

	// Queue of future transactions (nonce too high)
	queue map[string]map[uint64]*storage.Transaction

	// All transactions by hash for quick lookup
	all map[string]*storage.Transaction

	// Configuration
	config TxPoolConfig

	// State interface for balance/nonce checks
	currentState StateReader

	// Event subscribers
	subscribers []chan<- *storage.Transaction
}

// TxPoolConfig holds configuration for transaction pool
type TxPoolConfig struct {
	MaxPoolSize      int           // Maximum number of transactions in pool
	MaxAccountSlots  int           // Maximum transactions per account
	MinGasPrice      *big.Int      // Minimum gas price for acceptance
	MaxTxAge         time.Duration // Maximum age of transaction
	PriceBump        int           // Minimum price bump percentage for replacement
	CleanupInterval  time.Duration // How often to clean up old transactions
}

// StateReader provides read access to blockchain state
type StateReader interface {
	GetBalance(address string) *big.Int
	GetNonce(address string) uint64
}

// DefaultTxPoolConfig returns default configuration
func DefaultTxPoolConfig() TxPoolConfig {
	return TxPoolConfig{
		MaxPoolSize:      10000,
		MaxAccountSlots:  64,
		MinGasPrice:      big.NewInt(1000000000), // 1 Gwei
		MaxTxAge:         3 * time.Hour,
		PriceBump:        10, // 10% minimum price increase for replacement
		CleanupInterval:  15 * time.Minute,
	}
}

// NewTxPool creates a new transaction pool
func NewTxPool(config TxPoolConfig, state StateReader) *TxPool {
	pool := &TxPool{
		pending:      make(map[string]map[uint64]*storage.Transaction),
		queue:        make(map[string]map[uint64]*storage.Transaction),
		all:          make(map[string]*storage.Transaction),
		config:       config,
		currentState: state,
		subscribers:  make([]chan<- *storage.Transaction, 0),
	}

	// Start cleanup routine
	go pool.cleanupLoop()

	return pool
}

// AddTransaction adds a transaction to the pool
func (p *TxPool) AddTransaction(tx *storage.Transaction) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Check if transaction already exists
	if p.all[tx.Hash] != nil {
		return ErrTransactionExists
	}

	// Validate transaction
	if err := p.validateTransaction(tx); err != nil {
		return err
	}

	// Check pool capacity
	if len(p.all) >= p.config.MaxPoolSize {
		// Try to evict lower priority transaction
		if !p.tryEviction(tx) {
			return ErrPoolFull
		}
	}

	from := tx.From
	nonce := tx.Nonce

	// Get expected nonce for this account
	expectedNonce := p.currentState.GetNonce(from)

	// Add to all transactions
	p.all[tx.Hash] = tx

	if nonce < expectedNonce {
		// Transaction is too old
		delete(p.all, tx.Hash)
		return ErrTransactionTooOld
	} else if nonce == expectedNonce {
		// Transaction can be executed immediately
		p.addToPending(from, tx)
		p.promoteQueuedTransactions(from)
	} else {
		// Transaction is for the future
		p.addToQueue(from, tx)
		// Check if this transaction fills a gap and enables promotion
		p.promoteQueuedTransactions(from)
	}

	// Notify subscribers
	p.notifySubscribers(tx)

	return nil
}

// GetPending returns all pending transactions ready for inclusion
func (p *TxPool) GetPending() map[string]map[uint64]*storage.Transaction {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// Deep copy to prevent external modifications
	pending := make(map[string]map[uint64]*storage.Transaction)
	for addr, txs := range p.pending {
		pending[addr] = make(map[uint64]*storage.Transaction)
		for nonce, tx := range txs {
			pending[addr][nonce] = tx
		}
	}
	return pending
}

// GetTransaction returns a transaction by hash
func (p *TxPool) GetTransaction(hash string) *storage.Transaction {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.all[hash]
}

// RemoveTransaction removes a transaction from the pool
func (p *TxPool) RemoveTransaction(hash string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	tx, exists := p.all[hash]
	if !exists {
		return
	}

	delete(p.all, hash)

	// Remove from pending
	if pendingTxs, ok := p.pending[tx.From]; ok {
		delete(pendingTxs, tx.Nonce)
		if len(pendingTxs) == 0 {
			delete(p.pending, tx.From)
		}
	}

	// Remove from queue
	if queuedTxs, ok := p.queue[tx.From]; ok {
		delete(queuedTxs, tx.Nonce)
		if len(queuedTxs) == 0 {
			delete(p.queue, tx.From)
		}
	}
}

// UpdateState updates the state reader and reorganizes transactions
func (p *TxPool) UpdateState(state StateReader) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.currentState = state

	// Revalidate all transactions
	p.revalidateTransactions()
}

// Size returns the total number of transactions in the pool
func (p *TxPool) Size() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.all)
}

// PendingSize returns the number of pending transactions
func (p *TxPool) PendingSize() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	count := 0
	for _, txs := range p.pending {
		count += len(txs)
	}
	return count
}

// Subscribe adds a channel to receive new transaction notifications
func (p *TxPool) Subscribe(ch chan<- *storage.Transaction) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.subscribers = append(p.subscribers, ch)
}

// validateTransaction validates a transaction before adding to pool
func (p *TxPool) validateTransaction(tx *storage.Transaction) error {
	// Check signature
	if !tx.VerifySignature() {
		return ErrInvalidSignature
	}

	// Check gas price
	if tx.GasPrice.Cmp(p.config.MinGasPrice) < 0 {
		return ErrGasPriceTooLow
	}

	// Check balance
	balance := p.currentState.GetBalance(tx.From)
	totalCost := new(big.Int).Add(tx.Value, new(big.Int).Mul(tx.GasPrice, big.NewInt(int64(tx.GasLimit))))
	if balance.Cmp(totalCost) < 0 {
		return ErrInsufficientBalance
	}

	return nil
}

// addToPending adds a transaction to pending
func (p *TxPool) addToPending(from string, tx *storage.Transaction) {
	if p.pending[from] == nil {
		p.pending[from] = make(map[uint64]*storage.Transaction)
	}

	// Check if there's an existing transaction with same nonce
	if existing := p.pending[from][tx.Nonce]; existing != nil {
		// Replace only if gas price is sufficiently higher
		priceBump := new(big.Int).Mul(existing.GasPrice, big.NewInt(int64(100+p.config.PriceBump)))
		priceBump.Div(priceBump, big.NewInt(100))

		if tx.GasPrice.Cmp(priceBump) < 0 {
			return // Don't replace
		}

		// Remove old transaction
		delete(p.all, existing.Hash)
	}

	p.pending[from][tx.Nonce] = tx
}

// addToQueue adds a transaction to the future queue
func (p *TxPool) addToQueue(from string, tx *storage.Transaction) {
	if p.queue[from] == nil {
		p.queue[from] = make(map[uint64]*storage.Transaction)
	}

	// Check account slot limit
	if len(p.queue[from])+len(p.pending[from]) >= p.config.MaxAccountSlots {
		return // Too many transactions from this account
	}

	p.queue[from][tx.Nonce] = tx
}

// promoteQueuedTransactions moves queued transactions to pending if possible
func (p *TxPool) promoteQueuedTransactions(from string) {
	queued := p.queue[from]
	if queued == nil {
		return
	}

	expectedNonce := p.currentState.GetNonce(from)
	if p.pending[from] != nil {
		// Find highest continuous nonce in pending
		for nonce := expectedNonce; ; nonce++ {
			if p.pending[from][nonce] == nil {
				expectedNonce = nonce
				break
			}
		}
	}

	// Try to promote transactions in a loop until no more can be promoted
	// This handles the case where queued transactions form a continuous sequence
	promoted := true
	for promoted {
		promoted = false
		if tx, exists := queued[expectedNonce]; exists {
			p.addToPending(from, tx)
			delete(queued, expectedNonce)
			expectedNonce++
			promoted = true
		}
	}

	if len(queued) == 0 {
		delete(p.queue, from)
	}
}

// revalidateTransactions revalidates all transactions after state change
func (p *TxPool) revalidateTransactions() {
	// Collect invalid transactions
	var toRemove []string

	// Check pending transactions
	for addr, txs := range p.pending {
		expectedNonce := p.currentState.GetNonce(addr)
		balance := p.currentState.GetBalance(addr)

		for nonce, tx := range txs {
			if nonce < expectedNonce {
				toRemove = append(toRemove, tx.Hash)
				delete(txs, nonce)
				continue
			}

			totalCost := new(big.Int).Add(tx.Value, new(big.Int).Mul(tx.GasPrice, big.NewInt(int64(tx.GasLimit))))
			if balance.Cmp(totalCost) < 0 {
				toRemove = append(toRemove, tx.Hash)
				delete(txs, nonce)
				continue
			}

			// Move to queue if nonce is now too high
			if nonce > expectedNonce {
				p.addToQueue(addr, tx)
				delete(txs, nonce)
			}
		}

		// Clean up empty maps
		if len(txs) == 0 {
			delete(p.pending, addr)
		}
	}

	// Remove invalid transactions from all map
	for _, hash := range toRemove {
		delete(p.all, hash)
	}

	// Promote queued transactions
	for addr := range p.queue {
		p.promoteQueuedTransactions(addr)
	}
}

// tryEviction attempts to evict a lower priority transaction
func (p *TxPool) tryEviction(newTx *storage.Transaction) bool {
	// Find transaction with lowest gas price
	var lowestPrice *big.Int
	var lowestHash string

	for hash, tx := range p.all {
		if lowestPrice == nil || tx.GasPrice.Cmp(lowestPrice) < 0 {
			lowestPrice = tx.GasPrice
			lowestHash = hash
		}
	}

	// Only evict if new transaction has higher gas price
	if lowestPrice != nil && newTx.GasPrice.Cmp(lowestPrice) > 0 {
		delete(p.all, lowestHash)
		return true
	}

	return false
}

// notifySubscribers sends transaction to all subscribers
func (p *TxPool) notifySubscribers(tx *storage.Transaction) {
	for _, ch := range p.subscribers {
		select {
		case ch <- tx:
		default:
			// Don't block if subscriber is not ready
		}
	}
}

// cleanupLoop periodically removes old transactions
func (p *TxPool) cleanupLoop() {
	ticker := time.NewTicker(p.config.CleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		p.cleanup()
	}
}

// cleanup removes old and invalid transactions
func (p *TxPool) cleanup() {
	p.mu.Lock()
	defer p.mu.Unlock()

	cutoffTime := time.Now().Add(-p.config.MaxTxAge)
	var toRemove []string

	for hash, tx := range p.all {
		if tx.Timestamp.Before(cutoffTime) {
			toRemove = append(toRemove, hash)
		}
	}

	for _, hash := range toRemove {
		delete(p.all, hash)
	}
}

// GetSortedPending returns pending transactions sorted by gas price (highest first)
func (p *TxPool) GetSortedPending(limit int) []*storage.Transaction {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var txs []*storage.Transaction
	for _, accountTxs := range p.pending {
		for _, tx := range accountTxs {
			txs = append(txs, tx)
		}
	}

	// Sort by gas price (highest first)
	for i := 0; i < len(txs); i++ {
		for j := i + 1; j < len(txs); j++ {
			if txs[j].GasPrice.Cmp(txs[i].GasPrice) > 0 {
				txs[i], txs[j] = txs[j], txs[i]
			}
		}
	}

	if limit > 0 && len(txs) > limit {
		txs = txs[:limit]
	}

	return txs
}

// Clear removes all transactions from the pool
func (p *TxPool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.pending = make(map[string]map[uint64]*storage.Transaction)
	p.queue = make(map[string]map[uint64]*storage.Transaction)
	p.all = make(map[string]*storage.Transaction)
}

// Stats returns statistics about the transaction pool
func (p *TxPool) Stats() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pendingCount := 0
	queuedCount := 0

	for _, txs := range p.pending {
		pendingCount += len(txs)
	}

	for _, txs := range p.queue {
		queuedCount += len(txs)
	}

	return map[string]interface{}{
		"pending": pendingCount,
		"queued":  queuedCount,
		"total":   len(p.all),
	}
}

// GetNonce returns the next expected nonce for an address
func (p *TxPool) GetNonce(address string) uint64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// Check if we have pending transactions for this address
	if pendingTxs, exists := p.pending[address]; exists && len(pendingTxs) > 0 {
		// Find the highest nonce in pending
		var maxNonce uint64
		for nonce := range pendingTxs {
			if nonce > maxNonce {
				maxNonce = nonce
			}
		}
		return maxNonce + 1
	}

	// Otherwise return the nonce from state
	return p.currentState.GetNonce(address)
}