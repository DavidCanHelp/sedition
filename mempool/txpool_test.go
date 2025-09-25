package mempool

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockStateReader implements StateReader for testing
type MockStateReader struct {
	balances map[string]*big.Int
	nonces   map[string]uint64
}

func NewMockStateReader() *MockStateReader {
	return &MockStateReader{
		balances: make(map[string]*big.Int),
		nonces:   make(map[string]uint64),
	}
}

func (m *MockStateReader) GetBalance(address string) *big.Int {
	if balance, ok := m.balances[address]; ok {
		return new(big.Int).Set(balance)
	}
	return big.NewInt(0)
}

func (m *MockStateReader) GetNonce(address string) uint64 {
	return m.nonces[address]
}

func (m *MockStateReader) SetBalance(address string, balance *big.Int) {
	m.balances[address] = new(big.Int).Set(balance)
}

func (m *MockStateReader) SetNonce(address string, nonce uint64) {
	m.nonces[address] = nonce
}

func createTestTransaction(from, to string, nonce uint64, value, gasPrice *big.Int) *storage.Transaction {
	return &storage.Transaction{
		Hash:      fmt.Sprintf("tx_%s_%d", from, nonce),
		From:      from,
		To:        to,
		Value:     value,
		Nonce:     nonce,
		GasLimit:  21000,
		GasPrice:  gasPrice,
		Timestamp: time.Now(),
		Signature: []byte("test_signature"),
	}
}

func TestTxPoolAddTransaction(t *testing.T) {
	state := NewMockStateReader()
	state.SetBalance("alice", big.NewInt(1000000000000000000)) // 1 ETH
	state.SetNonce("alice", 0)

	config := DefaultTxPoolConfig()
	pool := NewTxPool(config, state)

	// Test adding valid transaction
	tx1 := createTestTransaction("alice", "bob", 0, big.NewInt(1000), big.NewInt(1000000000))
	err := pool.AddTransaction(tx1)
	require.NoError(t, err)
	assert.Equal(t, 1, pool.Size())
	assert.Equal(t, 1, pool.PendingSize())

	// Test adding duplicate transaction
	err = pool.AddTransaction(tx1)
	assert.Equal(t, ErrTransactionExists, err)

	// Test adding future transaction
	tx2 := createTestTransaction("alice", "bob", 2, big.NewInt(1000), big.NewInt(1000000000))
	err = pool.AddTransaction(tx2)
	require.NoError(t, err)
	assert.Equal(t, 2, pool.Size())
	assert.Equal(t, 1, pool.PendingSize()) // Still 1 pending, tx2 is queued

	// Test adding transaction that fills the gap
	tx3 := createTestTransaction("alice", "bob", 1, big.NewInt(1000), big.NewInt(1000000000))
	err = pool.AddTransaction(tx3)
	require.NoError(t, err)
	assert.Equal(t, 3, pool.Size())
	assert.Equal(t, 3, pool.PendingSize()) // All 3 should be pending now
}

func TestTxPoolInsufficientBalance(t *testing.T) {
	state := NewMockStateReader()
	state.SetBalance("alice", big.NewInt(1000)) // Only 1000 wei
	state.SetNonce("alice", 0)

	config := DefaultTxPoolConfig()
	pool := NewTxPool(config, state)

	// Try to send more than balance
	tx := createTestTransaction("alice", "bob", 0, big.NewInt(10000), big.NewInt(1000000000))
	err := pool.AddTransaction(tx)
	assert.Equal(t, ErrInsufficientBalance, err)
}

func TestTxPoolGasPrice(t *testing.T) {
	state := NewMockStateReader()
	state.SetBalance("alice", big.NewInt(1000000000000000000))
	state.SetNonce("alice", 0)

	config := DefaultTxPoolConfig()
	config.MinGasPrice = big.NewInt(1000000000) // 1 Gwei
	pool := NewTxPool(config, state)

	// Test transaction with low gas price
	tx := createTestTransaction("alice", "bob", 0, big.NewInt(1000), big.NewInt(100))
	err := pool.AddTransaction(tx)
	assert.Equal(t, ErrGasPriceTooLow, err)

	// Test transaction with sufficient gas price
	tx2 := createTestTransaction("alice", "bob", 0, big.NewInt(1000), big.NewInt(1000000000))
	err = pool.AddTransaction(tx2)
	require.NoError(t, err)
}

func TestTxPoolReplacement(t *testing.T) {
	state := NewMockStateReader()
	state.SetBalance("alice", big.NewInt(1000000000000000000))
	state.SetNonce("alice", 0)

	config := DefaultTxPoolConfig()
	config.PriceBump = 10 // 10% increase required
	pool := NewTxPool(config, state)

	// Add initial transaction
	tx1 := createTestTransaction("alice", "bob", 0, big.NewInt(1000), big.NewInt(1000000000))
	err := pool.AddTransaction(tx1)
	require.NoError(t, err)

	// Try to replace with same gas price - should fail
	tx2 := createTestTransaction("alice", "charlie", 0, big.NewInt(1000), big.NewInt(1000000000))
	tx2.Hash = "tx_alice_0_replacement"
	err = pool.AddTransaction(tx2)
	require.NoError(t, err) // Transaction added but doesn't replace

	// Verify original transaction still exists
	gotTx := pool.GetTransaction(tx1.Hash)
	assert.NotNil(t, gotTx)

	// Replace with higher gas price
	tx3 := createTestTransaction("alice", "charlie", 0, big.NewInt(1000), big.NewInt(1100000000))
	tx3.Hash = "tx_alice_0_higher"
	err = pool.AddTransaction(tx3)
	require.NoError(t, err)

	// Verify replacement happened
	pending := pool.GetPending()
	assert.Equal(t, 1, len(pending["alice"]))
	assert.Equal(t, tx3.Hash, pending["alice"][0].Hash)
}

func TestTxPoolGetSortedPending(t *testing.T) {
	state := NewMockStateReader()
	state.SetBalance("alice", big.NewInt(1000000000000000000))
	state.SetBalance("bob", big.NewInt(1000000000000000000))
	state.SetBalance("charlie", big.NewInt(1000000000000000000))

	config := DefaultTxPoolConfig()
	pool := NewTxPool(config, state)

	// Add transactions with different gas prices
	tx1 := createTestTransaction("alice", "dave", 0, big.NewInt(1000), big.NewInt(1000000000))
	tx2 := createTestTransaction("bob", "dave", 0, big.NewInt(1000), big.NewInt(3000000000))
	tx3 := createTestTransaction("charlie", "dave", 0, big.NewInt(1000), big.NewInt(2000000000))

	pool.AddTransaction(tx1)
	pool.AddTransaction(tx2)
	pool.AddTransaction(tx3)

	// Get sorted transactions
	sorted := pool.GetSortedPending(0)
	require.Len(t, sorted, 3)

	// Verify sorted by gas price (highest first)
	assert.Equal(t, tx2.Hash, sorted[0].Hash) // 3 Gwei
	assert.Equal(t, tx3.Hash, sorted[1].Hash) // 2 Gwei
	assert.Equal(t, tx1.Hash, sorted[2].Hash) // 1 Gwei

	// Test with limit
	limited := pool.GetSortedPending(2)
	require.Len(t, limited, 2)
	assert.Equal(t, tx2.Hash, limited[0].Hash)
	assert.Equal(t, tx3.Hash, limited[1].Hash)
}

func TestTxPoolUpdateState(t *testing.T) {
	state := NewMockStateReader()
	state.SetBalance("alice", big.NewInt(1000000000000000000))
	state.SetNonce("alice", 0)

	config := DefaultTxPoolConfig()
	pool := NewTxPool(config, state)

	// Add transactions
	tx1 := createTestTransaction("alice", "bob", 0, big.NewInt(1000), big.NewInt(1000000000))
	tx2 := createTestTransaction("alice", "bob", 1, big.NewInt(1000), big.NewInt(1000000000))
	tx3 := createTestTransaction("alice", "bob", 2, big.NewInt(1000), big.NewInt(1000000000))

	pool.AddTransaction(tx1)
	pool.AddTransaction(tx2)
	pool.AddTransaction(tx3)

	assert.Equal(t, 3, pool.PendingSize())

	// Update state - simulate tx1 being mined
	state.SetNonce("alice", 1)
	pool.UpdateState(state)

	// tx1 should be removed, tx2 and tx3 should remain
	assert.Equal(t, 2, pool.Size())
	assert.Equal(t, 2, pool.PendingSize())
	assert.Nil(t, pool.GetTransaction(tx1.Hash))
	assert.NotNil(t, pool.GetTransaction(tx2.Hash))
	assert.NotNil(t, pool.GetTransaction(tx3.Hash))
}

func TestTxPoolEviction(t *testing.T) {
	state := NewMockStateReader()
	state.SetBalance("alice", big.NewInt(1000000000000000000))
	state.SetBalance("bob", big.NewInt(1000000000000000000))

	config := DefaultTxPoolConfig()
	config.MaxPoolSize = 2 // Small pool for testing
	pool := NewTxPool(config, state)

	// Fill the pool
	tx1 := createTestTransaction("alice", "charlie", 0, big.NewInt(1000), big.NewInt(1000000000))
	tx2 := createTestTransaction("bob", "charlie", 0, big.NewInt(1000), big.NewInt(2000000000))

	pool.AddTransaction(tx1)
	pool.AddTransaction(tx2)
	assert.Equal(t, 2, pool.Size())

	// Add transaction with higher gas price - should evict tx1
	tx3 := createTestTransaction("alice", "charlie", 1, big.NewInt(1000), big.NewInt(3000000000))
	err := pool.AddTransaction(tx3)
	require.NoError(t, err)

	assert.Equal(t, 2, pool.Size()) // Still 2 transactions
	assert.Nil(t, pool.GetTransaction(tx1.Hash)) // tx1 evicted
	assert.NotNil(t, pool.GetTransaction(tx2.Hash))
	assert.NotNil(t, pool.GetTransaction(tx3.Hash))

	// Try to add transaction with lower gas price - should fail
	tx4 := createTestTransaction("bob", "charlie", 1, big.NewInt(1000), big.NewInt(500000000))
	err = pool.AddTransaction(tx4)
	assert.Equal(t, ErrPoolFull, err)
}