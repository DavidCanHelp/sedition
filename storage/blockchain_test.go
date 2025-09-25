package storage

import (
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/consensus"
)

func TestNewBlockchain(t *testing.T) {
	// Create temp directory for test
	tempDir := t.TempDir()
	config := &BlockchainConfig{
		DataDir:         tempDir,
		MaxBlockSize:    1024 * 1024,
		MaxTxPerBlock:   100,
		BlockCacheSize:  10,
		StateCheckpoint: 10,
	}

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Check initial state
	if bc.GetHeight() != 0 {
		t.Errorf("expected height 0, got %d", bc.GetHeight())
	}

	state := bc.GetChainState()
	if state == nil {
		t.Fatal("chain state should not be nil")
	}
	if state.Height != 0 {
		t.Errorf("expected state height 0, got %d", state.Height)
	}
}

func TestBlockchain_AddTransaction(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Initialize some balances
	bc.currentState.AccountBalances["alice"] = big.NewInt(1000)
	bc.currentState.AccountBalances["bob"] = big.NewInt(500)

	// Valid transaction
	tx := Transaction{
		ID:        "tx1",
		From:      "alice",
		To:        "bob",
		Amount:    big.NewInt(100),
		Timestamp: time.Now(),
	}

	if err := bc.AddTransaction(tx); err != nil {
		t.Errorf("failed to add valid transaction: %v", err)
	}

	// Check pending transactions
	pending := bc.GetPendingTransactions()
	if len(pending) != 1 {
		t.Errorf("expected 1 pending transaction, got %d", len(pending))
	}

	// Invalid transaction (insufficient balance)
	invalidTx := Transaction{
		ID:        "tx2",
		From:      "alice",
		To:        "bob",
		Amount:    big.NewInt(10000),
		Timestamp: time.Now(),
	}

	if err := bc.AddTransaction(invalidTx); err == nil {
		t.Error("expected error for insufficient balance")
	}
}

func TestBlockchain_CreateBlock(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Initialize balances
	bc.currentState.AccountBalances["alice"] = big.NewInt(1000)

	// Add some transactions
	for i := 0; i < 5; i++ {
		tx := Transaction{
			ID:        fmt.Sprintf("tx%d", i),
			From:      "alice",
			To:        "bob",
			Amount:    big.NewInt(10),
			Timestamp: time.Now(),
		}
		bc.AddTransaction(tx)
	}

	// Create block
	block, err := bc.CreateBlock("validator1")
	if err != nil {
		t.Fatalf("failed to create block: %v", err)
	}

	if block.Header.Height != 1 {
		t.Errorf("expected block height 1, got %d", block.Header.Height)
	}
	if block.Header.Proposer != "validator1" {
		t.Errorf("expected proposer validator1, got %s", block.Header.Proposer)
	}
	if len(block.Transactions) != 5 {
		t.Errorf("expected 5 transactions, got %d", len(block.Transactions))
	}
	if block.Hash == "" {
		t.Error("block hash should not be empty")
	}
}

func TestBlockchain_AddBlock(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Initialize balances
	bc.currentState.AccountBalances["alice"] = big.NewInt(1000)

	// Create and add block
	block := &BlockData{
		Header: BlockHeader{
			Height:       1,
			PreviousHash: bc.currentState.LastBlockHash,
			Timestamp:    time.Now(),
			Proposer:     "validator1",
			StateRoot:    bc.calculateStateRoot(),
			TxRoot:       "",
		},
		Transactions: []Transaction{
			{
				ID:        "tx1",
				From:      "alice",
				To:        "bob",
				Amount:    big.NewInt(100),
				Timestamp: time.Now(),
			},
		},
		Signatures: []Signature{},
	}
	block.Hash = bc.calculateBlockHash(block)

	if err := bc.AddBlock(block); err != nil {
		t.Fatalf("failed to add block: %v", err)
	}

	// Check updated state
	if bc.GetHeight() != 1 {
		t.Errorf("expected height 1, got %d", bc.GetHeight())
	}

	// Check balances updated
	aliceBalance := bc.GetBalance("alice")
	if aliceBalance.Cmp(big.NewInt(900)) != 0 {
		t.Errorf("expected alice balance 900, got %s", aliceBalance.String())
	}

	bobBalance := bc.GetBalance("bob")
	if bobBalance.Cmp(big.NewInt(100)) != 0 {
		t.Errorf("expected bob balance 100, got %s", bobBalance.String())
	}
}

func TestBlockchain_GetBlock(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Get genesis block
	genesis, err := bc.GetBlock(0)
	if err != nil {
		t.Fatalf("failed to get genesis block: %v", err)
	}

	if genesis.Header.Height != 0 {
		t.Error("genesis block should have height 0")
	}
	if genesis.Header.Proposer != "genesis" {
		t.Error("genesis block should have proposer 'genesis'")
	}

	// Try to get non-existent block
	_, err = bc.GetBlock(999)
	if err == nil {
		t.Error("expected error for non-existent block")
	}
}

func TestBlockchain_InvalidBlock(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Invalid height
	invalidBlock := &BlockData{
		Header: BlockHeader{
			Height:       10, // Should be 1
			PreviousHash: bc.currentState.LastBlockHash,
			Timestamp:    time.Now(),
			Proposer:     "validator1",
		},
		Transactions: []Transaction{},
	}
	invalidBlock.Hash = bc.calculateBlockHash(invalidBlock)

	if err := bc.AddBlock(invalidBlock); err == nil {
		t.Error("expected error for invalid block height")
	}

	// Invalid previous hash
	invalidBlock2 := &BlockData{
		Header: BlockHeader{
			Height:       1,
			PreviousHash: "invalid_hash",
			Timestamp:    time.Now(),
			Proposer:     "validator1",
		},
		Transactions: []Transaction{},
	}
	invalidBlock2.Hash = bc.calculateBlockHash(invalidBlock2)

	if err := bc.AddBlock(invalidBlock2); err == nil {
		t.Error("expected error for invalid previous hash")
	}
}

func TestBlockchain_ConcurrentOperations(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Initialize many accounts
	for i := 0; i < 100; i++ {
		account := fmt.Sprintf("account%d", i)
		bc.currentState.AccountBalances[account] = big.NewInt(1000)
	}

	// Concurrent transaction additions
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 10; j++ {
				tx := Transaction{
					ID:        fmt.Sprintf("tx_%d_%d", id, j),
					From:      fmt.Sprintf("account%d", id),
					To:        fmt.Sprintf("account%d", (id+1)%100),
					Amount:    big.NewInt(1),
					Timestamp: time.Now(),
				}
				bc.AddTransaction(tx)
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Check pending transactions
	pending := bc.GetPendingTransactions()
	if len(pending) != 100 {
		t.Errorf("expected 100 pending transactions, got %d", len(pending))
	}
}

func TestBlockchain_StateCheckpoint(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir
	config.StateCheckpoint = 5 // Save state every 5 blocks

	engine := consensus.NewEngine(nil)
	bc, err := NewBlockchain(engine, config)
	if err != nil {
		t.Fatalf("failed to create blockchain: %v", err)
	}
	defer bc.Close()

	// Add multiple blocks
	for i := 1; i <= 10; i++ {
		block := &BlockData{
			Header: BlockHeader{
				Height:       uint64(i),
				PreviousHash: bc.currentState.LastBlockHash,
				Timestamp:    time.Now(),
				Proposer:     fmt.Sprintf("validator%d", i),
			},
			Transactions: []Transaction{},
		}
		block.Hash = bc.calculateBlockHash(block)

		if err := bc.AddBlock(block); err != nil {
			t.Fatalf("failed to add block %d: %v", i, err)
		}
	}

	// Verify state was saved (would need to check database directly)
	if bc.GetHeight() != 10 {
		t.Errorf("expected height 10, got %d", bc.GetHeight())
	}
}

// Benchmark tests
func BenchmarkBlockchain_AddTransaction(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, _ := NewBlockchain(engine, config)
	defer bc.Close()

	// Initialize balance
	bc.currentState.AccountBalances["alice"] = big.NewInt(1000000000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tx := Transaction{
			ID:        fmt.Sprintf("tx%d", i),
			From:      "alice",
			To:        "bob",
			Amount:    big.NewInt(1),
			Timestamp: time.Now(),
		}
		bc.AddTransaction(tx)
	}
}

func BenchmarkBlockchain_CreateBlock(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultBlockchainConfig()
	config.DataDir = tempDir

	engine := consensus.NewEngine(nil)
	bc, _ := NewBlockchain(engine, config)
	defer bc.Close()

	// Pre-add transactions
	bc.currentState.AccountBalances["alice"] = big.NewInt(1000000000)
	for i := 0; i < 1000; i++ {
		tx := Transaction{
			ID:        fmt.Sprintf("tx%d", i),
			From:      "alice",
			To:        "bob",
			Amount:    big.NewInt(1),
			Timestamp: time.Now(),
		}
		bc.pendingTxs = append(bc.pendingTxs, tx)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bc.CreateBlock("validator1")
	}
}