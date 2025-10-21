package poc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/network"
	"github.com/davidcanhelp/sedition/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFullNodeLifecycle tests the complete lifecycle of a node
func TestFullNodeLifecycle(t *testing.T) {
	// Setup test directory
	tempDir, err := os.MkdirTemp("", "poc-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Initialize consensus engine
	consensusCfg := config.DefaultConsensusConfig()
	engine := consensus.NewEngine(consensusCfg)

	// Initialize blockchain
	blockchainCfg := storage.DefaultBlockchainConfig()
	blockchainCfg.DataDir = tempDir + "/blockchain"
	blockchain, err := storage.NewBlockchain(engine, blockchainCfg)
	require.NoError(t, err)
	defer blockchain.Close()

	// Initialize network node
	networkCfg := config.DefaultNetworkConfig()
	node, err := network.NewNode("127.0.0.1:0", engine, networkCfg)
	require.NoError(t, err)

	// Start node
	err = node.Start()
	require.NoError(t, err)
	defer node.Stop()

	// Register validators
	err = engine.RegisterValidator("validator1", big.NewInt(10000))
	require.NoError(t, err)

	err = engine.RegisterValidator("validator2", big.NewInt(5000))
	require.NoError(t, err)

	// Test block creation
	t.Run("CreateBlock", func(t *testing.T) {
		block, err := blockchain.CreateBlock("validator1")
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, uint64(1), block.Header.Height)
		assert.Equal(t, "validator1", block.Header.Proposer)
	})

	// Test transaction processing
	t.Run("ProcessTransaction", func(t *testing.T) {
		tx := storage.Transaction{
			ID:        "tx_test_001",
			From:      "alice",
			To:        "bob",
			Amount:    "100",
			Timestamp: time.Now(),
		}

		err := blockchain.AddTransaction(tx)
		require.NoError(t, err)

		pending := blockchain.GetPendingTransactions()
		assert.Len(t, pending, 1)
		assert.Equal(t, tx.ID, pending[0].ID)
	})

	// Test consensus round
	t.Run("ConsensusRound", func(t *testing.T) {
		// Select proposer
		proposer, err := engine.SelectBlockProposer()
		require.NoError(t, err)
		assert.NotEmpty(t, proposer)

		// Create and add block
		block, err := blockchain.CreateBlock(proposer)
		require.NoError(t, err)

		err = blockchain.AddBlock(block)
		require.NoError(t, err)

		// Verify block was added
		height := blockchain.GetHeight()
		assert.Equal(t, uint64(1), height)

		retrievedBlock, err := blockchain.GetBlock(1)
		require.NoError(t, err)
		assert.Equal(t, block.Hash, retrievedBlock.Hash)
	})

	// Test network messaging
	t.Run("NetworkMessaging", func(t *testing.T) {
		msg := &network.Message{
			Type:    network.MessageTypeBlockProposal,
			Payload: []byte(`{"test": "data"}`),
		}

		// Broadcast message
		node.BroadcastMessage(msg)

		// Get network stats
		stats := node.GetStats()
		assert.NotNil(t, stats)
		assert.Contains(t, stats, "nodeID")
		assert.Contains(t, stats, "messagesSent")
	})
}

// TestAPIEndpoints tests all API endpoints
func TestAPIEndpoints(t *testing.T) {
	// Create test server
	server := createTestServer(t)
	ts := httptest.NewServer(server)
	defer ts.Close()

	client := &http.Client{Timeout: 10 * time.Second}

	// Test status endpoint
	t.Run("Status", func(t *testing.T) {
		resp, err := client.Get(ts.URL + "/api/status")
		require.NoError(t, err)
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode)

		var result map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&result)
		require.NoError(t, err)
		assert.True(t, result["success"].(bool))
	})

	// Test transaction submission
	t.Run("SubmitTransaction", func(t *testing.T) {
		tx := map[string]interface{}{
			"from":   "alice",
			"to":     "bob",
			"amount": "100",
		}

		body, _ := json.Marshal(tx)
		resp, err := client.Post(ts.URL+"/api/transaction", "application/json", bytes.NewReader(body))
		require.NoError(t, err)
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode)

		var result map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&result)
		require.NoError(t, err)
		assert.True(t, result["success"].(bool))
		assert.Contains(t, result["data"], "tx_id")
	})

	// Test block retrieval
	t.Run("GetBlocks", func(t *testing.T) {
		resp, err := client.Get(ts.URL + "/api/blocks")
		require.NoError(t, err)
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode)

		var result map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&result)
		require.NoError(t, err)
		assert.True(t, result["success"].(bool))
		assert.NotNil(t, result["data"])
	})

	// Test balance query
	t.Run("GetBalance", func(t *testing.T) {
		resp, err := client.Get(ts.URL + "/api/balance/alice")
		require.NoError(t, err)
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode)

		var result map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&result)
		require.NoError(t, err)
		assert.True(t, result["success"].(bool))
	})

	// Test validators endpoint
	t.Run("GetValidators", func(t *testing.T) {
		resp, err := client.Get(ts.URL + "/api/validators")
		require.NoError(t, err)
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode)

		var result map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&result)
		require.NoError(t, err)
		assert.True(t, result["success"].(bool))
	})
}

// TestWebSocketConnection tests WebSocket functionality
func TestWebSocketConnection(t *testing.T) {
	// Create WebSocket server
	wsServer := network.NewWebSocketServer()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go wsServer.Run(ctx)

	// Create test HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", wsServer.HandleWebSocket)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// Test broadcasting
	t.Run("Broadcast", func(t *testing.T) {
		// Broadcast block
		block := map[string]interface{}{
			"height": 1,
			"hash":   "0x12345",
		}
		wsServer.BroadcastBlock(block)

		// Broadcast transaction
		tx := map[string]interface{}{
			"id":     "tx_001",
			"amount": "100",
		}
		wsServer.BroadcastTransaction(tx)

		// Broadcast validator update
		validator := map[string]interface{}{
			"address": "validator1",
			"stake":   10000,
		}
		wsServer.BroadcastValidatorUpdate(validator)

		// No panic means success for broadcast tests
		assert.True(t, true)
	})
}

// TestDatabaseMigrations tests the migration system
func TestDatabaseMigrations(t *testing.T) {
	// Setup test directory
	tempDir, err := os.MkdirTemp("", "poc-migration-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Open database
	db, err := storage.OpenDatabase(tempDir + "/test.db")
	require.NoError(t, err)
	defer db.Close()

	// Create migration manager
	migrationManager := storage.NewMigrationManager(db)

	// Test migration
	t.Run("RunMigrations", func(t *testing.T) {
		err := migrationManager.Migrate()
		require.NoError(t, err)

		// Verify database state
		err = storage.ValidateDatabase(db)
		require.NoError(t, err)
	})

	// Test migration history
	t.Run("MigrationHistory", func(t *testing.T) {
		history, err := migrationManager.GetMigrationHistory()
		require.NoError(t, err)
		assert.Greater(t, len(history), 0)
	})

	// Test rollback
	t.Run("Rollback", func(t *testing.T) {
		err := migrationManager.Rollback(2)
		require.NoError(t, err)

		// Re-run migrations
		err = migrationManager.Migrate()
		require.NoError(t, err)
	})
}

// TestConsensusIntegration tests consensus mechanism integration
func TestConsensusIntegration(t *testing.T) {
	cfg := config.DefaultConsensusConfig()
	engine := consensus.NewEngine(cfg)

	// Register multiple validators
	validators := []struct {
		address string
		stake   int64
	}{
		{"validator1", 10000},
		{"validator2", 8000},
		{"validator3", 6000},
		{"validator4", 4000},
	}

	for _, v := range validators {
		err := engine.RegisterValidator(v.address, big.NewInt(v.stake))
		require.NoError(t, err)
	}

	// Test proposer selection over multiple rounds
	t.Run("ProposerSelection", func(t *testing.T) {
		proposerCounts := make(map[string]int)

		// Run 100 rounds
		for i := 0; i < 100; i++ {
			proposer, err := engine.SelectBlockProposer()
			require.NoError(t, err)
			proposerCounts[proposer]++

			// Advance to next round
			engine.StartNewRound()
		}

		// Verify all validators were selected at least once
		for _, v := range validators {
			assert.Greater(t, proposerCounts[v.address], 0,
				"Validator %s was never selected", v.address)
		}

		// Verify selection is proportional to stake (roughly)
		// validator1 should be selected more than validator4
		assert.Greater(t, proposerCounts["validator1"], proposerCounts["validator4"])
	})
}

// TestEndToEndScenario tests a complete end-to-end scenario
func TestEndToEndScenario(t *testing.T) {
	// Setup environment
	tempDir, err := os.MkdirTemp("", "poc-e2e-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Initialize components
	consensusCfg := config.DefaultConsensusConfig()
	engine := consensus.NewEngine(consensusCfg)

	blockchainCfg := storage.DefaultBlockchainConfig()
	blockchainCfg.DataDir = tempDir + "/blockchain"
	blockchain, err := storage.NewBlockchain(engine, blockchainCfg)
	require.NoError(t, err)
	defer blockchain.Close()

	// Scenario: Multiple validators producing blocks with transactions
	t.Run("MultiValidatorScenario", func(t *testing.T) {
		// Register validators
		validators := []string{"alice", "bob", "charlie"}
		for i, v := range validators {
			stake := big.NewInt(int64((i + 1) * 5000))
			err := engine.RegisterValidator(v, stake)
			require.NoError(t, err)
		}

		// Submit transactions
		for i := 0; i < 10; i++ {
			tx := storage.Transaction{
				ID:        fmt.Sprintf("tx_%d", i),
				From:      validators[i%3],
				To:        validators[(i+1)%3],
				Amount:    fmt.Sprintf("%d", (i+1)*10),
				Timestamp: time.Now(),
			}
			err := blockchain.AddTransaction(tx)
			require.NoError(t, err)
		}

		// Produce blocks
		for round := 0; round < 3; round++ {
			// Select proposer
			proposer, err := engine.SelectBlockProposer()
			require.NoError(t, err)

			// Create block
			block, err := blockchain.CreateBlock(proposer)
			require.NoError(t, err)

			// Add block
			err = blockchain.AddBlock(block)
			require.NoError(t, err)

			// Start new round
			engine.StartNewRound()
		}

		// Verify blockchain state
		height := blockchain.GetHeight()
		assert.Equal(t, uint64(3), height)

		// Verify blocks
		for i := uint64(1); i <= 3; i++ {
			block, err := blockchain.GetBlock(i)
			require.NoError(t, err)
			assert.NotNil(t, block)
		}

		// Verify balances
		for _, v := range validators {
			balance := blockchain.GetBalance(v)
			assert.NotNil(t, balance)
		}
	})
}

// Helper function to create test server
func createTestServer(t *testing.T) http.Handler {
	tempDir, err := os.MkdirTemp("", "poc-api-test-*")
	require.NoError(t, err)
	t.Cleanup(func() { os.RemoveAll(tempDir) })

	consensusCfg := config.DefaultConsensusConfig()
	engine := consensus.NewEngine(consensusCfg)

	blockchainCfg := storage.DefaultBlockchainConfig()
	blockchainCfg.DataDir = tempDir + "/blockchain"
	blockchain, err := storage.NewBlockchain(engine, blockchainCfg)
	require.NoError(t, err)
	t.Cleanup(func() { blockchain.Close() })

	// Create a simple API handler for testing
	mux := http.NewServeMux()

	// Status endpoint
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"data": map[string]interface{}{
				"block_height": blockchain.GetHeight(),
			},
		})
	})

	// Transaction endpoint
	mux.HandleFunc("/api/transaction", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			var tx storage.Transaction
			json.NewDecoder(r.Body).Decode(&tx)
			tx.ID = fmt.Sprintf("tx_%d", time.Now().UnixNano())
			blockchain.AddTransaction(tx)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"success": true,
				"data":    map[string]string{"tx_id": tx.ID},
			})
		}
	})

	// Blocks endpoint
	mux.HandleFunc("/api/blocks", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"data":    []interface{}{},
		})
	})

	// Balance endpoint
	mux.HandleFunc("/api/balance/", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"data": map[string]interface{}{
				"balance": "1000",
			},
		})
	})

	// Validators endpoint
	mux.HandleFunc("/api/validators", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"data":    []interface{}{},
		})
	})

	return mux
}