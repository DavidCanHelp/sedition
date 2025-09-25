package rpc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/mempool"
	"github.com/davidcanhelp/sedition/mining"
	"github.com/davidcanhelp/sedition/storage"
	"github.com/davidcanhelp/sedition/wallet"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock state reader for testing
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

func setupTestServer(t *testing.T) (*Server, func()) {
	// Create test components
	engine := consensus.NewEngine(nil)
	blockchain, err := storage.NewBlockchain(engine, &storage.BlockchainConfig{
		DataDir: t.TempDir(),
	})
	require.NoError(t, err)

	state := NewMockStateReader()
	txPool := mempool.NewTxPool(mempool.DefaultTxPoolConfig(), state)

	wallet, err := wallet.NewWallet()
	require.NoError(t, err)

	producer := mining.NewBlockProducer(
		blockchain,
		txPool,
		engine,
		wallet,
		mining.DefaultProducerConfig(),
	)

	// Create server
	config := DefaultServerConfig()
	config.Port = 0 // Use random port for testing
	server := NewServer(blockchain, txPool, producer, wallet, config)

	// Start server
	err = server.Start()
	require.NoError(t, err)

	// Wait for server to start
	time.Sleep(100 * time.Millisecond)

	cleanup := func() {
		server.Stop()
		blockchain.Close()
	}

	return server, cleanup
}

func makeRPCRequest(t *testing.T, url string, method string, params interface{}) *RPCResponse {
	req := RPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		ID:      1,
	}

	if params != nil {
		data, err := json.Marshal(params)
		require.NoError(t, err)
		req.Params = data
	}

	reqBody, err := json.Marshal(req)
	require.NoError(t, err)

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(reqBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	var rpcResp RPCResponse
	err = json.NewDecoder(resp.Body).Decode(&rpcResp)
	require.NoError(t, err)

	return &rpcResp
}

func TestServerStartStop(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	assert.True(t, server.isRunning)

	err := server.Stop()
	require.NoError(t, err)
	assert.False(t, server.isRunning)

	// Starting again should work
	err = server.Start()
	require.NoError(t, err)
	assert.True(t, server.isRunning)
}

func TestGetAccounts(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)
	resp := makeRPCRequest(t, url, "eth_accounts", nil)

	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)

	accounts, ok := resp.Result.([]interface{})
	require.True(t, ok)
	assert.Equal(t, 1, len(accounts))
	assert.Equal(t, server.wallet.GetAddress(), accounts[0])
}

func TestGetBalance(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Set test balance
	address := server.wallet.GetAddress()
	// Note: In a real test, we'd need to properly set the balance in blockchain state

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)
	resp := makeRPCRequest(t, url, "eth_getBalance", []string{address, "latest"})

	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)

	balance, ok := resp.Result.(string)
	require.True(t, ok)
	assert.Equal(t, "0x0", balance) // Should be 0 for new address
}

func TestGetBlockNumber(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)
	resp := makeRPCRequest(t, url, "eth_blockNumber", nil)

	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)

	blockNum, ok := resp.Result.(string)
	require.True(t, ok)
	assert.Equal(t, "0x0", blockNum) // Genesis block
}

func TestSendTransaction(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Prepare transaction params
	params := SendTxParams{
		From:     server.wallet.GetAddress(),
		To:       "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb9",
		Value:    "0x1000",
		Gas:      "0x5208",
		GasPrice: "0x3b9aca00",
	}

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)
	resp := makeRPCRequest(t, url, "eth_sendTransaction", params)

	// Transaction should fail due to insufficient balance, but RPC call should succeed
	if resp.Error != nil {
		assert.Contains(t, resp.Error.Message, "insufficient")
	} else {
		require.NotNil(t, resp.Result)
		txHash, ok := resp.Result.(string)
		require.True(t, ok)
		assert.NotEmpty(t, txHash)
	}
}

func TestGetTransactionByHash(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	// Create a test transaction
	tx := &storage.Transaction{
		Hash:      "test_tx_hash",
		From:      server.wallet.GetAddress(),
		To:        "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb9",
		Value:     big.NewInt(1000),
		Nonce:     0,
		GasLimit:  21000,
		GasPrice:  big.NewInt(1000000000),
		Signature: []byte("test_sig"),
	}

	// Add to pool
	server.txPool.AddTransaction(tx)

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)
	resp := makeRPCRequest(t, url, "eth_getTransactionByHash", []string{tx.Hash})

	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)

	txData, ok := resp.Result.(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, tx.Hash, txData["hash"])
	assert.Equal(t, tx.From, txData["from"])
	assert.Equal(t, tx.To, txData["to"])
}

func TestMiningStatus(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)

	// Check initial mining status
	resp := makeRPCRequest(t, url, "eth_mining", nil)
	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)

	isMining, ok := resp.Result.(bool)
	require.True(t, ok)
	assert.False(t, isMining) // Should be false initially

	// Start mining
	err := server.producer.Start()
	require.NoError(t, err)

	// Check mining status again
	resp = makeRPCRequest(t, url, "eth_mining", nil)
	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)

	isMining, ok = resp.Result.(bool)
	require.True(t, ok)
	assert.True(t, isMining) // Should be true now
}

func TestNetworkMethods(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)

	// Test net_version
	resp := makeRPCRequest(t, url, "net_version", nil)
	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)
	version, ok := resp.Result.(string)
	require.True(t, ok)
	assert.Equal(t, "1337", version)

	// Test net_peerCount
	resp = makeRPCRequest(t, url, "net_peerCount", nil)
	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)
	peerCount, ok := resp.Result.(string)
	require.True(t, ok)
	assert.Equal(t, "0x0", peerCount)

	// Test net_listening
	resp = makeRPCRequest(t, url, "net_listening", nil)
	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)
	listening, ok := resp.Result.(bool)
	require.True(t, ok)
	assert.True(t, listening)
}

func TestWeb3Methods(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)

	// Test web3_clientVersion
	resp := makeRPCRequest(t, url, "web3_clientVersion", nil)
	require.Nil(t, resp.Error)
	require.NotNil(t, resp.Result)
	version, ok := resp.Result.(string)
	require.True(t, ok)
	assert.Equal(t, "Sedition/v1.0.0", version)
}

func TestInvalidMethod(t *testing.T) {
	server, cleanup := setupTestServer(t)
	defer cleanup()

	url := fmt.Sprintf("http://%s:%d", server.config.Host, server.config.Port)
	resp := makeRPCRequest(t, url, "invalid_method", nil)

	require.NotNil(t, resp.Error)
	assert.Contains(t, resp.Error.Message, "method not found")
}