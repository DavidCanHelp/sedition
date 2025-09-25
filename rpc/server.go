package rpc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/mempool"
	"github.com/davidcanhelp/sedition/mining"
	"github.com/davidcanhelp/sedition/storage"
	"github.com/davidcanhelp/sedition/wallet"
)

// Server implements the JSON-RPC server
type Server struct {
	mu sync.RWMutex

	// Core components
	blockchain *storage.Blockchain
	txPool     *mempool.TxPool
	producer   *mining.BlockProducer
	wallet     *wallet.Wallet

	// Configuration
	config ServerConfig

	// HTTP server
	httpServer *http.Server
	isRunning  bool
}

// ServerConfig holds RPC server configuration
type ServerConfig struct {
	Host     string
	Port     int
	MaxConns int
}

// DefaultServerConfig returns default configuration
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Host:     "127.0.0.1",
		Port:     8545,
		MaxConns: 100,
	}
}

// NewServer creates a new RPC server
func NewServer(
	blockchain *storage.Blockchain,
	txPool *mempool.TxPool,
	producer *mining.BlockProducer,
	wallet *wallet.Wallet,
	config ServerConfig,
) *Server {
	return &Server{
		blockchain: blockchain,
		txPool:     txPool,
		producer:   producer,
		wallet:     wallet,
		config:     config,
	}
}

// RPCRequest represents a JSON-RPC request
type RPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
	ID      interface{}     `json:"id"`
}

// RPCResponse represents a JSON-RPC response
type RPCResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result,omitempty"`
	Error   *RPCError   `json:"error,omitempty"`
	ID      interface{} `json:"id"`
}

// RPCError represents a JSON-RPC error
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Standard JSON-RPC error codes
const (
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603
)

// Start starts the RPC server
func (s *Server) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.isRunning {
		return fmt.Errorf("server already running")
	}

	// Create HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRequest)

	s.httpServer = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", s.config.Host, s.config.Port),
		Handler: mux,
	}

	// Start listening
	go func() {
		fmt.Printf("RPC server listening on %s:%d\n", s.config.Host, s.config.Port)
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("RPC server error: %v\n", err)
		}
	}()

	s.isRunning = true
	return nil
}

// Stop stops the RPC server
func (s *Server) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.isRunning {
		return fmt.Errorf("server not running")
	}

	ctx := context.Background()
	if err := s.httpServer.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown server: %w", err)
	}

	s.isRunning = false
	return nil
}

// handleRequest handles incoming RPC requests
func (s *Server) handleRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST method allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request
	var req RPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.sendError(w, nil, ParseError, "Parse error")
		return
	}

	// Handle method
	result, err := s.handleMethod(req.Method, req.Params)
	if err != nil {
		s.sendError(w, req.ID, InternalError, err.Error())
		return
	}

	// Send response
	s.sendResponse(w, req.ID, result)
}

// handleMethod routes the method call
func (s *Server) handleMethod(method string, params json.RawMessage) (interface{}, error) {
	switch method {
	// Account methods
	case "eth_accounts":
		return s.getAccounts()
	case "eth_getBalance":
		return s.getBalance(params)

	// Transaction methods
	case "eth_sendTransaction":
		return s.sendTransaction(params)
	case "eth_getTransactionByHash":
		return s.getTransactionByHash(params)
	case "eth_getTransactionReceipt":
		return s.getTransactionReceipt(params)
	case "eth_sendRawTransaction":
		return s.sendRawTransaction(params)

	// Block methods
	case "eth_blockNumber":
		return s.getBlockNumber()
	case "eth_getBlockByNumber":
		return s.getBlockByNumber(params)
	case "eth_getBlockByHash":
		return s.getBlockByHash(params)

	// Mining methods
	case "eth_mining":
		return s.isMining()
	case "eth_hashrate":
		return s.getHashrate()

	// Gas methods
	case "eth_gasPrice":
		return s.getGasPrice()
	case "eth_estimateGas":
		return s.estimateGas(params)

	// Network methods
	case "net_version":
		return s.getNetworkVersion()
	case "net_peerCount":
		return s.getPeerCount()
	case "net_listening":
		return s.isListening()

	// Web3 methods
	case "web3_clientVersion":
		return s.getClientVersion()
	case "web3_sha3":
		return s.getSha3(params)

	default:
		return nil, fmt.Errorf("method not found: %s", method)
	}
}

// Account methods

func (s *Server) getAccounts() ([]string, error) {
	if s.wallet == nil {
		return []string{}, nil
	}
	return []string{s.wallet.GetAddress()}, nil
}

type GetBalanceParams struct {
	Address string `json:"address"`
	Block   string `json:"block,omitempty"`
}

func (s *Server) getBalance(params json.RawMessage) (string, error) {
	var p []string
	if err := json.Unmarshal(params, &p); err != nil {
		return "", err
	}
	if len(p) < 1 {
		return "", fmt.Errorf("missing address parameter")
	}

	balance := s.blockchain.GetBalance(p[0])
	return fmt.Sprintf("0x%x", balance), nil
}

// Transaction methods

type SendTxParams struct {
	From     string   `json:"from"`
	To       string   `json:"to"`
	Value    string   `json:"value,omitempty"`
	Gas      string   `json:"gas,omitempty"`
	GasPrice string   `json:"gasPrice,omitempty"`
	Data     string   `json:"data,omitempty"`
	Nonce    string   `json:"nonce,omitempty"`
}

func (s *Server) sendTransaction(params json.RawMessage) (string, error) {
	// Handle both array and single object formats
	var p SendTxParams

	// First try to unmarshal as array
	var paramsArray []SendTxParams
	if err := json.Unmarshal(params, &paramsArray); err == nil && len(paramsArray) > 0 {
		p = paramsArray[0]
	} else {
		// If that fails, try as single object
		if err := json.Unmarshal(params, &p); err != nil {
			return "", fmt.Errorf("invalid params format: %v", err)
		}
	}

	// Parse value
	value := big.NewInt(0)
	if p.Value != "" {
		value, _ = new(big.Int).SetString(p.Value, 0)
	}

	// Parse gas
	gasLimit := uint64(21000)
	if p.Gas != "" {
		gas, _ := new(big.Int).SetString(p.Gas, 0)
		gasLimit = gas.Uint64()
	}

	// Parse gas price
	gasPrice := big.NewInt(1000000000) // 1 Gwei default
	if p.GasPrice != "" {
		gasPrice, _ = new(big.Int).SetString(p.GasPrice, 0)
	}

	// Get nonce
	nonce := s.txPool.GetNonce(p.From)

	// Create transaction
	tx, err := s.wallet.CreateTransaction(p.To, value, nonce, gasLimit, gasPrice, nil)
	if err != nil {
		return "", err
	}

	// Add to pool
	if err := s.txPool.AddTransaction(tx); err != nil {
		return "", err
	}

	return tx.Hash, nil
}

func (s *Server) sendRawTransaction(params json.RawMessage) (string, error) {
	var p []string
	if err := json.Unmarshal(params, &p); err != nil {
		return "", err
	}
	if len(p) < 1 {
		return "", fmt.Errorf("missing raw transaction parameter")
	}

	// Decode the raw transaction hex string
	rawTxHex := p[0]
	if len(rawTxHex) < 2 || rawTxHex[:2] != "0x" {
		return "", fmt.Errorf("invalid hex string")
	}

	// In a real implementation, we would:
	// 1. Decode the hex string to bytes
	// 2. Unmarshal the transaction structure
	// 3. Verify the signature
	// 4. Submit to the transaction pool

	// For now, create a mock transaction
	// This would normally be decoded from the raw hex
	tx := &storage.Transaction{
		Hash:      fmt.Sprintf("0x%x", time.Now().UnixNano()),
		From:      s.wallet.GetAddress(), // Would be recovered from signature
		To:        "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb9",
		Value:     big.NewInt(1000),
		Nonce:     s.txPool.GetNonce(s.wallet.GetAddress()),
		GasLimit:  21000,
		GasPrice:  big.NewInt(1000000000),
		Timestamp: time.Now(),
		Signature: []byte("mock_signature"),
	}

	// Add to pool
	if err := s.txPool.AddTransaction(tx); err != nil {
		return "", err
	}

	return tx.Hash, nil
}

func (s *Server) getTransactionByHash(params json.RawMessage) (interface{}, error) {
	var p []string
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	if len(p) < 1 {
		return nil, fmt.Errorf("missing hash parameter")
	}

	tx := s.txPool.GetTransaction(p[0])
	if tx == nil {
		return nil, nil
	}

	return map[string]interface{}{
		"hash":      tx.Hash,
		"from":      tx.From,
		"to":        tx.To,
		"value":     fmt.Sprintf("0x%x", tx.Value),
		"nonce":     fmt.Sprintf("0x%x", tx.Nonce),
		"gas":       fmt.Sprintf("0x%x", tx.GasLimit),
		"gasPrice":  fmt.Sprintf("0x%x", tx.GasPrice),
		"input":     fmt.Sprintf("0x%x", tx.Data),
	}, nil
}

func (s *Server) getTransactionReceipt(params json.RawMessage) (interface{}, error) {
	var p []string
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	if len(p) < 1 {
		return nil, fmt.Errorf("missing hash parameter")
	}

	txHash := p[0]

	// In a real implementation, we would search through blocks for the transaction
	// For now, we'll create a mock receipt if the transaction exists in the pool
	tx := s.txPool.GetTransaction(txHash)
	if tx == nil {
		// Transaction not in pool, might be mined
		// Check recent blocks (simplified - just return nil for now)
		return nil, nil
	}

	// Create a mock receipt for pending transactions
	receipt := map[string]interface{}{
		"transactionHash":   tx.Hash,
		"transactionIndex":  "0x0", // Would be actual index in block
		"blockHash":         nil,   // null for pending
		"blockNumber":       nil,   // null for pending
		"from":              tx.From,
		"to":                tx.To,
		"cumulativeGasUsed": fmt.Sprintf("0x%x", tx.GasLimit),
		"gasUsed":           fmt.Sprintf("0x%x", tx.GasLimit),
		"contractAddress":   nil, // null unless contract creation
		"logs":              []interface{}{},
		"logsBloom":         "0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
		"status":            "0x1", // Success
	}

	return receipt, nil
}

// Block methods

func (s *Server) getBlockNumber() (string, error) {
	height := s.blockchain.GetHeight()
	return fmt.Sprintf("0x%x", height), nil
}

func (s *Server) getBlockByNumber(params json.RawMessage) (interface{}, error) {
	var p []interface{}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	if len(p) < 1 {
		return nil, fmt.Errorf("missing block number parameter")
	}

	// Parse block number
	var blockNum uint64
	switch v := p[0].(type) {
	case string:
		if v == "latest" {
			blockNum = s.blockchain.GetHeight()
		} else {
			n, _ := new(big.Int).SetString(v, 0)
			blockNum = n.Uint64()
		}
	case float64:
		blockNum = uint64(v)
	}

	block, err := s.blockchain.GetBlock(blockNum)
	if err != nil {
		return nil, nil
	}

	// Include full transactions if requested
	fullTx := false
	if len(p) > 1 {
		fullTx, _ = p[1].(bool)
	}

	return s.formatBlock(block, fullTx), nil
}

func (s *Server) getBlockByHash(params json.RawMessage) (interface{}, error) {
	// TODO: Implement block retrieval by hash
	return nil, fmt.Errorf("not implemented")
}

func (s *Server) formatBlock(block *storage.BlockData, fullTx bool) map[string]interface{} {
	txs := []interface{}{}
	if fullTx {
		for _, tx := range block.Transactions {
			txs = append(txs, map[string]interface{}{
				"hash":      tx.Hash,
				"from":      tx.From,
				"to":        tx.To,
				"value":     fmt.Sprintf("0x%x", tx.Value),
				"nonce":     fmt.Sprintf("0x%x", tx.Nonce),
				"gas":       fmt.Sprintf("0x%x", tx.GasLimit),
				"gasPrice":  fmt.Sprintf("0x%x", tx.GasPrice),
				"input":     fmt.Sprintf("0x%x", tx.Data),
			})
		}
	} else {
		for _, tx := range block.Transactions {
			txs = append(txs, tx.Hash)
		}
	}

	return map[string]interface{}{
		"number":           fmt.Sprintf("0x%x", block.Header.Height),
		"hash":             block.Hash,
		"parentHash":       block.Header.PreviousHash,
		"timestamp":        fmt.Sprintf("0x%x", block.Header.Timestamp.Unix()),
		"miner":            block.Header.Proposer,
		"stateRoot":        block.Header.StateRoot,
		"transactionsRoot": block.Header.TxRoot,
		"transactions":     txs,
	}
}

// Mining methods

func (s *Server) isMining() (bool, error) {
	if s.producer == nil {
		return false, nil
	}
	return s.producer.IsRunning(), nil
}

func (s *Server) getHashrate() (string, error) {
	// Simplified - return 0 for PoS
	return "0x0", nil
}

// Gas methods

func (s *Server) getGasPrice() (string, error) {
	// Return minimum gas price
	return "0x3b9aca00", nil // 1 Gwei
}

func (s *Server) estimateGas(params json.RawMessage) (string, error) {
	// Simplified estimation
	return "0x5208", nil // 21000 gas
}

// Network methods

func (s *Server) getNetworkVersion() (string, error) {
	return "1337", nil // Custom network ID
}

func (s *Server) getPeerCount() (string, error) {
	// TODO: Implement once P2P is added
	return "0x0", nil
}

func (s *Server) isListening() (bool, error) {
	return s.isRunning, nil
}

// Web3 methods

func (s *Server) getClientVersion() (string, error) {
	return "Sedition/v1.0.0", nil
}

func (s *Server) getSha3(params json.RawMessage) (string, error) {
	var p []string
	if err := json.Unmarshal(params, &p); err != nil {
		return "", err
	}
	if len(p) < 1 {
		return "", fmt.Errorf("missing data parameter")
	}

	// In Ethereum, web3_sha3 actually uses Keccak-256, not SHA3-256
	// For this implementation, we'll use SHA256 as a placeholder
	// In production, use golang.org/x/crypto/sha3 for Keccak-256

	data := p[0]
	if len(data) >= 2 && data[:2] == "0x" {
		data = data[2:]
	}

	// Decode hex string
	bytes, err := hex.DecodeString(data)
	if err != nil {
		return "", fmt.Errorf("invalid hex string: %v", err)
	}

	// Calculate hash (using SHA256 as placeholder for Keccak-256)
	hash := sha256.Sum256(bytes)

	return fmt.Sprintf("0x%x", hash), nil
}

// Helper methods

func (s *Server) sendResponse(w http.ResponseWriter, id interface{}, result interface{}) {
	resp := RPCResponse{
		JSONRPC: "2.0",
		Result:  result,
		ID:      id,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) sendError(w http.ResponseWriter, id interface{}, code int, message string) {
	resp := RPCResponse{
		JSONRPC: "2.0",
		Error: &RPCError{
			Code:    code,
			Message: message,
		},
		ID: id,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}