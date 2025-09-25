// Package main implements a demo server for the PoC blockchain
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"math/big"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/contribution"
	"github.com/davidcanhelp/sedition/network"
	"github.com/davidcanhelp/sedition/storage"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Server represents the demo server
type Server struct {
	mu              sync.RWMutex
	engine          *consensus.Engine
	blockchain      *storage.Blockchain
	node            *network.Node
	httpServer      *http.Server
	metrics         *Metrics
	validatorAddr   string
	isRunning       bool
}

// Metrics holds Prometheus metrics
type Metrics struct {
	blocksCreated      prometheus.Counter
	transactionsAdded  prometheus.Counter
	peersConnected     prometheus.Gauge
	blockHeight        prometheus.Gauge
	consensusRounds    prometheus.Counter
	apiRequests        *prometheus.CounterVec
	responseTime       *prometheus.HistogramVec
}

// APIResponse represents a standard API response
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// NodeStatus represents the current node status
type NodeStatus struct {
	NodeID          string                 `json:"node_id"`
	IsValidator     bool                   `json:"is_validator"`
	ValidatorAddr   string                 `json:"validator_address,omitempty"`
	BlockHeight     uint64                 `json:"block_height"`
	PeerCount       int                    `json:"peer_count"`
	PendingTxCount  int                    `json:"pending_tx_count"`
	NetworkStats    map[string]interface{} `json:"network_stats"`
	ConsensusActive bool                   `json:"consensus_active"`
	Uptime          string                 `json:"uptime"`
}

var startTime = time.Now()

func main() {
	var (
		httpAddr      = flag.String("http", ":8080", "HTTP server address")
		p2pAddr       = flag.String("p2p", ":8545", "P2P network address")
		dataDir       = flag.String("data", "./data", "Data directory")
		bootstrapNode = flag.String("bootstrap", "", "Bootstrap node address")
		validatorMode = flag.Bool("validator", false, "Run as validator")
		validatorName = flag.String("name", "validator1", "Validator name")
		stake         = flag.Int64("stake", 10000, "Initial stake amount")
	)
	flag.Parse()

	log.Println("Starting PoC Blockchain Demo Server...")

	// Create server
	server, err := NewServer(*dataDir, *p2pAddr, *validatorMode, *validatorName, *stake)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	// Connect to bootstrap node if provided
	if *bootstrapNode != "" {
		log.Printf("Connecting to bootstrap node: %s", *bootstrapNode)
		if err := server.node.Connect(*bootstrapNode); err != nil {
			log.Printf("Warning: Failed to connect to bootstrap node: %v", err)
		}
	}

	// Start HTTP server
	server.StartHTTP(*httpAddr)

	// Handle shutdown gracefully
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down...")
	server.Shutdown()
}

// NewServer creates a new demo server
func NewServer(dataDir, p2pAddr string, isValidator bool, validatorName string, stake int64) (*Server, error) {
	// Initialize consensus engine
	consensusCfg := config.DefaultConsensusConfig()
	engine := consensus.NewEngine(consensusCfg)

	// Initialize blockchain
	blockchainCfg := storage.DefaultBlockchainConfig()
	blockchainCfg.DataDir = dataDir + "/blockchain"
	blockchain, err := storage.NewBlockchain(engine, blockchainCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create blockchain: %w", err)
	}

	// Initialize network node
	networkCfg := config.DefaultNetworkConfig()
	node, err := network.NewNode(p2pAddr, engine, networkCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create network node: %w", err)
	}

	// Start network node
	if err := node.Start(); err != nil {
		return nil, fmt.Errorf("failed to start network node: %w", err)
	}

	server := &Server{
		engine:     engine,
		blockchain: blockchain,
		node:       node,
		metrics:    initMetrics(),
		isRunning:  true,
	}

	// Register as validator if requested
	if isValidator {
		if err := engine.RegisterValidator(validatorName, big.NewInt(stake)); err != nil {
			return nil, fmt.Errorf("failed to register validator: %w", err)
		}
		server.validatorAddr = validatorName
		log.Printf("Registered as validator: %s with stake %d", validatorName, stake)
	}

	// Start background tasks
	go server.blockProducerLoop()
	go server.metricsUpdater()

	return server, nil
}

// initMetrics initializes Prometheus metrics
func initMetrics() *Metrics {
	m := &Metrics{
		blocksCreated: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "poc_blocks_created_total",
			Help: "Total number of blocks created",
		}),
		transactionsAdded: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "poc_transactions_added_total",
			Help: "Total number of transactions added",
		}),
		peersConnected: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "poc_peers_connected",
			Help: "Number of connected peers",
		}),
		blockHeight: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "poc_block_height",
			Help: "Current blockchain height",
		}),
		consensusRounds: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "poc_consensus_rounds_total",
			Help: "Total consensus rounds",
		}),
		apiRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "poc_api_requests_total",
			Help: "Total API requests",
		}, []string{"method", "endpoint"}),
		responseTime: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "poc_api_response_time_seconds",
			Help: "API response time",
		}, []string{"method", "endpoint"}),
	}

	// Register metrics
	prometheus.MustRegister(m.blocksCreated, m.transactionsAdded, m.peersConnected,
		m.blockHeight, m.consensusRounds, m.apiRequests, m.responseTime)

	return m
}

// StartHTTP starts the HTTP server
func (s *Server) StartHTTP(addr string) {
	mux := http.NewServeMux()

	// API endpoints
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/blocks", s.handleGetBlocks)
	mux.HandleFunc("/api/block/", s.handleGetBlock)
	mux.HandleFunc("/api/transaction", s.handleTransaction)
	mux.HandleFunc("/api/balance/", s.handleGetBalance)
	mux.HandleFunc("/api/validators", s.handleGetValidators)
	mux.HandleFunc("/api/contribute", s.handleContribution)
	mux.HandleFunc("/api/peers", s.handleGetPeers)

	// Metrics endpoint
	mux.Handle("/metrics", promhttp.Handler())

	// Web UI
	mux.HandleFunc("/", s.handleWebUI)
	mux.HandleFunc("/static/", s.handleStatic)

	s.httpServer = &http.Server{
		Addr:    addr,
		Handler: s.middleware(mux),
	}

	go func() {
		log.Printf("HTTP server listening on %s", addr)
		if err := s.httpServer.ListenAndServe(); err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()
}

// middleware adds logging and metrics to HTTP requests
func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Add CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			return
		}

		// Log request
		log.Printf("%s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

		// Track metrics
		s.metrics.apiRequests.WithLabelValues(r.Method, r.URL.Path).Inc()

		next.ServeHTTP(w, r)

		// Record response time
		duration := time.Since(start).Seconds()
		s.metrics.responseTime.WithLabelValues(r.Method, r.URL.Path).Observe(duration)
	})
}

// API Handlers

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	networkStats := s.node.GetStats()
	peerCount := 0
	if pc, ok := networkStats["peerCount"].(int); ok {
		peerCount = pc
	}

	status := NodeStatus{
		NodeID:          networkStats["nodeID"].(string),
		IsValidator:     s.validatorAddr != "",
		ValidatorAddr:   s.validatorAddr,
		BlockHeight:     s.blockchain.GetHeight(),
		PeerCount:       peerCount,
		PendingTxCount:  len(s.blockchain.GetPendingTransactions()),
		NetworkStats:    networkStats,
		ConsensusActive: s.isRunning,
		Uptime:          time.Since(startTime).String(),
	}

	s.sendJSON(w, APIResponse{Success: true, Data: status})
}

func (s *Server) handleGetBlocks(w http.ResponseWriter, r *http.Request) {
	height := s.blockchain.GetHeight()
	blocks := make([]storage.BlockData, 0)

	// Get last 10 blocks
	start := uint64(0)
	if height > 10 {
		start = height - 10
	}

	for i := start; i <= height; i++ {
		block, err := s.blockchain.GetBlock(i)
		if err == nil {
			blocks = append(blocks, *block)
		}
	}

	s.sendJSON(w, APIResponse{Success: true, Data: blocks})
}

func (s *Server) handleGetBlock(w http.ResponseWriter, r *http.Request) {
	// Parse block height from URL
	var height uint64
	fmt.Sscanf(r.URL.Path, "/api/block/%d", &height)

	block, err := s.blockchain.GetBlock(height)
	if err != nil {
		s.sendJSON(w, APIResponse{Success: false, Error: err.Error()})
		return
	}

	s.sendJSON(w, APIResponse{Success: true, Data: block})
}

func (s *Server) handleTransaction(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		s.sendJSON(w, APIResponse{Success: false, Error: "Method not allowed"})
		return
	}

	var tx storage.Transaction
	if err := json.NewDecoder(r.Body).Decode(&tx); err != nil {
		s.sendJSON(w, APIResponse{Success: false, Error: "Invalid transaction data"})
		return
	}

	// Generate transaction ID if not provided
	if tx.ID == "" {
		tx.ID = fmt.Sprintf("tx_%d", time.Now().UnixNano())
	}
	tx.Timestamp = time.Now()

	if err := s.blockchain.AddTransaction(tx); err != nil {
		s.sendJSON(w, APIResponse{Success: false, Error: err.Error()})
		return
	}

	s.metrics.transactionsAdded.Inc()
	s.sendJSON(w, APIResponse{Success: true, Data: map[string]string{"tx_id": tx.ID}})
}

func (s *Server) handleGetBalance(w http.ResponseWriter, r *http.Request) {
	var address string
	fmt.Sscanf(r.URL.Path, "/api/balance/%s", &address)

	balance := s.blockchain.GetBalance(address)
	s.sendJSON(w, APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"address": address,
			"balance": balance.String(),
		},
	})
}

func (s *Server) handleGetValidators(w http.ResponseWriter, r *http.Request) {
	// This would integrate with the consensus engine
	validators := make([]map[string]interface{}, 0)

	// Add mock data for now
	if s.validatorAddr != "" {
		validators = append(validators, map[string]interface{}{
			"address":    s.validatorAddr,
			"stake":      "10000",
			"reputation": 5.0,
			"active":     true,
		})
	}

	s.sendJSON(w, APIResponse{Success: true, Data: validators})
}

func (s *Server) handleContribution(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		s.sendJSON(w, APIResponse{Success: false, Error: "Method not allowed"})
		return
	}

	if s.validatorAddr == "" {
		s.sendJSON(w, APIResponse{Success: false, Error: "Not a validator"})
		return
	}

	var contrib contribution.Contribution
	if err := json.NewDecoder(r.Body).Decode(&contrib); err != nil {
		s.sendJSON(w, APIResponse{Success: false, Error: "Invalid contribution data"})
		return
	}

	contrib.Timestamp = time.Now()
	if err := s.engine.SubmitContribution(s.validatorAddr, contrib); err != nil {
		s.sendJSON(w, APIResponse{Success: false, Error: err.Error()})
		return
	}

	s.sendJSON(w, APIResponse{Success: true, Data: "Contribution submitted"})
}

func (s *Server) handleGetPeers(w http.ResponseWriter, r *http.Request) {
	stats := s.node.GetStats()
	s.sendJSON(w, APIResponse{Success: true, Data: stats["peers"]})
}

// handleWebUI serves the web interface
func (s *Server) handleWebUI(w http.ResponseWriter, r *http.Request) {
	tmpl := `<!DOCTYPE html>
<html>
<head>
    <title>PoC Blockchain Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f23; color: #ccc; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #00ff00; margin-bottom: 20px; font-size: 2.5em; text-shadow: 0 0 10px #00ff00; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #1a1a2e; border: 1px solid #16213e; border-radius: 8px; padding: 20px; }
        .card h2 { color: #00ff00; margin-bottom: 15px; font-size: 1.2em; }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #2a2a3e; }
        .stat:last-child { border-bottom: none; }
        .label { color: #888; }
        .value { color: #fff; font-weight: bold; }
        .blocks { margin-top: 30px; }
        .block { background: #16213e; border-left: 3px solid #00ff00; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
        .block-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .block-height { color: #00ff00; font-weight: bold; }
        .tx-form { display: flex; gap: 10px; margin-top: 20px; }
        input, button { padding: 10px; border: 1px solid #2a2a3e; background: #1a1a2e; color: #fff; border-radius: 4px; }
        button { background: #00ff00; color: #000; cursor: pointer; font-weight: bold; }
        button:hover { background: #00cc00; }
        .loading { animation: pulse 1s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔗 Proof of Contribution Blockchain</h1>

        <div class="grid">
            <div class="card">
                <h2>📊 Node Status</h2>
                <div id="nodeStatus" class="loading">Loading...</div>
            </div>

            <div class="card">
                <h2>🌐 Network</h2>
                <div id="networkStatus" class="loading">Loading...</div>
            </div>

            <div class="card">
                <h2>💰 Send Transaction</h2>
                <div class="tx-form">
                    <input type="text" id="txFrom" placeholder="From">
                    <input type="text" id="txTo" placeholder="To">
                    <input type="number" id="txAmount" placeholder="Amount">
                    <button onclick="sendTransaction()">Send</button>
                </div>
                <div id="txResult"></div>
            </div>
        </div>

        <div class="blocks">
            <h2 style="color: #00ff00; margin-bottom: 20px;">📦 Recent Blocks</h2>
            <div id="blockList" class="loading">Loading blocks...</div>
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.success) {
                    updateNodeStatus(data.data);
                }
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }

        async function fetchBlocks() {
            try {
                const response = await fetch('/api/blocks');
                const data = await response.json();
                if (data.success) {
                    updateBlockList(data.data);
                }
            } catch (error) {
                console.error('Error fetching blocks:', error);
            }
        }

        function updateNodeStatus(status) {
            const html = ` + "`" + `
                <div class="stat"><span class="label">Node ID:</span><span class="value">${status.node_id.substring(0, 8)}...</span></div>
                <div class="stat"><span class="label">Block Height:</span><span class="value">${status.block_height}</span></div>
                <div class="stat"><span class="label">Pending Txs:</span><span class="value">${status.pending_tx_count}</span></div>
                <div class="stat"><span class="label">Uptime:</span><span class="value">${status.uptime}</span></div>
                <div class="stat"><span class="label">Validator:</span><span class="value">${status.is_validator ? '✓' : '✗'}</span></div>
            ` + "`" + `;
            document.getElementById('nodeStatus').innerHTML = html;
            document.getElementById('nodeStatus').classList.remove('loading');

            const netHtml = ` + "`" + `
                <div class="stat"><span class="label">Connected Peers:</span><span class="value">${status.peer_count}</span></div>
                <div class="stat"><span class="label">Messages Sent:</span><span class="value">${status.network_stats.messagesSent || 0}</span></div>
                <div class="stat"><span class="label">Messages Received:</span><span class="value">${status.network_stats.messagesReceived || 0}</span></div>
                <div class="stat"><span class="label">Consensus:</span><span class="value">${status.consensus_active ? 'Active' : 'Inactive'}</span></div>
            ` + "`" + `;
            document.getElementById('networkStatus').innerHTML = netHtml;
            document.getElementById('networkStatus').classList.remove('loading');
        }

        function updateBlockList(blocks) {
            let html = '';
            blocks.reverse().forEach(block => {
                const time = new Date(block.header.timestamp).toLocaleTimeString();
                html += ` + "`" + `
                    <div class="block">
                        <div class="block-header">
                            <span class="block-height">Block #${block.header.height}</span>
                            <span>${time}</span>
                        </div>
                        <div class="stat"><span class="label">Hash:</span><span class="value" style="font-size: 0.9em">${block.hash.substring(0, 16)}...</span></div>
                        <div class="stat"><span class="label">Proposer:</span><span class="value">${block.header.proposer}</span></div>
                        <div class="stat"><span class="label">Transactions:</span><span class="value">${block.transactions.length}</span></div>
                    </div>
                ` + "`" + `;
            });
            document.getElementById('blockList').innerHTML = html || '<p>No blocks yet</p>';
            document.getElementById('blockList').classList.remove('loading');
        }

        async function sendTransaction() {
            const from = document.getElementById('txFrom').value;
            const to = document.getElementById('txTo').value;
            const amount = document.getElementById('txAmount').value;

            if (!from || !to || !amount) {
                alert('Please fill all fields');
                return;
            }

            try {
                const response = await fetch('/api/transaction', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({from, to, amount})
                });
                const data = await response.json();

                const resultDiv = document.getElementById('txResult');
                if (data.success) {
                    resultDiv.innerHTML = '<p style="color: #00ff00">Transaction sent!</p>';
                    document.getElementById('txFrom').value = '';
                    document.getElementById('txTo').value = '';
                    document.getElementById('txAmount').value = '';
                } else {
                    resultDiv.innerHTML = ` + "`" + `<p style="color: #ff0000">Error: ${data.error}</p>` + "`" + `;
                }
            } catch (error) {
                console.error('Error sending transaction:', error);
            }
        }

        // Auto-refresh
        setInterval(fetchStatus, 5000);
        setInterval(fetchBlocks, 10000);

        // Initial load
        fetchStatus();
        fetchBlocks();
    </script>
</body>
</html>`

	t, _ := template.New("index").Parse(tmpl)
	t.Execute(w, nil)
}

func (s *Server) handleStatic(w http.ResponseWriter, r *http.Request) {
	// Serve static files if needed
	http.NotFound(w, r)
}

// Helper functions

func (s *Server) sendJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

// Background tasks

func (s *Server) blockProducerLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for s.isRunning {
		select {
		case <-ticker.C:
			if s.validatorAddr != "" {
				s.tryProduceBlock()
			}
		}
	}
}

func (s *Server) tryProduceBlock() {
	// Check if we're selected as block proposer
	proposer, err := s.engine.SelectBlockProposer()
	if err != nil || proposer != s.validatorAddr {
		return
	}

	log.Printf("Selected as block proposer!")

	// Create new block
	block, err := s.blockchain.CreateBlock(s.validatorAddr)
	if err != nil {
		log.Printf("Failed to create block: %v", err)
		return
	}

	// Add block to chain
	if err := s.blockchain.AddBlock(block); err != nil {
		log.Printf("Failed to add block: %v", err)
		return
	}

	// Broadcast block to network
	blockData, _ := json.Marshal(block)
	msg := &network.Message{
		Type:    network.MessageTypeBlockProposal,
		Payload: blockData,
	}
	s.node.BroadcastMessage(msg)

	s.metrics.blocksCreated.Inc()
	s.metrics.consensusRounds.Inc()
	log.Printf("Created block #%d", block.Header.Height)
}

func (s *Server) metricsUpdater() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for s.isRunning {
		select {
		case <-ticker.C:
			s.updateMetrics()
		}
	}
}

func (s *Server) updateMetrics() {
	s.metrics.blockHeight.Set(float64(s.blockchain.GetHeight()))

	stats := s.node.GetStats()
	if peerCount, ok := stats["peerCount"].(int); ok {
		s.metrics.peersConnected.Set(float64(peerCount))
	}
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown() {
	s.mu.Lock()
	s.isRunning = false
	s.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if s.httpServer != nil {
		s.httpServer.Shutdown(ctx)
	}

	if s.node != nil {
		s.node.Stop()
	}

	if s.blockchain != nil {
		s.blockchain.Close()
	}
}