package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/big"
	"os"
	"os/signal"
	"syscall"
	"time"

	pocconfig "github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/mempool"
	"github.com/davidcanhelp/sedition/mining"
	"github.com/davidcanhelp/sedition/network"
	"github.com/davidcanhelp/sedition/rpc"
	"github.com/davidcanhelp/sedition/storage"
	"github.com/davidcanhelp/sedition/validator"
	"github.com/davidcanhelp/sedition/wallet"
)

// NodeConfig holds the node configuration
type NodeConfig struct {
	DataDir      string
	RPCHost      string
	RPCPort      int
	P2PPort      int
	Mining       bool
	WalletFile   string
	Genesis      bool
	BootstrapNodes []string
}

// Node represents a blockchain node
type Node struct {
	config *NodeConfig

	// Core components
	blockchain  *storage.Blockchain
	consensus   *consensus.Engine
	txPool      *mempool.TxPool
	wallet      *wallet.Wallet
	producer    *mining.BlockProducer
	broadcaster *network.Broadcaster
	rpcServer   *rpc.Server

	// Network
	peerListener *network.PeerListener
}

func main() {
	// Parse command line flags
	config := parseFlags()

	// Create and start node
	node, err := NewNode(config)
	if err != nil {
		log.Fatalf("Failed to create node: %v", err)
	}

	if err := node.Start(); err != nil {
		log.Fatalf("Failed to start node: %v", err)
	}

	fmt.Println("Sedition blockchain node started successfully")
	fmt.Printf("RPC server: http://%s:%d\n", config.RPCHost, config.RPCPort)
	fmt.Printf("P2P port: %d\n", config.P2PPort)
	fmt.Printf("Mining: %v\n", config.Mining)
	fmt.Printf("Wallet address: %s\n", node.wallet.GetAddress())

	// Wait for interrupt signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	fmt.Println("\nShutting down node...")
	if err := node.Stop(); err != nil {
		log.Printf("Error stopping node: %v", err)
	}
}

func parseFlags() *NodeConfig {
	config := &NodeConfig{}

	flag.StringVar(&config.DataDir, "datadir", "./data", "Data directory for blockchain storage")
	flag.StringVar(&config.RPCHost, "rpc-host", "127.0.0.1", "RPC server host")
	flag.IntVar(&config.RPCPort, "rpc-port", 8545, "RPC server port")
	flag.IntVar(&config.P2PPort, "p2p-port", 30303, "P2P network port")
	flag.BoolVar(&config.Mining, "mine", false, "Enable mining")
	flag.StringVar(&config.WalletFile, "wallet", "", "Wallet file path")
	flag.BoolVar(&config.Genesis, "genesis", false, "Initialize with genesis block")

	// Parse bootstrap nodes
	var bootstrapNodes string
	flag.StringVar(&bootstrapNodes, "bootnodes", "", "Comma-separated list of bootstrap nodes")

	flag.Parse()

	// Parse bootstrap nodes
	if bootstrapNodes != "" {
		// Split by comma and add to config
		// config.BootstrapNodes = strings.Split(bootstrapNodes, ",")
	}

	return config
}

// NewNode creates a new blockchain node
func NewNode(config *NodeConfig) (*Node, error) {
	// Create or load wallet
	nodeWallet, err := setupWallet(config.WalletFile)
	if err != nil {
		return nil, fmt.Errorf("failed to setup wallet: %w", err)
	}

	// Create validator set
	validatorSet := validator.NewValidatorSet()

	// Add initial validators (for testing, add self as validator)
	val := validator.CreateValidator(
		nodeWallet.GetAddress(),
		nodeWallet.GetPublicKey(),
		big.NewInt(1000000), // 1M stake
	)
	validatorSet.AddValidator(val)

	// Create consensus engine with default config
	consensusConfig := pocconfig.DefaultConsensusConfig()
	consensusConfig.BlockTime = 10 * time.Second
	consensusConfig.MinStakeRequired = big.NewInt(1000)
	consensusEngine := consensus.NewEngine(consensusConfig)
	consensusEngine.SetValidatorSet(validatorSet)

	// Create blockchain
	blockchainConfig := storage.DefaultBlockchainConfig()
	blockchainConfig.DataDir = fmt.Sprintf("%s/blockchain", config.DataDir)

	blockchain, err := storage.NewBlockchain(consensusEngine, blockchainConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create blockchain: %w", err)
	}

	// Create state reader for transaction pool
	stateReader := NewBlockchainStateReader(blockchain)

	// Create transaction pool
	txPoolConfig := mempool.DefaultTxPoolConfig()
	txPool := mempool.NewTxPool(txPoolConfig, stateReader)

	// Create block producer
	producerConfig := mining.DefaultProducerConfig()
	producer := mining.NewBlockProducer(
		blockchain,
		txPool,
		consensusEngine,
		nodeWallet,
		producerConfig,
	)

	// Create network broadcaster
	nodeID := fmt.Sprintf("node_%s", nodeWallet.GetAddress()[:8])
	broadcasterConfig := network.DefaultBroadcasterConfig()
	broadcaster := network.NewBroadcaster(nodeID, broadcasterConfig)

	// Create RPC server
	rpcConfig := rpc.DefaultServerConfig()
	rpcConfig.Host = config.RPCHost
	rpcConfig.Port = config.RPCPort

	rpcServer := rpc.NewServer(
		blockchain,
		txPool,
		producer,
		nodeWallet,
		rpcConfig,
	)

	// Create peer listener
	p2pAddress := fmt.Sprintf(":%d", config.P2PPort)
	peerListener := network.NewPeerListener(p2pAddress, func(peer *network.PeerConnection) {
		// Handle new peer connection
		if err := broadcaster.AddPeer(peer); err != nil {
			log.Printf("Failed to add peer: %v", err)
		} else {
			log.Printf("New peer connected: %s", peer.ID)
		}
	})

	return &Node{
		config:       config,
		blockchain:   blockchain,
		consensus:    consensusEngine,
		txPool:       txPool,
		wallet:       nodeWallet,
		producer:     producer,
		broadcaster:  broadcaster,
		rpcServer:    rpcServer,
		peerListener: peerListener,
	}, nil
}

// Start starts the node
func (n *Node) Start() error {
	// Register message handlers
	n.registerMessageHandlers()

	// Start network components
	if err := n.broadcaster.Start(); err != nil {
		return fmt.Errorf("failed to start broadcaster: %w", err)
	}

	if err := n.peerListener.Start(); err != nil {
		return fmt.Errorf("failed to start peer listener: %w", err)
	}

	// Connect to bootstrap nodes
	for _, bootnode := range n.config.BootstrapNodes {
		go n.connectToPeer(bootnode)
	}

	// Start RPC server
	if err := n.rpcServer.Start(); err != nil {
		return fmt.Errorf("failed to start RPC server: %w", err)
	}

	// Start mining if enabled
	if n.config.Mining {
		if err := n.producer.Start(); err != nil {
			return fmt.Errorf("failed to start mining: %w", err)
		}

		// Subscribe to new blocks
		blockCh := make(chan *storage.Block, 10)
		n.producer.Subscribe(blockCh)

		// Handle new blocks
		go n.handleNewBlocks(blockCh)
	}

	return nil
}

// Stop stops the node
func (n *Node) Stop() error {
	// Stop mining
	if n.config.Mining {
		if err := n.producer.Stop(); err != nil {
			log.Printf("Failed to stop producer: %v", err)
		}
	}

	// Stop RPC server
	if err := n.rpcServer.Stop(); err != nil {
		log.Printf("Failed to stop RPC server: %v", err)
	}

	// Stop network components
	if err := n.peerListener.Stop(); err != nil {
		log.Printf("Failed to stop peer listener: %v", err)
	}

	if err := n.broadcaster.Stop(); err != nil {
		log.Printf("Failed to stop broadcaster: %v", err)
	}

	// Close blockchain
	if err := n.blockchain.Close(); err != nil {
		log.Printf("Failed to close blockchain: %v", err)
	}

	return nil
}

// registerMessageHandlers registers handlers for network messages
func (n *Node) registerMessageHandlers() {
	// Handle incoming transactions
	n.broadcaster.RegisterHandler(network.MessageTypeTransaction, func(msg *network.BroadcastMessage, sender *network.PeerConnection) error {
		var payload network.TransactionPayload
		if err := json.Unmarshal(msg.Payload, &payload); err != nil {
			return err
		}

		// Add to transaction pool
		if err := n.txPool.AddTransaction(payload.Transaction); err != nil {
			log.Printf("Failed to add transaction to pool: %v", err)
			return err
		}

		log.Printf("Received transaction %s from peer", payload.Transaction.Hash)
		return nil
	})

	// Handle incoming blocks
	n.broadcaster.RegisterHandler(network.MessageTypeBlock, func(msg *network.BroadcastMessage, sender *network.PeerConnection) error {
		var payload network.BlockPayload
		if err := json.Unmarshal(msg.Payload, &payload); err != nil {
			return err
		}

		// Validate and add block
		if err := n.consensus.ValidateBlock(payload.Block); err != nil {
			log.Printf("Invalid block received: %v", err)
			return err
		}

		if err := n.blockchain.AddBlock(payload.Block); err != nil {
			log.Printf("Failed to add block: %v", err)
			return err
		}

		log.Printf("Received block #%d from peer", payload.Block.Height)
		return nil
	})

	// Handle block requests
	n.broadcaster.RegisterHandler(network.MessageTypeBlockRequest, func(msg *network.BroadcastMessage, sender *network.PeerConnection) error {
		var payload network.BlockRequestPayload
		if err := json.Unmarshal(msg.Payload, &payload); err != nil {
			return err
		}

		// Get requested block
		block, err := n.blockchain.GetBlock(payload.Height)
		if err != nil {
			return err
		}

		// Convert and send block
		simpleBlock := &storage.Block{
			Height:       block.Header.Height,
			PreviousHash: block.Header.PreviousHash,
			Timestamp:    block.Header.Timestamp,
			Proposer:     block.Header.Proposer,
			StateRoot:    block.Header.StateRoot,
			TxRoot:       block.Header.TxRoot,
			Hash:         block.Hash,
		}

		// Send block to requester
		blockPayload, _ := json.Marshal(&network.BlockPayload{Block: simpleBlock})
		response := &network.BroadcastMessage{
			Type:      network.MessageTypeBlock,
			Payload:   blockPayload,
			Timestamp: time.Now(),
			Sender:    fmt.Sprintf("node_%s", n.wallet.GetAddress()[:8]),
			ID:        fmt.Sprintf("block_response_%d_%d", payload.Height, time.Now().UnixNano()),
		}

		return sender.SendMessage(response)
	})
}

// handleNewBlocks handles newly mined blocks
func (n *Node) handleNewBlocks(blockCh <-chan *storage.Block) {
	for block := range blockCh {
		// Broadcast new block to network
		if err := n.broadcaster.BroadcastBlock(block); err != nil {
			log.Printf("Failed to broadcast block: %v", err)
		}
	}
}

// connectToPeer connects to a peer node
func (n *Node) connectToPeer(address string) {
	peer, err := network.NewPeerConnection(address)
	if err != nil {
		log.Printf("Failed to connect to peer %s: %v", address, err)
		return
	}

	if err := n.broadcaster.AddPeer(peer); err != nil {
		log.Printf("Failed to add peer %s: %v", address, err)
		peer.Close()
		return
	}

	log.Printf("Connected to peer: %s", address)
}

// setupWallet creates or loads a wallet
func setupWallet(walletFile string) (*wallet.Wallet, error) {
	if walletFile != "" {
		// Try to load existing wallet
		if _, err := os.Stat(walletFile); err == nil {
			return wallet.LoadFromFile(walletFile)
		}
	}

	// Create new wallet
	w, err := wallet.NewWallet()
	if err != nil {
		return nil, err
	}

	// Save wallet if file specified
	if walletFile != "" {
		if err := w.SaveToFile(walletFile); err != nil {
			return nil, fmt.Errorf("failed to save wallet: %w", err)
		}
	}

	return w, nil
}

// BlockchainStateReader implements mempool.StateReader using blockchain
type BlockchainStateReader struct {
	blockchain *storage.Blockchain
}

func NewBlockchainStateReader(blockchain *storage.Blockchain) *BlockchainStateReader {
	return &BlockchainStateReader{
		blockchain: blockchain,
	}
}

func (r *BlockchainStateReader) GetBalance(address string) *big.Int {
	return r.blockchain.GetBalance(address)
}

func (r *BlockchainStateReader) GetNonce(address string) uint64 {
	return r.blockchain.GetNonce(address)
}