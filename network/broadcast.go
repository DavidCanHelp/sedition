package network

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/storage"
)

// MessageType represents the type of broadcast message
type MessageType string

const (
	MessageTypeTransaction MessageType = "transaction"
	MessageTypeBlock       MessageType = "block"
	MessageTypeBlockRequest MessageType = "block_request"
	MessageTypePeerRequest  MessageType = "peer_request"
	MessageTypeHeartbeat    MessageType = "heartbeat"
)

// BroadcastMessage represents a message to be broadcast
type BroadcastMessage struct {
	Type      MessageType     `json:"type"`
	Payload   json.RawMessage `json:"payload"`
	Timestamp time.Time       `json:"timestamp"`
	Sender    string          `json:"sender"`
	ID        string          `json:"id"`
}

// TransactionPayload represents transaction broadcast data
type TransactionPayload struct {
	Transaction *storage.Transaction `json:"transaction"`
}

// BlockPayload represents block broadcast data
type BlockPayload struct {
	Block *storage.Block `json:"block"`
}

// BlockRequestPayload represents a block request
type BlockRequestPayload struct {
	Height uint64 `json:"height"`
	Hash   string `json:"hash,omitempty"`
}

// Broadcaster handles broadcasting messages to network peers
type Broadcaster struct {
	mu sync.RWMutex

	// Node information
	nodeID string

	// Peer connections
	peers map[string]*PeerConnection

	// Message handling
	incomingMessages  chan *BroadcastMessage
	outgoingMessages  chan *BroadcastMessage
	messageHandlers   map[MessageType]MessageHandler
	processedMessages map[string]bool // Track processed messages to prevent loops

	// Configuration
	config BroadcasterConfig

	// State
	isRunning bool
	ctx       context.Context
	cancel    context.CancelFunc
}

// BroadcasterConfig holds broadcaster configuration
type BroadcasterConfig struct {
	MaxPeers          int
	MaxMessageSize    int
	HeartbeatInterval time.Duration
	MessageTTL        time.Duration
	RebroadcastDelay  time.Duration
}

// DefaultBroadcasterConfig returns default configuration
func DefaultBroadcasterConfig() BroadcasterConfig {
	return BroadcasterConfig{
		MaxPeers:          50,
		MaxMessageSize:    1024 * 1024, // 1MB
		HeartbeatInterval: 30 * time.Second,
		MessageTTL:        5 * time.Minute,
		RebroadcastDelay:  100 * time.Millisecond,
	}
}

// MessageHandler handles incoming messages of a specific type
type MessageHandler func(message *BroadcastMessage, sender *PeerConnection) error

// NewBroadcaster creates a new broadcaster
func NewBroadcaster(nodeID string, config BroadcasterConfig) *Broadcaster {
	return &Broadcaster{
		nodeID:            nodeID,
		peers:             make(map[string]*PeerConnection),
		incomingMessages:  make(chan *BroadcastMessage, 1000),
		outgoingMessages:  make(chan *BroadcastMessage, 1000),
		messageHandlers:   make(map[MessageType]MessageHandler),
		processedMessages: make(map[string]bool),
		config:            config,
	}
}

// Start starts the broadcaster
func (b *Broadcaster) Start() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.isRunning {
		return fmt.Errorf("broadcaster already running")
	}

	b.ctx, b.cancel = context.WithCancel(context.Background())
	b.isRunning = true

	// Start message processing loops
	go b.processIncomingMessages()
	go b.processOutgoingMessages()
	go b.heartbeatLoop()
	go b.cleanupLoop()

	return nil
}

// Stop stops the broadcaster
func (b *Broadcaster) Stop() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.isRunning {
		return fmt.Errorf("broadcaster not running")
	}

	b.cancel()
	b.isRunning = false

	// Close all peer connections
	for _, peer := range b.peers {
		peer.Close()
	}

	return nil
}

// BroadcastTransaction broadcasts a transaction to all peers
func (b *Broadcaster) BroadcastTransaction(tx *storage.Transaction) error {
	payload, err := json.Marshal(&TransactionPayload{
		Transaction: tx,
	})
	if err != nil {
		return fmt.Errorf("failed to marshal transaction: %w", err)
	}

	message := &BroadcastMessage{
		Type:      MessageTypeTransaction,
		Payload:   payload,
		Timestamp: time.Now(),
		Sender:    b.nodeID,
		ID:        fmt.Sprintf("%s_%s_%d", MessageTypeTransaction, tx.Hash, time.Now().UnixNano()),
	}

	return b.broadcast(message)
}

// BroadcastBlock broadcasts a block to all peers
func (b *Broadcaster) BroadcastBlock(block *storage.Block) error {
	payload, err := json.Marshal(&BlockPayload{
		Block: block,
	})
	if err != nil {
		return fmt.Errorf("failed to marshal block: %w", err)
	}

	message := &BroadcastMessage{
		Type:      MessageTypeBlock,
		Payload:   payload,
		Timestamp: time.Now(),
		Sender:    b.nodeID,
		ID:        fmt.Sprintf("%s_%s_%d", MessageTypeBlock, block.Hash, time.Now().UnixNano()),
	}

	return b.broadcast(message)
}

// RegisterHandler registers a message handler
func (b *Broadcaster) RegisterHandler(messageType MessageType, handler MessageHandler) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.messageHandlers[messageType] = handler
}

// AddPeer adds a new peer connection
func (b *Broadcaster) AddPeer(peer *PeerConnection) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if len(b.peers) >= b.config.MaxPeers {
		return fmt.Errorf("maximum peers reached")
	}

	b.peers[peer.ID] = peer

	// Start listening for messages from this peer
	go b.handlePeerMessages(peer)

	return nil
}

// RemovePeer removes a peer connection
func (b *Broadcaster) RemovePeer(peerID string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if peer, exists := b.peers[peerID]; exists {
		peer.Close()
		delete(b.peers, peerID)
	}
}

// GetPeers returns current peer connections
func (b *Broadcaster) GetPeers() []*PeerConnection {
	b.mu.RLock()
	defer b.mu.RUnlock()

	peers := make([]*PeerConnection, 0, len(b.peers))
	for _, peer := range b.peers {
		peers = append(peers, peer)
	}
	return peers
}

// broadcast sends a message to all peers
func (b *Broadcaster) broadcast(message *BroadcastMessage) error {
	b.mu.Lock()
	// Mark as processed to prevent rebroadcasting our own messages
	b.processedMessages[message.ID] = true
	b.mu.Unlock()

	select {
	case b.outgoingMessages <- message:
		return nil
	case <-time.After(time.Second):
		return fmt.Errorf("broadcast timeout")
	}
}

// processIncomingMessages processes incoming messages
func (b *Broadcaster) processIncomingMessages() {
	for {
		select {
		case <-b.ctx.Done():
			return
		case message := <-b.incomingMessages:
			b.handleIncomingMessage(message)
		}
	}
}

// processOutgoingMessages sends messages to peers
func (b *Broadcaster) processOutgoingMessages() {
	for {
		select {
		case <-b.ctx.Done():
			return
		case message := <-b.outgoingMessages:
			b.sendToPeers(message)
		}
	}
}

// handleIncomingMessage handles an incoming message
func (b *Broadcaster) handleIncomingMessage(message *BroadcastMessage) {
	// Check if already processed
	b.mu.RLock()
	if b.processedMessages[message.ID] {
		b.mu.RUnlock()
		return
	}
	b.mu.RUnlock()

	// Mark as processed
	b.mu.Lock()
	b.processedMessages[message.ID] = true
	b.mu.Unlock()

	// Find handler
	b.mu.RLock()
	handler, exists := b.messageHandlers[message.Type]
	b.mu.RUnlock()

	if !exists {
		fmt.Printf("No handler for message type: %s\n", message.Type)
		return
	}

	// Get sender peer
	b.mu.RLock()
	var sender *PeerConnection
	for _, peer := range b.peers {
		if peer.ID == message.Sender {
			sender = peer
			break
		}
	}
	b.mu.RUnlock()

	// Handle the message
	if err := handler(message, sender); err != nil {
		fmt.Printf("Failed to handle message: %v\n", err)
		return
	}

	// Rebroadcast to other peers (gossip protocol)
	time.Sleep(b.config.RebroadcastDelay)
	b.sendToPeers(message)
}

// sendToPeers sends a message to all connected peers
func (b *Broadcaster) sendToPeers(message *BroadcastMessage) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for _, peer := range b.peers {
		// Don't send back to the original sender
		if peer.ID == message.Sender {
			continue
		}

		// Send message to peer
		if err := peer.SendMessage(message); err != nil {
			fmt.Printf("Failed to send message to peer %s: %v\n", peer.ID, err)
		}
	}
}

// handlePeerMessages handles messages from a specific peer
func (b *Broadcaster) handlePeerMessages(peer *PeerConnection) {
	for {
		message, err := peer.ReceiveMessage()
		if err != nil {
			fmt.Printf("Failed to receive message from peer %s: %v\n", peer.ID, err)
			b.RemovePeer(peer.ID)
			return
		}

		select {
		case b.incomingMessages <- message:
		case <-b.ctx.Done():
			return
		}
	}
}

// heartbeatLoop sends periodic heartbeats
func (b *Broadcaster) heartbeatLoop() {
	ticker := time.NewTicker(b.config.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-b.ctx.Done():
			return
		case <-ticker.C:
			b.sendHeartbeat()
		}
	}
}

// sendHeartbeat sends a heartbeat message
func (b *Broadcaster) sendHeartbeat() {
	message := &BroadcastMessage{
		Type:      MessageTypeHeartbeat,
		Payload:   json.RawMessage("{}"),
		Timestamp: time.Now(),
		Sender:    b.nodeID,
		ID:        fmt.Sprintf("heartbeat_%s_%d", b.nodeID, time.Now().UnixNano()),
	}

	b.broadcast(message)
}

// cleanupLoop periodically cleans up old processed messages
func (b *Broadcaster) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-b.ctx.Done():
			return
		case <-ticker.C:
			b.cleanupProcessedMessages()
		}
	}
}

// cleanupProcessedMessages removes old processed message IDs
func (b *Broadcaster) cleanupProcessedMessages() {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Keep only recent messages to prevent memory leak
	// In production, use a more sophisticated approach
	if len(b.processedMessages) > 10000 {
		b.processedMessages = make(map[string]bool)
	}
}

// GetStats returns broadcaster statistics
func (b *Broadcaster) GetStats() map[string]interface{} {
	b.mu.RLock()
	defer b.mu.RUnlock()

	return map[string]interface{}{
		"node_id":           b.nodeID,
		"peer_count":        len(b.peers),
		"is_running":        b.isRunning,
		"processed_messages": len(b.processedMessages),
		"incoming_queue":    len(b.incomingMessages),
		"outgoing_queue":    len(b.outgoingMessages),
	}
}