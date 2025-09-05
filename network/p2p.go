// Package network implements the P2P networking layer for PoC consensus
package network

import (
	"bufio"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/crypto"
)

// MessageType defines the type of P2P message
type MessageType uint8

const (
	MessageTypeHandshake MessageType = iota
	MessageTypePing
	MessageTypePong
	MessageTypeBlockProposal
	MessageTypeBlockVote
	MessageTypeBlockRequest
	MessageTypeBlockResponse
	MessageTypeCommitAnnounce
	MessageTypePeerRequest
	MessageTypePeerResponse
	MessageTypeStateSync
	MessageTypeConsensusMessage
)

// Message represents a P2P network message
type Message struct {
	Type      MessageType
	Timestamp time.Time
	Sender    string
	Payload   []byte
	Signature []byte
}

// Peer represents a network peer
type Peer struct {
	ID            string
	Address       string
	PublicKey     ed25519.PublicKey
	conn          net.Conn
	inbound       bool
	lastSeen      time.Time
	latency       time.Duration
	version       string
	capabilities  []string
	reputation    float64
	messageCount  int64
	bytesReceived int64
	bytesSent     int64
}

// P2PNode represents a node in the P2P network
type P2PNode struct {
	mu sync.RWMutex

	// Identity
	id         string
	signer     *crypto.Signer
	address    string
	listenAddr string

	// Network state
	peers       map[string]*Peer
	bannedPeers map[string]time.Time
	maxPeers    int
	listener    net.Listener

	// Message handling
	messageHandlers map[MessageType]MessageHandler
	messageQueue    chan *MessageEnvelope

	// Gossip protocol
	gossipPool map[string]*GossipMessage
	gossipTTL  int

	// Discovery
	bootstrapNodes []string
	dht            *DHT

	// Metrics
	totalMessages int64
	totalBytes    int64
	startTime     time.Time

	// Control
	ctx    context.Context
	cancel context.CancelFunc
}

// MessageHandler processes received messages
type MessageHandler func(peer *Peer, msg *Message) error

// MessageEnvelope wraps a message with routing info
type MessageEnvelope struct {
	From    *Peer
	To      []*Peer
	Message *Message
}

// GossipMessage represents a message being gossiped
type GossipMessage struct {
	Message   *Message
	FirstSeen time.Time
	HopCount  int
	SeenPeers map[string]bool
}

// NewP2PNode creates a new P2P node
func NewP2PNode(listenAddr string, signer *crypto.Signer, bootstrapNodes []string) (*P2PNode, error) {
	ctx, cancel := context.WithCancel(context.Background())

	node := &P2PNode{
		id:              signer.GetAddress(),
		signer:          signer,
		listenAddr:      listenAddr,
		peers:           make(map[string]*Peer),
		bannedPeers:     make(map[string]time.Time),
		maxPeers:        50,
		messageHandlers: make(map[MessageType]MessageHandler),
		messageQueue:    make(chan *MessageEnvelope, 1000),
		gossipPool:      make(map[string]*GossipMessage),
		gossipTTL:       6,
		bootstrapNodes:  bootstrapNodes,
		startTime:       time.Now(),
		ctx:             ctx,
		cancel:          cancel,
	}

	// Register default handlers
	node.registerDefaultHandlers()

	// Initialize DHT
	node.dht = NewDHT(node.id)

	return node, nil
}

// Start begins listening and connecting to peers
func (n *P2PNode) Start() error {
	// Start listener
	listener, err := net.Listen("tcp", n.listenAddr)
	if err != nil {
		return fmt.Errorf("failed to start listener: %w", err)
	}
	n.listener = listener
	n.address = listener.Addr().String()

	// Start goroutines
	go n.acceptLoop()
	go n.messageLoop()
	go n.gossipLoop()
	go n.discoveryLoop()
	go n.maintenanceLoop()

	// Bootstrap network
	if err := n.bootstrap(); err != nil {
		return fmt.Errorf("bootstrap failed: %w", err)
	}

	return nil
}

// Stop gracefully shuts down the node
func (n *P2PNode) Stop() {
	n.cancel()

	n.mu.Lock()
	defer n.mu.Unlock()

	// Close all peer connections
	for _, peer := range n.peers {
		peer.conn.Close()
	}

	// Close listener
	if n.listener != nil {
		n.listener.Close()
	}
}

// acceptLoop accepts incoming connections
func (n *P2PNode) acceptLoop() {
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			conn, err := n.listener.Accept()
			if err != nil {
				continue
			}

			go n.handleIncomingConnection(conn)
		}
	}
}

// handleIncomingConnection processes a new incoming connection
func (n *P2PNode) handleIncomingConnection(conn net.Conn) {
	// Set timeout for handshake
	conn.SetDeadline(time.Now().Add(10 * time.Second))

	// Perform handshake
	peer, err := n.performHandshake(conn, true)
	if err != nil {
		conn.Close()
		return
	}

	// Clear deadline
	conn.SetDeadline(time.Time{})

	// Add peer
	n.addPeer(peer)

	// Start handling messages
	go n.handlePeerMessages(peer)
}

// ConnectToPeer connects to a remote peer
func (n *P2PNode) ConnectToPeer(address string) error {
	conn, err := net.DialTimeout("tcp", address, 10*time.Second)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	// Perform handshake
	peer, err := n.performHandshake(conn, false)
	if err != nil {
		conn.Close()
		return fmt.Errorf("handshake failed: %w", err)
	}

	// Add peer
	n.addPeer(peer)

	// Start handling messages
	go n.handlePeerMessages(peer)

	return nil
}

// performHandshake performs the protocol handshake
func (n *P2PNode) performHandshake(conn net.Conn, inbound bool) (*Peer, error) {
	handshake := &HandshakeMessage{
		Version:      "1.0.0",
		NodeID:       n.id,
		PublicKey:    n.signer.GetPublicKey(),
		Timestamp:    time.Now(),
		Capabilities: []string{"poc", "gossip", "sync"},
	}

	// Sign handshake
	data, err := json.Marshal(handshake)
	if err != nil {
		return nil, err
	}

	sig, err := n.signer.Sign(data)
	if err != nil {
		return nil, err
	}
	handshake.Signature = sig

	// Send our handshake
	if err := n.sendHandshake(conn, handshake); err != nil {
		return nil, err
	}

	// Receive peer handshake
	peerHandshake, err := n.receiveHandshake(conn)
	if err != nil {
		return nil, err
	}

	// Verify signature
	peerData, _ := json.Marshal(HandshakeMessage{
		Version:      peerHandshake.Version,
		NodeID:       peerHandshake.NodeID,
		PublicKey:    peerHandshake.PublicKey,
		Timestamp:    peerHandshake.Timestamp,
		Capabilities: peerHandshake.Capabilities,
	})

	if !ed25519.Verify(peerHandshake.PublicKey, peerData, peerHandshake.Signature) {
		return nil, errors.New("invalid handshake signature")
	}

	// Create peer
	peer := &Peer{
		ID:           peerHandshake.NodeID,
		Address:      conn.RemoteAddr().String(),
		PublicKey:    peerHandshake.PublicKey,
		conn:         conn,
		inbound:      inbound,
		lastSeen:     time.Now(),
		version:      peerHandshake.Version,
		capabilities: peerHandshake.Capabilities,
		reputation:   0.5,
	}

	return peer, nil
}

// HandshakeMessage is sent during connection establishment
type HandshakeMessage struct {
	Version      string
	NodeID       string
	PublicKey    ed25519.PublicKey
	Timestamp    time.Time
	Capabilities []string
	Signature    []byte
}

// sendHandshake sends a handshake message
func (n *P2PNode) sendHandshake(conn net.Conn, h *HandshakeMessage) error {
	data, err := json.Marshal(h)
	if err != nil {
		return err
	}

	// Write length prefix
	length := uint32(len(data))
	if err := binary.Write(conn, binary.BigEndian, length); err != nil {
		return err
	}

	// Write data
	_, err = conn.Write(data)
	return err
}

// receiveHandshake receives a handshake message
func (n *P2PNode) receiveHandshake(conn net.Conn) (*HandshakeMessage, error) {
	// Read length prefix
	var length uint32
	if err := binary.Read(conn, binary.BigEndian, &length); err != nil {
		return nil, err
	}

	if length > 1024*1024 { // Max 1MB
		return nil, errors.New("handshake too large")
	}

	// Read data
	data := make([]byte, length)
	if _, err := io.ReadFull(conn, data); err != nil {
		return nil, err
	}

	// Unmarshal
	var h HandshakeMessage
	if err := json.Unmarshal(data, &h); err != nil {
		return nil, err
	}

	return &h, nil
}

// addPeer adds a peer to the node
func (n *P2PNode) addPeer(peer *Peer) {
	n.mu.Lock()
	defer n.mu.Unlock()

	// Check if banned
	if banTime, banned := n.bannedPeers[peer.ID]; banned {
		if time.Now().Before(banTime) {
			peer.conn.Close()
			return
		}
		delete(n.bannedPeers, peer.ID)
	}

	// Check max peers
	if len(n.peers) >= n.maxPeers {
		// Disconnect lowest reputation peer
		var lowestPeer *Peer
		lowestRep := 1.0
		for _, p := range n.peers {
			if p.reputation < lowestRep {
				lowestRep = p.reputation
				lowestPeer = p
			}
		}
		if lowestPeer != nil && lowestPeer.reputation < peer.reputation {
			n.disconnectPeer(lowestPeer.ID)
		} else {
			peer.conn.Close()
			return
		}
	}

	n.peers[peer.ID] = peer
}

// disconnectPeer removes and disconnects a peer
func (n *P2PNode) disconnectPeer(peerID string) {
	peer, exists := n.peers[peerID]
	if !exists {
		return
	}

	peer.conn.Close()
	delete(n.peers, peerID)
}

// handlePeerMessages reads and processes messages from a peer
func (n *P2PNode) handlePeerMessages(peer *Peer) {
	reader := bufio.NewReader(peer.conn)

	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			// Set read deadline
			peer.conn.SetReadDeadline(time.Now().Add(30 * time.Second))

			// Read message
			msg, err := n.readMessage(reader)
			if err != nil {
				n.disconnectPeer(peer.ID)
				return
			}

			// Update peer stats
			peer.lastSeen = time.Now()
			peer.messageCount++
			peer.bytesReceived += int64(len(msg.Payload))

			// Verify signature
			if !n.verifyMessage(peer, msg) {
				peer.reputation -= 0.1
				if peer.reputation < 0 {
					n.banPeer(peer.ID, 1*time.Hour)
					n.disconnectPeer(peer.ID)
					return
				}
				continue
			}

			// Process message
			envelope := &MessageEnvelope{
				From:    peer,
				Message: msg,
			}

			select {
			case n.messageQueue <- envelope:
			default:
				// Queue full, drop message
			}
		}
	}
}

// readMessage reads a message from the connection
func (n *P2PNode) readMessage(reader *bufio.Reader) (*Message, error) {
	// Read length prefix
	lengthBytes := make([]byte, 4)
	if _, err := io.ReadFull(reader, lengthBytes); err != nil {
		return nil, err
	}

	length := binary.BigEndian.Uint32(lengthBytes)
	if length > 10*1024*1024 { // Max 10MB
		return nil, errors.New("message too large")
	}

	// Read message data
	data := make([]byte, length)
	if _, err := io.ReadFull(reader, data); err != nil {
		return nil, err
	}

	// Unmarshal message
	var msg Message
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, err
	}

	return &msg, nil
}

// verifyMessage verifies a message signature
func (n *P2PNode) verifyMessage(peer *Peer, msg *Message) bool {
	// Serialize message without signature
	msgCopy := *msg
	msgCopy.Signature = nil
	data, err := json.Marshal(msgCopy)
	if err != nil {
		return false
	}

	return ed25519.Verify(peer.PublicKey, data, msg.Signature)
}

// BroadcastBlock broadcasts a block proposal to all peers
// Block represents a consensus block
type Block struct {
	Height       uint64
	PreviousHash []byte
	Timestamp    time.Time
	Data         []byte
	Proposer     string
	Signatures   map[string][]byte
}

func (n *P2PNode) BroadcastBlock(block *Block) error {
	data, err := json.Marshal(block)
	if err != nil {
		return err
	}

	msg := &Message{
		Type:      MessageTypeBlockProposal,
		Timestamp: time.Now(),
		Sender:    n.id,
		Payload:   data,
	}

	return n.broadcast(msg)
}

// BroadcastVote broadcasts a block vote
func (n *P2PNode) BroadcastVote(blockHash []byte, approve bool) error {
	vote := &BlockVote{
		BlockHash: blockHash,
		Approve:   approve,
		Timestamp: time.Now(),
		Voter:     n.id,
	}

	data, err := json.Marshal(vote)
	if err != nil {
		return err
	}

	msg := &Message{
		Type:      MessageTypeBlockVote,
		Timestamp: time.Now(),
		Sender:    n.id,
		Payload:   data,
	}

	return n.broadcast(msg)
}

// BlockVote represents a vote on a block
type BlockVote struct {
	BlockHash []byte
	Approve   bool
	Timestamp time.Time
	Voter     string
	Signature []byte
}

// broadcast sends a message to all connected peers
func (n *P2PNode) broadcast(msg *Message) error {
	// Sign message
	msgCopy := *msg
	msgCopy.Signature = nil
	data, err := json.Marshal(msgCopy)
	if err != nil {
		return err
	}

	sig, err := n.signer.Sign(data)
	if err != nil {
		return err
	}
	msg.Signature = sig

	// Send to all peers
	n.mu.RLock()
	peers := make([]*Peer, 0, len(n.peers))
	for _, peer := range n.peers {
		peers = append(peers, peer)
	}
	n.mu.RUnlock()

	for _, peer := range peers {
		go n.sendMessage(peer, msg)
	}

	return nil
}

// sendMessage sends a message to a specific peer
func (n *P2PNode) sendMessage(peer *Peer, msg *Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	// Write length prefix
	lengthBytes := make([]byte, 4)
	binary.BigEndian.PutUint32(lengthBytes, uint32(len(data)))

	// Set write deadline
	peer.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

	// Write length and data
	if _, err := peer.conn.Write(lengthBytes); err != nil {
		return err
	}
	if _, err := peer.conn.Write(data); err != nil {
		return err
	}

	// Update stats
	peer.bytesSent += int64(len(data) + 4)

	return nil
}

// Gossip propagates a message through the network
func (n *P2PNode) Gossip(msg *Message) {
	msgID := n.getMessageID(msg)

	n.mu.Lock()
	// Check if we've seen this message
	if _, exists := n.gossipPool[msgID]; exists {
		n.mu.Unlock()
		return
	}

	// Add to gossip pool
	n.gossipPool[msgID] = &GossipMessage{
		Message:   msg,
		FirstSeen: time.Now(),
		HopCount:  0,
		SeenPeers: make(map[string]bool),
	}
	n.mu.Unlock()

	// Propagate to subset of peers
	n.propagateGossip(msgID)
}

// propagateGossip sends a gossip message to selected peers
func (n *P2PNode) propagateGossip(msgID string) {
	n.mu.RLock()
	gossip, exists := n.gossipPool[msgID]
	if !exists {
		n.mu.RUnlock()
		return
	}

	// Select random subset of peers
	peers := make([]*Peer, 0, len(n.peers))
	for _, peer := range n.peers {
		if !gossip.SeenPeers[peer.ID] {
			peers = append(peers, peer)
		}
	}
	n.mu.RUnlock()

	// Randomly select sqrt(n) peers
	numToSend := int(math.Sqrt(float64(len(peers))))
	if numToSend < 3 {
		numToSend = min(3, len(peers))
	}

	// Shuffle and select
	rand.Shuffle(len(peers), func(i, j int) {
		peers[i], peers[j] = peers[j], peers[i]
	})

	for i := 0; i < numToSend && i < len(peers); i++ {
		go n.sendMessage(peers[i], gossip.Message)

		n.mu.Lock()
		gossip.SeenPeers[peers[i].ID] = true
		n.mu.Unlock()
	}
}

// getMessageID generates a unique ID for a message
func (n *P2PNode) getMessageID(msg *Message) string {
	data, _ := json.Marshal(msg)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// messageLoop processes incoming messages
func (n *P2PNode) messageLoop() {
	for {
		select {
		case <-n.ctx.Done():
			return
		case envelope := <-n.messageQueue:
			n.handleMessage(envelope)
		}
	}
}

// handleMessage routes a message to the appropriate handler
func (n *P2PNode) handleMessage(envelope *MessageEnvelope) {
	handler, exists := n.messageHandlers[envelope.Message.Type]
	if !exists {
		return
	}

	if err := handler(envelope.From, envelope.Message); err != nil {
		// Decrease peer reputation on error
		envelope.From.reputation -= 0.01
	} else {
		// Increase reputation on success
		envelope.From.reputation += 0.001
		if envelope.From.reputation > 1.0 {
			envelope.From.reputation = 1.0
		}
	}

	// Update metrics
	n.totalMessages++
	n.totalBytes += int64(len(envelope.Message.Payload))
}

// registerDefaultHandlers registers the default message handlers
func (n *P2PNode) registerDefaultHandlers() {
	n.messageHandlers[MessageTypePing] = n.handlePing
	n.messageHandlers[MessageTypePong] = n.handlePong
	n.messageHandlers[MessageTypePeerRequest] = n.handlePeerRequest
	n.messageHandlers[MessageTypePeerResponse] = n.handlePeerResponse
}

// handlePing handles ping messages
func (n *P2PNode) handlePing(peer *Peer, msg *Message) error {
	// Send pong
	pong := &Message{
		Type:      MessageTypePong,
		Timestamp: time.Now(),
		Sender:    n.id,
		Payload:   msg.Payload,
	}

	return n.sendMessage(peer, pong)
}

// handlePong handles pong messages
func (n *P2PNode) handlePong(peer *Peer, msg *Message) error {
	// Calculate latency
	var pingTime time.Time
	if err := json.Unmarshal(msg.Payload, &pingTime); err == nil {
		peer.latency = time.Since(pingTime)
	}
	return nil
}

// handlePeerRequest handles peer discovery requests
func (n *P2PNode) handlePeerRequest(peer *Peer, msg *Message) error {
	n.mu.RLock()
	peers := make([]string, 0, len(n.peers))
	for _, p := range n.peers {
		if p.ID != peer.ID {
			peers = append(peers, p.Address)
		}
	}
	n.mu.RUnlock()

	// Send random subset
	if len(peers) > 10 {
		rand.Shuffle(len(peers), func(i, j int) {
			peers[i], peers[j] = peers[j], peers[i]
		})
		peers = peers[:10]
	}

	data, err := json.Marshal(peers)
	if err != nil {
		return err
	}

	response := &Message{
		Type:      MessageTypePeerResponse,
		Timestamp: time.Now(),
		Sender:    n.id,
		Payload:   data,
	}

	return n.sendMessage(peer, response)
}

// handlePeerResponse handles peer discovery responses
func (n *P2PNode) handlePeerResponse(peer *Peer, msg *Message) error {
	var peers []string
	if err := json.Unmarshal(msg.Payload, &peers); err != nil {
		return err
	}

	// Try to connect to new peers
	for _, addr := range peers {
		go n.ConnectToPeer(addr)
	}

	return nil
}

// gossipLoop periodically cleans up old gossip messages
func (n *P2PNode) gossipLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.cleanupGossip()
		}
	}
}

// cleanupGossip removes old messages from gossip pool
func (n *P2PNode) cleanupGossip() {
	n.mu.Lock()
	defer n.mu.Unlock()

	cutoff := time.Now().Add(-5 * time.Minute)
	for id, gossip := range n.gossipPool {
		if gossip.FirstSeen.Before(cutoff) {
			delete(n.gossipPool, id)
		}
	}
}

// discoveryLoop periodically discovers new peers
func (n *P2PNode) discoveryLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.discoverPeers()
		}
	}
}

// discoverPeers requests peer lists from connected peers
func (n *P2PNode) discoverPeers() {
	n.mu.RLock()
	if len(n.peers) >= n.maxPeers {
		n.mu.RUnlock()
		return
	}

	peers := make([]*Peer, 0, len(n.peers))
	for _, peer := range n.peers {
		peers = append(peers, peer)
	}
	n.mu.RUnlock()

	// Request peers from random selection
	numToAsk := min(3, len(peers))
	rand.Shuffle(len(peers), func(i, j int) {
		peers[i], peers[j] = peers[j], peers[i]
	})

	for i := 0; i < numToAsk; i++ {
		msg := &Message{
			Type:      MessageTypePeerRequest,
			Timestamp: time.Now(),
			Sender:    n.id,
		}
		n.sendMessage(peers[i], msg)
	}
}

// maintenanceLoop performs periodic maintenance
func (n *P2PNode) maintenanceLoop() {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.performMaintenance()
		}
	}
}

// performMaintenance cleans up connections and banned peers
func (n *P2PNode) performMaintenance() {
	n.mu.Lock()
	defer n.mu.Unlock()

	// Remove dead peers
	cutoff := time.Now().Add(-2 * time.Minute)
	for id, peer := range n.peers {
		if peer.lastSeen.Before(cutoff) {
			peer.conn.Close()
			delete(n.peers, id)
		}
	}

	// Clean up expired bans
	now := time.Now()
	for id, banTime := range n.bannedPeers {
		if now.After(banTime) {
			delete(n.bannedPeers, id)
		}
	}
}

// banPeer bans a peer for a duration
func (n *P2PNode) banPeer(peerID string, duration time.Duration) {
	n.mu.Lock()
	defer n.mu.Unlock()

	n.bannedPeers[peerID] = time.Now().Add(duration)
	n.disconnectPeer(peerID)
}

// bootstrap connects to bootstrap nodes
func (n *P2PNode) bootstrap() error {
	if len(n.bootstrapNodes) == 0 {
		return nil
	}

	for _, addr := range n.bootstrapNodes {
		if err := n.ConnectToPeer(addr); err != nil {
			continue
		}
	}

	return nil
}

// GetPeers returns the list of connected peers
func (n *P2PNode) GetPeers() []*Peer {
	n.mu.RLock()
	defer n.mu.RUnlock()

	peers := make([]*Peer, 0, len(n.peers))
	for _, peer := range n.peers {
		peers = append(peers, peer)
	}

	return peers
}

// GetMetrics returns network metrics
func (n *P2PNode) GetMetrics() *NetworkMetrics {
	n.mu.RLock()
	defer n.mu.RUnlock()

	return &NetworkMetrics{
		PeerCount:      len(n.peers),
		TotalMessages:  n.totalMessages,
		TotalBytes:     n.totalBytes,
		Uptime:         time.Since(n.startTime),
		BannedPeers:    len(n.bannedPeers),
		GossipPoolSize: len(n.gossipPool),
	}
}

// NetworkMetrics contains network statistics
type NetworkMetrics struct {
	PeerCount      int
	TotalMessages  int64
	TotalBytes     int64
	Uptime         time.Duration
	BannedPeers    int
	GossipPoolSize int
}

// RegisterHandler registers a message handler
func (n *P2PNode) RegisterHandler(msgType MessageType, handler MessageHandler) {
	n.messageHandlers[msgType] = handler
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
