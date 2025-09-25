// Package network implements the P2P networking layer
package network

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/consensus"
	"github.com/davidcanhelp/sedition/errors"
)

// Node represents a P2P network node
type Node struct {
	mu           sync.RWMutex
	id           string
	privKey      ed25519.PrivateKey
	pubKey       ed25519.PublicKey
	address      string
	listener     net.Listener
	peers        map[string]*NodePeer
	engine       *consensus.Engine
	config       *config.NetworkConfig
	messageQueue chan *Message
	ctx          context.Context
	cancel       context.CancelFunc

	// Metrics
	messagesReceived uint64
	messagesSent     uint64
	bytesReceived    uint64
	bytesSent        uint64
}

// NodePeer represents a connected peer in the node context
type NodePeer struct {
	ID            string
	Address       string
	PublicKey     ed25519.PublicKey
	Connection    net.Conn
	LastSeen      time.Time
	MessageCount  uint64
	IsValidator   bool
	Reputation    float64
}

// NewNode creates a new P2P node
func NewNode(address string, engine *consensus.Engine, cfg *config.NetworkConfig) (*Node, error) {
	if cfg == nil {
		cfg = config.DefaultNetworkConfig()
	}

	// Generate node keypair
	pubKey, privKey, err := ed25519.GenerateKey(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate keys: %w", err)
	}

	// Generate node ID from public key
	nodeID := hex.EncodeToString(pubKey[:8])

	ctx, cancel := context.WithCancel(context.Background())

	node := &Node{
		id:           nodeID,
		privKey:      privKey,
		pubKey:       pubKey,
		address:      address,
		peers:        make(map[string]*NodePeer),
		engine:       engine,
		config:       cfg,
		messageQueue: make(chan *Message, 1000),
		ctx:          ctx,
		cancel:       cancel,
	}

	return node, nil
}

// Start initializes and starts the P2P node
func (n *Node) Start() error {
	// Start listening for incoming connections
	listener, err := net.Listen("tcp", n.address)
	if err != nil {
		return fmt.Errorf("failed to start listener: %w", err)
	}
	n.listener = listener

	// Start background workers
	go n.acceptConnections()
	go n.processMessages()
	go n.maintainPeers()
	go n.gossipLoop()

	return nil
}

// Stop gracefully shuts down the node
func (n *Node) Stop() error {
	n.cancel()

	if n.listener != nil {
		n.listener.Close()
	}

	// Close all peer connections
	n.mu.Lock()
	for _, peer := range n.peers {
		peer.Connection.Close()
	}
	n.mu.Unlock()

	close(n.messageQueue)
	return nil
}

// Connect establishes a connection to a peer
func (n *Node) Connect(peerAddress string) error {
	conn, err := net.DialTimeout("tcp", peerAddress, 10*time.Second)
	if err != nil {
		return fmt.Errorf("failed to connect to peer: %w", err)
	}

	// Perform handshake
	if err := n.performHandshake(conn, true); err != nil {
		conn.Close()
		return fmt.Errorf("handshake failed: %w", err)
	}

	return nil
}

// BroadcastMessage sends a message to all connected peers
func (n *Node) BroadcastMessage(msg *Message) error {
	// Sign the message
	msg.Sender = n.id
	msg.Timestamp = time.Now()

	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	msg.Signature = ed25519.Sign(n.privKey, data)

	n.mu.RLock()
	peers := make([]*NodePeer, 0, len(n.peers))
	for _, peer := range n.peers {
		peers = append(peers, peer)
	}
	n.mu.RUnlock()

	// Send to all peers
	for _, peer := range peers {
		go n.sendToPeer(peer, msg)
	}

	n.messagesSent++
	return nil
}

// SendMessage sends a message to a specific peer
func (n *Node) SendMessage(peerID string, msg *Message) error {
	n.mu.RLock()
	peer, exists := n.peers[peerID]
	n.mu.RUnlock()

	if !exists {
		return errors.NewConsensusError(
			errors.ErrPeerNotFound,
			"peer not found",
		).WithDetails("peerID", peerID)
	}

	return n.sendToPeer(peer, msg)
}

// acceptConnections handles incoming peer connections
func (n *Node) acceptConnections() {
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			conn, err := n.listener.Accept()
			if err != nil {
				continue
			}

			go n.handleConnection(conn)
		}
	}
}

// handleConnection processes an incoming connection
func (n *Node) handleConnection(conn net.Conn) {
	defer conn.Close()

	// Perform handshake
	if err := n.performHandshake(conn, false); err != nil {
		return
	}

	// Read messages from peer
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			msg, err := n.readMessage(conn)
			if err != nil {
				n.removePeer(conn)
				return
			}

			// Queue message for processing
			select {
			case n.messageQueue <- msg:
				n.messagesReceived++
			default:
				// Queue full, drop message
			}
		}
	}
}

// processMessages handles incoming messages
func (n *Node) processMessages() {
	for {
		select {
		case <-n.ctx.Done():
			return
		case msg := <-n.messageQueue:
			if msg == nil {
				continue
			}
			n.handleMessage(msg)
		}
	}
}

// handleMessage processes a specific message type
func (n *Node) handleMessage(msg *Message) {
	switch msg.Type {
	case LegacyMessageTypePing:
		n.handlePing(msg)
	case LegacyMessageTypePong:
		n.handlePong(msg)
	case LegacyMessageTypeBlockProposal:
		n.handleBlockProposal(msg)
	case LegacyMessageTypeBlockVote:
		n.handleBlockVote(msg)
	case LegacyMessageTypeConsensusMessage:
		n.handleConsensusMessage(msg)
	case LegacyMessageTypePeerRequest:
		n.handlePeerRequest(msg)
	case LegacyMessageTypePeerResponse:
		n.handlePeerResponse(msg)
	default:
		// Unknown message type
	}
}

// maintainPeers periodically checks peer health and manages connections
func (n *Node) maintainPeers() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.checkPeerHealth()
			n.discoverNewPeers()
		}
	}
}

// checkPeerHealth removes inactive peers
func (n *Node) checkPeerHealth() {
	n.mu.Lock()
	defer n.mu.Unlock()

	cutoff := time.Now().Add(-n.config.PeerTimeout)
	for id, peer := range n.peers {
		if peer.LastSeen.Before(cutoff) {
			peer.Connection.Close()
			delete(n.peers, id)
		}
	}
}

// discoverNewPeers attempts to find and connect to new peers
func (n *Node) discoverNewPeers() {
	n.mu.RLock()
	currentPeerCount := len(n.peers)
	n.mu.RUnlock()

	if currentPeerCount < n.config.MinPeers {
		// Request peers from existing connections
		msg := &Message{
			Type: LegacyMessageTypePeerRequest,
		}
		n.BroadcastMessage(msg)
	}
}

// gossipLoop periodically shares information with peers
func (n *Node) gossipLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.sendPing()
		}
	}
}

// sendPing sends a ping to all peers
func (n *Node) sendPing() {
	msg := &Message{
		Type:    LegacyMessageTypePing,
		Payload: []byte(fmt.Sprintf("%d", time.Now().Unix())),
	}
	n.BroadcastMessage(msg)
}

// Helper functions

func (n *Node) performHandshake(conn net.Conn, initiator bool) error {
	// Simplified handshake - in production, use proper protocol
	handshake := map[string]interface{}{
		"nodeID":    n.id,
		"publicKey": hex.EncodeToString(n.pubKey),
		"version":   "1.0.0",
		"timestamp": time.Now().Unix(),
	}

	data, err := json.Marshal(handshake)
	if err != nil {
		return err
	}

	if initiator {
		// Send our handshake
		if _, err := conn.Write(data); err != nil {
			return err
		}

		// Read peer handshake
		buf := make([]byte, 4096)
		nRead, err := conn.Read(buf)
		if err != nil {
			return err
		}

		var peerHandshake map[string]interface{}
		if err := json.Unmarshal(buf[:nRead], &peerHandshake); err != nil {
			return err
		}

		// Create peer
		n.addPeer(conn, peerHandshake)
	} else {
		// Read peer handshake first
		buf := make([]byte, 4096)
		nRead, err := conn.Read(buf)
		if err != nil {
			return err
		}

		var peerHandshake map[string]interface{}
		if err := json.Unmarshal(buf[:nRead], &peerHandshake); err != nil {
			return err
		}

		// Send our handshake
		if _, err := conn.Write(data); err != nil {
			return err
		}

		// Create peer
		n.addPeer(conn, peerHandshake)
	}

	return nil
}

func (n *Node) addPeer(conn net.Conn, handshake map[string]interface{}) {
	peerID := handshake["nodeID"].(string)

	peer := &NodePeer{
		ID:         peerID,
		Address:    conn.RemoteAddr().String(),
		Connection: conn,
		LastSeen:   time.Now(),
		Reputation: 5.0,
	}

	n.mu.Lock()
	n.peers[peerID] = peer
	n.mu.Unlock()
}

func (n *Node) removePeer(conn net.Conn) {
	n.mu.Lock()
	defer n.mu.Unlock()

	for id, peer := range n.peers {
		if peer.Connection == conn {
			delete(n.peers, id)
			break
		}
	}
}

func (n *Node) readMessage(conn net.Conn) (*Message, error) {
	// Simplified message reading - in production, use proper framing
	buf := make([]byte, 65536)
	nRead, err := conn.Read(buf)
	if err != nil {
		return nil, err
	}

	var msg Message
	if err := json.Unmarshal(buf[:nRead], &msg); err != nil {
		return nil, err
	}

	// TODO: Verify signature

	return &msg, nil
}

func (n *Node) sendToPeer(peer *NodePeer, msg *Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	_, err = peer.Connection.Write(data)
	if err != nil {
		// Remove dead peer
		n.mu.Lock()
		delete(n.peers, peer.ID)
		n.mu.Unlock()
		return err
	}

	n.bytesSent += uint64(len(data))
	return nil
}

// Message handlers

func (n *Node) handlePing(msg *Message) {
	// Update peer's last seen time
	n.mu.Lock()
	if peer, exists := n.peers[msg.Sender]; exists {
		peer.LastSeen = time.Now()
	}
	n.mu.Unlock()

	// Send pong response
	pong := &Message{
		Type:    LegacyMessageTypePong,
		Payload: msg.Payload,
	}
	n.SendMessage(msg.Sender, pong)
}

func (n *Node) handlePong(msg *Message) {
	// Update peer's last seen time
	n.mu.Lock()
	if peer, exists := n.peers[msg.Sender]; exists {
		peer.LastSeen = time.Now()
	}
	n.mu.Unlock()
}

func (n *Node) handleBlockProposal(msg *Message) {
	// Forward to consensus engine
	// This would integrate with the consensus.Engine
}

func (n *Node) handleBlockVote(msg *Message) {
	// Forward to consensus engine
	// This would integrate with the consensus.Engine
}

func (n *Node) handleConsensusMessage(msg *Message) {
	// Forward to consensus engine
	// This would integrate with the consensus.Engine
}

func (n *Node) handlePeerRequest(msg *Message) {
	// Share known peers
	n.mu.RLock()
	peerList := make([]string, 0, len(n.peers))
	for _, peer := range n.peers {
		peerList = append(peerList, peer.Address)
	}
	n.mu.RUnlock()

	response, _ := json.Marshal(peerList)
	reply := &Message{
		Type:    LegacyMessageTypePeerResponse,
		Payload: response,
	}
	n.SendMessage(msg.Sender, reply)
}

func (n *Node) handlePeerResponse(msg *Message) {
	var peerList []string
	if err := json.Unmarshal(msg.Payload, &peerList); err != nil {
		return
	}

	// Try to connect to new peers
	for _, addr := range peerList {
		go n.Connect(addr)
	}
}

// GetStats returns network statistics
func (n *Node) GetStats() map[string]interface{} {
	n.mu.RLock()
	peerCount := len(n.peers)
	peers := make([]string, 0, peerCount)
	for id := range n.peers {
		peers = append(peers, id)
	}
	n.mu.RUnlock()

	return map[string]interface{}{
		"nodeID":           n.id,
		"address":          n.address,
		"peerCount":        peerCount,
		"peers":            peers,
		"messagesReceived": n.messagesReceived,
		"messagesSent":     n.messagesSent,
		"bytesReceived":    n.bytesReceived,
		"bytesSent":        n.bytesSent,
	}
}