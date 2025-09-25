package network

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

// PeerConnection represents a connection to a peer node
type PeerConnection struct {
	mu sync.RWMutex

	// Peer information
	ID      string
	Address string

	// Connection
	conn   net.Conn
	reader *bufio.Reader
	writer *bufio.Writer

	// State
	isConnected bool
	lastSeen    time.Time
	sentMessages    uint64
	receivedMessages uint64

	// Channels
	sendCh chan []byte
	stopCh chan struct{}
}

// NewPeerConnection creates a new peer connection
func NewPeerConnection(address string) (*PeerConnection, error) {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to peer: %w", err)
	}

	peer := &PeerConnection{
		ID:          generatePeerID(address),
		Address:     address,
		conn:        conn,
		reader:      bufio.NewReader(conn),
		writer:      bufio.NewWriter(conn),
		isConnected: true,
		lastSeen:    time.Now(),
		sendCh:      make(chan []byte, 100),
		stopCh:      make(chan struct{}),
	}

	// Start send loop
	go peer.sendLoop()

	return peer, nil
}

// NewPeerConnectionFromConn creates a peer connection from existing connection
func NewPeerConnectionFromConn(conn net.Conn) *PeerConnection {
	address := conn.RemoteAddr().String()
	peer := &PeerConnection{
		ID:          generatePeerID(address),
		Address:     address,
		conn:        conn,
		reader:      bufio.NewReader(conn),
		writer:      bufio.NewWriter(conn),
		isConnected: true,
		lastSeen:    time.Now(),
		sendCh:      make(chan []byte, 100),
		stopCh:      make(chan struct{}),
	}

	// Start send loop
	go peer.sendLoop()

	return peer
}

// SendMessage sends a message to the peer
func (p *PeerConnection) SendMessage(message *BroadcastMessage) error {
	if !p.IsConnected() {
		return fmt.Errorf("peer not connected")
	}

	data, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	// Add newline delimiter
	data = append(data, '\n')

	select {
	case p.sendCh <- data:
		p.mu.Lock()
		p.sentMessages++
		p.mu.Unlock()
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("send timeout")
	}
}

// ReceiveMessage receives a message from the peer
func (p *PeerConnection) ReceiveMessage() (*BroadcastMessage, error) {
	if !p.IsConnected() {
		return nil, fmt.Errorf("peer not connected")
	}

	// Set read deadline
	p.conn.SetReadDeadline(time.Now().Add(30 * time.Second))

	// Read line (message terminated by newline)
	data, err := p.reader.ReadBytes('\n')
	if err != nil {
		if err == io.EOF {
			p.Close()
			return nil, fmt.Errorf("peer disconnected")
		}
		return nil, fmt.Errorf("failed to read message: %w", err)
	}

	// Parse message
	var message BroadcastMessage
	if err := json.Unmarshal(data[:len(data)-1], &message); err != nil {
		return nil, fmt.Errorf("failed to unmarshal message: %w", err)
	}

	p.mu.Lock()
	p.receivedMessages++
	p.lastSeen = time.Now()
	p.mu.Unlock()

	return &message, nil
}

// IsConnected returns whether the peer is connected
func (p *PeerConnection) IsConnected() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.isConnected
}

// Close closes the peer connection
func (p *PeerConnection) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.isConnected {
		return nil
	}

	p.isConnected = false
	close(p.stopCh)
	return p.conn.Close()
}

// sendLoop handles sending messages
func (p *PeerConnection) sendLoop() {
	for {
		select {
		case <-p.stopCh:
			return
		case data := <-p.sendCh:
			if err := p.writeData(data); err != nil {
				fmt.Printf("Failed to write to peer %s: %v\n", p.ID, err)
				p.Close()
				return
			}
		}
	}
}

// writeData writes data to the connection
func (p *PeerConnection) writeData(data []byte) error {
	// Set write deadline
	p.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

	if _, err := p.writer.Write(data); err != nil {
		return err
	}

	return p.writer.Flush()
}

// GetStats returns peer statistics
func (p *PeerConnection) GetStats() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return map[string]interface{}{
		"id":               p.ID,
		"address":          p.Address,
		"is_connected":     p.isConnected,
		"last_seen":        p.lastSeen,
		"sent_messages":    p.sentMessages,
		"received_messages": p.receivedMessages,
	}
}

// generatePeerID generates a unique peer ID
func generatePeerID(address string) string {
	return fmt.Sprintf("peer_%s_%d", address, time.Now().UnixNano())
}

// PeerListener listens for incoming peer connections
type PeerListener struct {
	mu sync.RWMutex

	// Configuration
	address string

	// Listener
	listener net.Listener

	// State
	isListening bool

	// Callback for new connections
	onNewPeer func(*PeerConnection)
}

// NewPeerListener creates a new peer listener
func NewPeerListener(address string, onNewPeer func(*PeerConnection)) *PeerListener {
	return &PeerListener{
		address:   address,
		onNewPeer: onNewPeer,
	}
}

// Start starts listening for connections
func (l *PeerListener) Start() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.isListening {
		return fmt.Errorf("already listening")
	}

	listener, err := net.Listen("tcp", l.address)
	if err != nil {
		return fmt.Errorf("failed to start listener: %w", err)
	}

	l.listener = listener
	l.isListening = true

	// Start accepting connections
	go l.acceptLoop()

	fmt.Printf("Peer listener started on %s\n", l.address)
	return nil
}

// Stop stops the listener
func (l *PeerListener) Stop() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if !l.isListening {
		return fmt.Errorf("not listening")
	}

	l.isListening = false
	return l.listener.Close()
}

// acceptLoop accepts incoming connections
func (l *PeerListener) acceptLoop() {
	for {
		conn, err := l.listener.Accept()
		if err != nil {
			l.mu.RLock()
			isListening := l.isListening
			l.mu.RUnlock()

			if !isListening {
				return
			}

			fmt.Printf("Failed to accept connection: %v\n", err)
			continue
		}

		// Create peer connection
		peer := NewPeerConnectionFromConn(conn)

		// Call callback
		if l.onNewPeer != nil {
			l.onNewPeer(peer)
		}
	}
}

// GetAddress returns the listener address
func (l *PeerListener) GetAddress() string {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if l.listener != nil {
		return l.listener.Addr().String()
	}
	return l.address
}