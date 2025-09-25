package network

import (
	"encoding/json"
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockPeerConnection for testing
type MockPeerConnection struct {
	ID               string
	sentMessages     []*BroadcastMessage
	receivedMessages chan *BroadcastMessage
	isConnected      bool
}

func NewMockPeerConnection(id string) *MockPeerConnection {
	return &MockPeerConnection{
		ID:               id,
		sentMessages:     make([]*BroadcastMessage, 0),
		receivedMessages: make(chan *BroadcastMessage, 100),
		isConnected:      true,
	}
}

func (m *MockPeerConnection) SendMessage(message *BroadcastMessage) error {
	if !m.isConnected {
		return fmt.Errorf("not connected")
	}
	m.sentMessages = append(m.sentMessages, message)
	return nil
}

func (m *MockPeerConnection) ReceiveMessage() (*BroadcastMessage, error) {
	if !m.isConnected {
		return nil, fmt.Errorf("not connected")
	}
	select {
	case msg := <-m.receivedMessages:
		return msg, nil
	case <-time.After(time.Second):
		return nil, fmt.Errorf("receive timeout")
	}
}

func (m *MockPeerConnection) IsConnected() bool {
	return m.isConnected
}

func (m *MockPeerConnection) Close() error {
	m.isConnected = false
	close(m.receivedMessages)
	return nil
}

func (m *MockPeerConnection) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"id":            m.ID,
		"sent_messages": len(m.sentMessages),
		"is_connected":  m.isConnected,
	}
}

func TestBroadcasterStartStop(t *testing.T) {
	config := DefaultBroadcasterConfig()
	broadcaster := NewBroadcaster("test-node", config)

	// Start broadcaster
	err := broadcaster.Start()
	require.NoError(t, err)
	assert.True(t, broadcaster.isRunning)

	// Try starting again - should fail
	err = broadcaster.Start()
	assert.Error(t, err)

	// Stop broadcaster
	err = broadcaster.Stop()
	require.NoError(t, err)
	assert.False(t, broadcaster.isRunning)

	// Try stopping again - should fail
	err = broadcaster.Stop()
	assert.Error(t, err)
}

func TestBroadcastTransaction(t *testing.T) {
	config := DefaultBroadcasterConfig()
	broadcaster := NewBroadcaster("test-node", config)

	err := broadcaster.Start()
	require.NoError(t, err)
	defer broadcaster.Stop()

	// Add mock peers
	peer1 := &PeerConnection{ID: "peer1"}
	peer2 := &PeerConnection{ID: "peer2"}

	// We need to use the actual peer structure for this test
	// For now, we'll test the broadcast method directly

	// Create test transaction
	tx := &storage.Transaction{
		Hash:      "test_tx_hash",
		From:      "alice",
		To:        "bob",
		Value:     big.NewInt(1000),
		Nonce:     0,
		GasLimit:  21000,
		GasPrice:  big.NewInt(1000000000),
		Timestamp: time.Now(),
		Signature: []byte("test_signature"),
	}

	// Broadcast transaction
	err = broadcaster.BroadcastTransaction(tx)
	require.NoError(t, err)

	// Check that message was added to outgoing queue
	select {
	case msg := <-broadcaster.outgoingMessages:
		assert.Equal(t, MessageTypeTransaction, msg.Type)
		assert.Equal(t, broadcaster.nodeID, msg.Sender)

		// Parse payload
		var payload TransactionPayload
		err = json.Unmarshal(msg.Payload, &payload)
		require.NoError(t, err)
		assert.Equal(t, tx.Hash, payload.Transaction.Hash)
	case <-time.After(time.Second):
		t.Fatal("No message in outgoing queue")
	}
}

func TestBroadcastBlock(t *testing.T) {
	config := DefaultBroadcasterConfig()
	broadcaster := NewBroadcaster("test-node", config)

	err := broadcaster.Start()
	require.NoError(t, err)
	defer broadcaster.Stop()

	// Create test block
	block := &storage.Block{
		Height:       1,
		PreviousHash: "0000000000000000000000000000000000000000000000000000000000000000",
		Timestamp:    time.Now(),
		Proposer:     "test-proposer",
		StateRoot:    "state_root",
		TxRoot:       "tx_root",
		Hash:         "block_hash",
		Transactions: []*storage.Transaction{},
	}

	// Broadcast block
	err = broadcaster.BroadcastBlock(block)
	require.NoError(t, err)

	// Check that message was added to outgoing queue
	select {
	case msg := <-broadcaster.outgoingMessages:
		assert.Equal(t, MessageTypeBlock, msg.Type)
		assert.Equal(t, broadcaster.nodeID, msg.Sender)

		// Parse payload
		var payload BlockPayload
		err = json.Unmarshal(msg.Payload, &payload)
		require.NoError(t, err)
		assert.Equal(t, block.Hash, payload.Block.Hash)
	case <-time.After(time.Second):
		t.Fatal("No message in outgoing queue")
	}
}

func TestMessageHandlerRegistration(t *testing.T) {
	config := DefaultBroadcasterConfig()
	broadcaster := NewBroadcaster("test-node", config)

	// Track if handler was called
	handlerCalled := false

	// Register handler
	broadcaster.RegisterHandler(MessageTypeTransaction, func(message *BroadcastMessage, sender *PeerConnection) error {
		handlerCalled = true
		return nil
	})

	err := broadcaster.Start()
	require.NoError(t, err)
	defer broadcaster.Stop()

	// Create test message
	tx := &storage.Transaction{
		Hash: "test_tx",
	}
	payload, _ := json.Marshal(&TransactionPayload{Transaction: tx})

	message := &BroadcastMessage{
		Type:      MessageTypeTransaction,
		Payload:   payload,
		Timestamp: time.Now(),
		Sender:    "other-node",
		ID:        "test-message-id",
	}

	// Send message to incoming queue
	broadcaster.incomingMessages <- message

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Check handler was called
	assert.True(t, handlerCalled)
}

func TestDuplicateMessagePrevention(t *testing.T) {
	config := DefaultBroadcasterConfig()
	broadcaster := NewBroadcaster("test-node", config)

	handleCount := 0
	broadcaster.RegisterHandler(MessageTypeTransaction, func(message *BroadcastMessage, sender *PeerConnection) error {
		handleCount++
		return nil
	})

	err := broadcaster.Start()
	require.NoError(t, err)
	defer broadcaster.Stop()

	// Create test message
	message := &BroadcastMessage{
		Type:      MessageTypeTransaction,
		Payload:   json.RawMessage("{}"),
		Timestamp: time.Now(),
		Sender:    "other-node",
		ID:        "duplicate-test-id",
	}

	// Send same message twice
	broadcaster.incomingMessages <- message
	broadcaster.incomingMessages <- message

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Handler should only be called once
	assert.Equal(t, 1, handleCount)
}

func TestPeerManagement(t *testing.T) {
	config := DefaultBroadcasterConfig()
	config.MaxPeers = 2
	broadcaster := NewBroadcaster("test-node", config)

	// Add peers up to limit
	peer1 := &PeerConnection{ID: "peer1"}
	peer2 := &PeerConnection{ID: "peer2"}
	peer3 := &PeerConnection{ID: "peer3"}

	err := broadcaster.AddPeer(peer1)
	require.NoError(t, err)

	err = broadcaster.AddPeer(peer2)
	require.NoError(t, err)

	// Should fail - max peers reached
	err = broadcaster.AddPeer(peer3)
	assert.Error(t, err)

	// Check peer count
	peers := broadcaster.GetPeers()
	assert.Equal(t, 2, len(peers))

	// Remove a peer
	broadcaster.RemovePeer("peer1")

	// Now we should be able to add peer3
	err = broadcaster.AddPeer(peer3)
	require.NoError(t, err)

	peers = broadcaster.GetPeers()
	assert.Equal(t, 2, len(peers))
}

func TestGetStats(t *testing.T) {
	config := DefaultBroadcasterConfig()
	broadcaster := NewBroadcaster("test-node", config)

	err := broadcaster.Start()
	require.NoError(t, err)
	defer broadcaster.Stop()

	// Add a peer
	peer := &PeerConnection{ID: "test-peer"}
	broadcaster.AddPeer(peer)

	// Process a message to update processed count
	message := &BroadcastMessage{
		Type:      MessageTypeHeartbeat,
		Payload:   json.RawMessage("{}"),
		Timestamp: time.Now(),
		Sender:    "other-node",
		ID:        "test-message",
	}
	broadcaster.processedMessages[message.ID] = true

	// Get stats
	stats := broadcaster.GetStats()

	assert.Equal(t, "test-node", stats["node_id"])
	assert.Equal(t, 1, stats["peer_count"])
	assert.Equal(t, true, stats["is_running"])
	assert.Equal(t, 1, stats["processed_messages"])
}