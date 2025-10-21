package network

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/config"
	"github.com/davidcanhelp/sedition/consensus"
)

func TestNewNode(t *testing.T) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	node, err := NewNode("localhost:0", engine, cfg)
	if err != nil {
		t.Fatalf("failed to create node: %v", err)
	}

	if node.id == "" {
		t.Error("node ID should not be empty")
	}
	if node.pubKey == nil {
		t.Error("public key should not be nil")
	}
	if node.privKey == nil {
		t.Error("private key should not be nil")
	}
	if len(node.peers) != 0 {
		t.Error("initial peer list should be empty")
	}
}

func TestNode_StartStop(t *testing.T) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	node, err := NewNode("localhost:0", engine, cfg)
	if err != nil {
		t.Fatalf("failed to create node: %v", err)
	}

	// Start node
	if err := node.Start(); err != nil {
		t.Fatalf("failed to start node: %v", err)
	}

	// Give it time to initialize
	time.Sleep(100 * time.Millisecond)

	// Stop node
	if err := node.Stop(); err != nil {
		t.Fatalf("failed to stop node: %v", err)
	}
}

func TestNode_PeerConnection(t *testing.T) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	// Create two nodes
	node1, err := NewNode("localhost:0", engine, cfg)
	if err != nil {
		t.Fatalf("failed to create node1: %v", err)
	}

	node2, err := NewNode("localhost:0", engine, cfg)
	if err != nil {
		t.Fatalf("failed to create node2: %v", err)
	}

	// Start both nodes
	if err := node1.Start(); err != nil {
		t.Fatalf("failed to start node1: %v", err)
	}
	defer node1.Stop()

	if err := node2.Start(); err != nil {
		t.Fatalf("failed to start node2: %v", err)
	}
	defer node2.Stop()

	// Get node1's actual listening address
	addr1 := node1.listener.Addr().String()

	// Node2 connects to node1
	if err := node2.Connect(addr1); err != nil {
		t.Fatalf("failed to connect: %v", err)
	}

	// Wait for connection to establish
	time.Sleep(500 * time.Millisecond)

	// Check peer counts
	node1.mu.RLock()
	node1PeerCount := len(node1.peers)
	node1.mu.RUnlock()

	node2.mu.RLock()
	node2PeerCount := len(node2.peers)
	node2.mu.RUnlock()

	if node1PeerCount != 1 {
		t.Errorf("node1 should have 1 peer, has %d", node1PeerCount)
	}
	if node2PeerCount != 1 {
		t.Errorf("node2 should have 1 peer, has %d", node2PeerCount)
	}
}

func TestNode_BroadcastMessage(t *testing.T) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	// Create three nodes
	nodes := make([]*Node, 3)
	for i := 0; i < 3; i++ {
		node, err := NewNode("localhost:0", engine, cfg)
		if err != nil {
			t.Fatalf("failed to create node%d: %v", i, err)
		}
		nodes[i] = node

		if err := node.Start(); err != nil {
			t.Fatalf("failed to start node%d: %v", i, err)
		}
		defer node.Stop()
	}

	// Connect nodes in a chain: 0 <-> 1 <-> 2
	addr0 := nodes[0].listener.Addr().String()
	addr1 := nodes[1].listener.Addr().String()

	if err := nodes[1].Connect(addr0); err != nil {
		t.Fatalf("failed to connect node1 to node0: %v", err)
	}
	if err := nodes[2].Connect(addr1); err != nil {
		t.Fatalf("failed to connect node2 to node1: %v", err)
	}

	// Wait for connections
	time.Sleep(500 * time.Millisecond)

	// Node1 broadcasts a message
	testMessage := &Message{
		Type:    LegacyMessageTypeConsensusMessage,
		Payload: []byte("test broadcast"),
	}

	if err := nodes[1].BroadcastMessage(testMessage); err != nil {
		t.Fatalf("failed to broadcast message: %v", err)
	}

	// Wait for message propagation
	time.Sleep(100 * time.Millisecond)

	// Check that messages were sent
	if nodes[1].messagesSent == 0 {
		t.Error("node1 should have sent messages")
	}
}

func TestNode_GetStats(t *testing.T) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	node, err := NewNode("localhost:0", engine, cfg)
	if err != nil {
		t.Fatalf("failed to create node: %v", err)
	}

	if err := node.Start(); err != nil {
		t.Fatalf("failed to start node: %v", err)
	}
	defer node.Stop()

	stats := node.GetStats()

	// Check required stats fields
	if stats["nodeID"] == "" {
		t.Error("stats should include nodeID")
	}
	if stats["peerCount"] == nil {
		t.Error("stats should include peerCount")
	}
	if stats["messagesReceived"] == nil {
		t.Error("stats should include messagesReceived")
	}
	if stats["messagesSent"] == nil {
		t.Error("stats should include messagesSent")
	}
}

func TestNode_MessageHandling(t *testing.T) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	node, err := NewNode("localhost:0", engine, cfg)
	if err != nil {
		t.Fatalf("failed to create node: %v", err)
	}

	if err := node.Start(); err != nil {
		t.Fatalf("failed to start node: %v", err)
	}
	defer node.Stop()

	// Test ping message
	pingMsg := &Message{
		Type:    LegacyMessageTypePing,
		Sender:  "test-peer",
		Payload: []byte(fmt.Sprintf("%d", time.Now().Unix())),
	}

	node.handleMessage(pingMsg)

	// Test peer request
	peerReqMsg := &Message{
		Type:   LegacyMessageTypePeerRequest,
		Sender: "test-peer",
	}

	node.handleMessage(peerReqMsg)

	// Test peer response
	peerList := []string{"peer1:8545", "peer2:8545"}
	peerData, _ := json.Marshal(peerList)
	peerRespMsg := &Message{
		Type:    LegacyMessageTypePeerResponse,
		Sender:  "test-peer",
		Payload: peerData,
	}

	node.handleMessage(peerRespMsg)
}

func TestNode_ConcurrentOperations(t *testing.T) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	node, err := NewNode("localhost:0", engine, cfg)
	if err != nil {
		t.Fatalf("failed to create node: %v", err)
	}

	if err := node.Start(); err != nil {
		t.Fatalf("failed to start node: %v", err)
	}
	defer node.Stop()

	// Simulate concurrent message sending
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	done := make(chan bool)

	// Multiple goroutines sending messages
	for i := 0; i < 10; i++ {
		go func(id int) {
			for {
				select {
				case <-ctx.Done():
					done <- true
					return
				default:
					msg := &Message{
						Type:    LegacyMessageTypePing,
						Payload: []byte(fmt.Sprintf("ping-%d", id)),
					}
					node.BroadcastMessage(msg)
					time.Sleep(10 * time.Millisecond)
				}
			}
		}(i)
	}

	// Wait for all goroutines to finish
	for i := 0; i < 10; i++ {
		<-done
	}

	// No panic means concurrent operations are safe
	t.Logf("Sent %d messages successfully", node.messagesSent)
}

// Benchmark tests
func BenchmarkNode_BroadcastMessage(b *testing.B) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	node, _ := NewNode("localhost:0", engine, cfg)
	node.Start()
	defer node.Stop()

	// Add some mock peers
	for i := 0; i < 10; i++ {
		node.peers[fmt.Sprintf("peer%d", i)] = &NodePeer{
			ID:       fmt.Sprintf("peer%d", i),
			LastSeen: time.Now(),
		}
	}

	msg := &Message{
		Type:    LegacyMessageTypeConsensusMessage,
		Payload: []byte("benchmark message"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.BroadcastMessage(msg)
	}
}

func BenchmarkNode_HandleMessage(b *testing.B) {
	engine := consensus.NewEngine(nil)
	cfg := config.TestNetworkConfig()

	node, _ := NewNode("localhost:0", engine, cfg)
	node.Start()
	defer node.Stop()

	msg := &Message{
		Type:    LegacyMessageTypePing,
		Sender:  "benchmark-peer",
		Payload: []byte("1234567890"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node.handleMessage(msg)
	}
}