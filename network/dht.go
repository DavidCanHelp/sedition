// Package network implements Distributed Hash Table (DHT) for peer discovery
package network

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// DHT implements a Kademlia-style Distributed Hash Table
type DHT struct {
	mu sync.RWMutex

	nodeID    string
	buckets   []*KBucket
	bucketNum int

	// Configuration
	k     int // Bucket size (default: 20)
	alpha int // Concurrency parameter (default: 3)

	// Routing table
	routingTable map[string]*DHTNode

	// Storage for key-value pairs
	storage map[string]*DHTRecord

	// Metrics
	lookupCount int64
	storeCount  int64
	findCount   int64
	pingCount   int64
}

// DHTNode represents a node in the DHT
type DHTNode struct {
	ID       string
	Address  string
	LastSeen time.Time
	RTT      time.Duration
	Failed   int
}

// KBucket represents a k-bucket in the routing table
type KBucket struct {
	nodes    []*DHTNode
	lastUsed time.Time
	maxSize  int
}

// DHTRecord represents a stored record in the DHT
type DHTRecord struct {
	Key       string
	Value     []byte
	Publisher string
	Timestamp time.Time
	TTL       time.Duration
}

// DHTQuery represents a DHT query/lookup
type DHTQuery struct {
	QueryID   string
	Target    string
	Type      QueryType
	StartTime time.Time
	Closest   []*DHTNode
	Queried   map[string]bool
	Responses int
}

// QueryType defines the type of DHT query
type QueryType int

const (
	QueryTypeFindNode QueryType = iota
	QueryTypeFindValue
	QueryTypeStore
	QueryTypePing
)

// NewDHT creates a new DHT instance
func NewDHT(nodeID string) *DHT {
	dht := &DHT{
		nodeID:       nodeID,
		bucketNum:    160, // SHA-1 hash space (160 bits)
		k:            20,  // Standard Kademlia k
		alpha:        3,   // Standard Kademlia alpha
		routingTable: make(map[string]*DHTNode),
		storage:      make(map[string]*DHTRecord),
	}

	// Initialize k-buckets
	dht.buckets = make([]*KBucket, dht.bucketNum)
	for i := range dht.buckets {
		dht.buckets[i] = &KBucket{
			nodes:   make([]*DHTNode, 0, dht.k),
			maxSize: dht.k,
		}
	}

	return dht
}

// AddNode adds a node to the DHT routing table
func (dht *DHT) AddNode(node *DHTNode) {
	dht.mu.Lock()
	defer dht.mu.Unlock()

	// Calculate distance and bucket index
	distance := dht.xorDistance(dht.nodeID, node.ID)
	bucketIndex := dht.getBucketIndex(distance)

	bucket := dht.buckets[bucketIndex]

	// Check if node already exists
	for i, existing := range bucket.nodes {
		if existing.ID == node.ID {
			// Update existing node
			bucket.nodes[i] = node
			bucket.lastUsed = time.Now()
			dht.routingTable[node.ID] = node
			return
		}
	}

	// Add new node
	if len(bucket.nodes) < bucket.maxSize {
		bucket.nodes = append(bucket.nodes, node)
		bucket.lastUsed = time.Now()
		dht.routingTable[node.ID] = node
	} else {
		// Bucket is full, check if we should replace oldest node
		oldest := bucket.nodes[0]
		for _, n := range bucket.nodes {
			if n.LastSeen.Before(oldest.LastSeen) {
				oldest = n
			}
		}

		// Replace if oldest node is stale
		if time.Since(oldest.LastSeen) > 15*time.Minute {
			dht.replaceNode(bucket, oldest, node)
			dht.routingTable[node.ID] = node
			delete(dht.routingTable, oldest.ID)
		}
	}
}

// RemoveNode removes a node from the DHT
func (dht *DHT) RemoveNode(nodeID string) {
	dht.mu.Lock()
	defer dht.mu.Unlock()

	_, exists := dht.routingTable[nodeID]
	if !exists {
		return
	}

	// Find and remove from bucket
	distance := dht.xorDistance(dht.nodeID, nodeID)
	bucketIndex := dht.getBucketIndex(distance)
	bucket := dht.buckets[bucketIndex]

	for i, n := range bucket.nodes {
		if n.ID == nodeID {
			bucket.nodes = append(bucket.nodes[:i], bucket.nodes[i+1:]...)
			break
		}
	}

	delete(dht.routingTable, nodeID)
}

// FindClosestNodes returns the k closest nodes to a target
func (dht *DHT) FindClosestNodes(target string, count int) []*DHTNode {
	dht.mu.RLock()
	defer dht.mu.RUnlock()

	if count <= 0 {
		count = dht.k
	}

	// Collect all nodes with distances
	type nodeDistance struct {
		node     *DHTNode
		distance *big.Int
	}

	candidates := make([]nodeDistance, 0, len(dht.routingTable))
	for _, node := range dht.routingTable {
		distance := dht.xorDistance(target, node.ID)
		candidates = append(candidates, nodeDistance{
			node:     node,
			distance: distance,
		})
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance.Cmp(candidates[j].distance) < 0
	})

	// Return closest nodes
	result := make([]*DHTNode, 0, count)
	for i := 0; i < count && i < len(candidates); i++ {
		result = append(result, candidates[i].node)
	}

	return result
}

// Store stores a key-value pair in the DHT
func (dht *DHT) Store(key string, value []byte, ttl time.Duration) error {
	dht.mu.Lock()
	defer dht.mu.Unlock()

	record := &DHTRecord{
		Key:       key,
		Value:     value,
		Publisher: dht.nodeID,
		Timestamp: time.Now(),
		TTL:       ttl,
	}

	dht.storage[key] = record
	dht.storeCount++

	return nil
}

// FindValue finds a value for a key in the DHT
func (dht *DHT) FindValue(key string) ([]byte, bool) {
	dht.mu.RLock()
	defer dht.mu.RUnlock()

	record, exists := dht.storage[key]
	if !exists {
		return nil, false
	}

	// Check if record has expired
	if time.Since(record.Timestamp) > record.TTL {
		// Remove expired record
		go func() {
			dht.mu.Lock()
			delete(dht.storage, key)
			dht.mu.Unlock()
		}()
		return nil, false
	}

	dht.findCount++
	return record.Value, true
}

// Lookup performs an iterative lookup for a target
func (dht *DHT) Lookup(target string, queryType QueryType) *DHTQuery {
	dht.mu.Lock()
	query := &DHTQuery{
		QueryID:   dht.generateQueryID(),
		Target:    target,
		Type:      queryType,
		StartTime: time.Now(),
		Closest:   dht.FindClosestNodes(target, dht.k),
		Queried:   make(map[string]bool),
		Responses: 0,
	}
	dht.lookupCount++
	dht.mu.Unlock()

	// Iterative lookup process
	for {
		// Find unqueried nodes to contact
		toQuery := make([]*DHTNode, 0, dht.alpha)
		for _, node := range query.Closest {
			if !query.Queried[node.ID] && len(toQuery) < dht.alpha {
				toQuery = append(toQuery, node)
				query.Queried[node.ID] = true
			}
		}

		if len(toQuery) == 0 {
			// No more nodes to query
			break
		}

		// Query nodes concurrently (simplified simulation)
		newNodes := dht.simulateParallelQuery(toQuery, target, queryType)

		// Merge results
		allNodes := append(query.Closest, newNodes...)
		query.Closest = dht.selectClosest(allNodes, target, dht.k)
		query.Responses += len(newNodes)

		// Check convergence
		if len(newNodes) == 0 {
			break
		}
	}

	return query
}

// simulateParallelQuery simulates querying multiple nodes in parallel
func (dht *DHT) simulateParallelQuery(nodes []*DHTNode, target string, queryType QueryType) []*DHTNode {
	results := make([]*DHTNode, 0)

	for range nodes {
		// Simulate network delay
		time.Sleep(time.Millisecond * 10)

		// Simulate node response (return random close nodes)
		dht.pingCount++

		// In real implementation, this would send network messages
		// For simulation, return some random nodes from routing table
		closeNodes := dht.getRandomNodes(3) // Simulate k=3 response
		results = append(results, closeNodes...)
	}

	return results
}

// selectClosest selects the k closest nodes to target
func (dht *DHT) selectClosest(nodes []*DHTNode, target string, k int) []*DHTNode {
	type nodeDistance struct {
		node     *DHTNode
		distance *big.Int
	}

	// Remove duplicates and calculate distances
	seen := make(map[string]bool)
	candidates := make([]nodeDistance, 0)

	for _, node := range nodes {
		if seen[node.ID] {
			continue
		}
		seen[node.ID] = true

		distance := dht.xorDistance(target, node.ID)
		candidates = append(candidates, nodeDistance{
			node:     node,
			distance: distance,
		})
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance.Cmp(candidates[j].distance) < 0
	})

	// Return k closest
	result := make([]*DHTNode, 0, k)
	for i := 0; i < k && i < len(candidates); i++ {
		result = append(result, candidates[i].node)
	}

	return result
}

// Ping checks if a node is still alive
func (dht *DHT) Ping(nodeID string) bool {
	dht.mu.RLock()
	node, exists := dht.routingTable[nodeID]
	dht.mu.RUnlock()

	if !exists {
		return false
	}

	// Simulate ping (in real implementation, send network message)
	dht.pingCount++

	// Update last seen time if ping successful
	if node.Failed < 3 { // Simulate some failures
		node.LastSeen = time.Now()
		node.Failed = 0
		return true
	} else {
		node.Failed++
		return false
	}
}

// Refresh refreshes the routing table by looking up random IDs in stale buckets
func (dht *DHT) Refresh() {
	dht.mu.Lock()
	defer dht.mu.Unlock()

	now := time.Now()

	for i, bucket := range dht.buckets {
		// Refresh bucket if it hasn't been used recently
		if now.Sub(bucket.lastUsed) > time.Hour {
			// Generate random ID in this bucket's range
			randomID := dht.generateRandomIDInBucket(i)

			// Perform lookup for random ID
			go dht.Lookup(randomID, QueryTypeFindNode)
		}

		// Remove failed nodes
		activeNodes := make([]*DHTNode, 0, len(bucket.nodes))
		for _, node := range bucket.nodes {
			if node.Failed < 5 && now.Sub(node.LastSeen) < 24*time.Hour {
				activeNodes = append(activeNodes, node)
			} else {
				delete(dht.routingTable, node.ID)
			}
		}
		bucket.nodes = activeNodes
	}
}

// Bootstrap bootstraps the DHT with seed nodes
func (dht *DHT) Bootstrap(seedNodes []*DHTNode) {
	// Add seed nodes to routing table
	for _, node := range seedNodes {
		dht.AddNode(node)
	}

	// Perform lookup for own ID to populate routing table
	dht.Lookup(dht.nodeID, QueryTypeFindNode)
}

// GetStats returns DHT statistics
func (dht *DHT) GetStats() *DHTStats {
	dht.mu.RLock()
	defer dht.mu.RUnlock()

	totalNodes := len(dht.routingTable)
	bucketsUsed := 0

	for _, bucket := range dht.buckets {
		if len(bucket.nodes) > 0 {
			bucketsUsed++
		}
	}

	return &DHTStats{
		NodeID:      dht.nodeID,
		TotalNodes:  totalNodes,
		BucketsUsed: bucketsUsed,
		StoredItems: len(dht.storage),
		LookupCount: dht.lookupCount,
		StoreCount:  dht.storeCount,
		FindCount:   dht.findCount,
		PingCount:   dht.pingCount,
	}
}

// DHTStats contains DHT statistics
type DHTStats struct {
	NodeID      string
	TotalNodes  int
	BucketsUsed int
	StoredItems int
	LookupCount int64
	StoreCount  int64
	FindCount   int64
	PingCount   int64
}

// Helper functions

// xorDistance calculates XOR distance between two node IDs
func (dht *DHT) xorDistance(id1, id2 string) *big.Int {
	// Hash IDs to ensure consistent length
	hash1 := sha256.Sum256([]byte(id1))
	hash2 := sha256.Sum256([]byte(id2))

	// Calculate XOR
	result := make([]byte, 32)
	for i := 0; i < 32; i++ {
		result[i] = hash1[i] ^ hash2[i]
	}

	return new(big.Int).SetBytes(result)
}

// getBucketIndex calculates which bucket a distance belongs to
func (dht *DHT) getBucketIndex(distance *big.Int) int {
	// Find the position of the most significant bit
	bitLen := distance.BitLen()
	if bitLen == 0 {
		return 0 // Distance is 0
	}

	return dht.bucketNum - bitLen
}

// replaceNode replaces an old node with a new one in a bucket
func (dht *DHT) replaceNode(bucket *KBucket, old, new *DHTNode) {
	for i, node := range bucket.nodes {
		if node.ID == old.ID {
			bucket.nodes[i] = new
			bucket.lastUsed = time.Now()
			return
		}
	}
}

// generateQueryID generates a unique query ID
func (dht *DHT) generateQueryID() string {
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", dht.nodeID, time.Now().UnixNano())))
	return hex.EncodeToString(hash[:8]) // Use first 8 bytes
}

// generateRandomIDInBucket generates a random ID that would fall in a specific bucket
func (dht *DHT) generateRandomIDInBucket(bucketIndex int) string {
	// Generate random bytes
	randomBytes := make([]byte, 32)
	rand.Read(randomBytes)

	// Ensure the ID falls in the correct bucket
	// This is a simplified approach
	return hex.EncodeToString(randomBytes)
}

// getRandomNodes returns random nodes from routing table
func (dht *DHT) getRandomNodes(count int) []*DHTNode {
	nodes := make([]*DHTNode, 0, count)

	i := 0
	for _, node := range dht.routingTable {
		if i >= count {
			break
		}
		nodes = append(nodes, node)
		i++
	}

	return nodes
}

// ExpireRecords removes expired records from storage
func (dht *DHT) ExpireRecords() {
	dht.mu.Lock()
	defer dht.mu.Unlock()

	now := time.Now()

	for key, record := range dht.storage {
		if now.Sub(record.Timestamp) > record.TTL {
			delete(dht.storage, key)
		}
	}
}

// GetNodeInfo returns information about a specific node
func (dht *DHT) GetNodeInfo(nodeID string) (*DHTNode, bool) {
	dht.mu.RLock()
	defer dht.mu.RUnlock()

	node, exists := dht.routingTable[nodeID]
	return node, exists
}

// GetBucketInfo returns information about routing table buckets
func (dht *DHT) GetBucketInfo() []BucketInfo {
	dht.mu.RLock()
	defer dht.mu.RUnlock()

	buckets := make([]BucketInfo, len(dht.buckets))

	for i, bucket := range dht.buckets {
		buckets[i] = BucketInfo{
			Index:    i,
			Size:     len(bucket.nodes),
			MaxSize:  bucket.maxSize,
			LastUsed: bucket.lastUsed,
		}
	}

	return buckets
}

// BucketInfo contains information about a k-bucket
type BucketInfo struct {
	Index    int
	Size     int
	MaxSize  int
	LastUsed time.Time
}

// CleanupRoutingTable removes stale nodes from the routing table
func (dht *DHT) CleanupRoutingTable() {
	dht.mu.Lock()
	defer dht.mu.Unlock()

	cutoff := time.Now().Add(-24 * time.Hour)

	for nodeID, node := range dht.routingTable {
		if node.LastSeen.Before(cutoff) || node.Failed > 5 {
			// Remove from routing table
			delete(dht.routingTable, nodeID)

			// Remove from bucket
			distance := dht.xorDistance(dht.nodeID, nodeID)
			bucketIndex := dht.getBucketIndex(distance)
			bucket := dht.buckets[bucketIndex]

			for i, n := range bucket.nodes {
				if n.ID == nodeID {
					bucket.nodes = append(bucket.nodes[:i], bucket.nodes[i+1:]...)
					break
				}
			}
		}
	}
}

// FindPeersForTopic finds peers interested in a specific topic
func (dht *DHT) FindPeersForTopic(topic string) []*DHTNode {
	// Use topic hash as the key
	topicHash := sha256.Sum256([]byte(topic))
	topicKey := hex.EncodeToString(topicHash[:])

	// Find closest nodes to the topic key
	return dht.FindClosestNodes(topicKey, dht.k)
}

// AnnounceTopic announces interest in a topic
func (dht *DHT) AnnounceTopic(topic string, ttl time.Duration) error {
	topicHash := sha256.Sum256([]byte(topic))
	topicKey := hex.EncodeToString(topicHash[:])

	// Store announcement
	announcement := []byte(dht.nodeID)
	return dht.Store(topicKey, announcement, ttl)
}

// GetTopicPeers retrieves peers for a topic
func (dht *DHT) GetTopicPeers(topic string) ([]string, bool) {
	topicHash := sha256.Sum256([]byte(topic))
	topicKey := hex.EncodeToString(topicHash[:])

	data, exists := dht.FindValue(topicKey)
	if !exists {
		return nil, false
	}

	// In a real implementation, this would be a list of peer IDs
	return []string{string(data)}, true
}
