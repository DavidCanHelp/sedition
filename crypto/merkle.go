package crypto

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
)

// MerkleTree implements a binary Merkle tree for efficient verification
type MerkleTree struct {
	root   []byte
	leaves [][]byte
	levels [][][]byte // All intermediate nodes by level
}

// NewMerkleTree creates a Merkle tree from data
func NewMerkleTree(data [][]byte) (*MerkleTree, error) {
	if len(data) == 0 {
		return nil, errors.New("cannot create Merkle tree with no data")
	}
	
	// Hash all data to create leaves
	leaves := make([][]byte, len(data))
	for i, d := range data {
		hash := sha256.Sum256(d)
		leaves[i] = hash[:]
	}
	
	tree := &MerkleTree{
		leaves: leaves,
		levels: make([][][]byte, 0),
	}
	
	// Build tree bottom-up
	tree.build()
	
	return tree, nil
}

// build constructs the Merkle tree
func (m *MerkleTree) build() {
	currentLevel := make([][]byte, len(m.leaves))
	copy(currentLevel, m.leaves)
	m.levels = append(m.levels, currentLevel)
	
	for len(currentLevel) > 1 {
		nextLevel := make([][]byte, 0, (len(currentLevel)+1)/2)
		
		for i := 0; i < len(currentLevel); i += 2 {
			var hash [32]byte
			if i+1 < len(currentLevel) {
				// Hash pair of nodes
				combined := append(currentLevel[i], currentLevel[i+1]...)
				hash = sha256.Sum256(combined)
			} else {
				// Odd number of nodes, hash with itself
				combined := append(currentLevel[i], currentLevel[i]...)
				hash = sha256.Sum256(combined)
			}
			nextLevel = append(nextLevel, hash[:])
		}
		
		m.levels = append(m.levels, nextLevel)
		currentLevel = nextLevel
	}
	
	m.root = currentLevel[0]
}

// GetRoot returns the Merkle root
func (m *MerkleTree) GetRoot() []byte {
	return m.root
}

// GetRootHex returns the Merkle root as hex string
func (m *MerkleTree) GetRootHex() string {
	return hex.EncodeToString(m.root)
}

// GetProof generates a Merkle proof for a leaf at given index
func (m *MerkleTree) GetProof(index int) (*MerkleProof, error) {
	if index < 0 || index >= len(m.leaves) {
		return nil, errors.New("index out of range")
	}
	
	proof := &MerkleProof{
		Leaf:     m.leaves[index],
		Index:    index,
		Siblings: make([][]byte, 0),
	}
	
	// Traverse tree from leaf to root
	currentIndex := index
	for level := 0; level < len(m.levels)-1; level++ {
		levelNodes := m.levels[level]
		siblingIndex := currentIndex ^ 1 // XOR with 1 to get sibling
		
		if siblingIndex < len(levelNodes) {
			proof.Siblings = append(proof.Siblings, levelNodes[siblingIndex])
		} else {
			// No sibling, use same node
			proof.Siblings = append(proof.Siblings, levelNodes[currentIndex])
		}
		
		currentIndex /= 2
	}
	
	return proof, nil
}

// VerifyProof verifies a Merkle proof
func (m *MerkleTree) VerifyProof(proof *MerkleProof) bool {
	return VerifyMerkleProof(proof, m.root)
}

// MerkleProof represents a proof of inclusion
type MerkleProof struct {
	Leaf     []byte   // The leaf being proved
	Index    int      // Index of the leaf
	Siblings [][]byte // Sibling hashes from leaf to root
}

// VerifyMerkleProof verifies a proof against a root
func VerifyMerkleProof(proof *MerkleProof, root []byte) bool {
	currentHash := proof.Leaf
	currentIndex := proof.Index
	
	for _, sibling := range proof.Siblings {
		var combined []byte
		if currentIndex%2 == 0 {
			// Current node is left child
			combined = append(currentHash, sibling...)
		} else {
			// Current node is right child
			combined = append(sibling, currentHash...)
		}
		
		hash := sha256.Sum256(combined)
		currentHash = hash[:]
		currentIndex /= 2
	}
	
	return bytes.Equal(currentHash, root)
}

// CompactMerkleTree implements a space-efficient Merkle tree
type CompactMerkleTree struct {
	hasher    func([]byte) []byte
	leafCount int
	nodes     map[string][]byte // Store only necessary nodes
	root      []byte
}

// NewCompactMerkleTree creates a compact Merkle tree
func NewCompactMerkleTree() *CompactMerkleTree {
	return &CompactMerkleTree{
		hasher: func(data []byte) []byte {
			hash := sha256.Sum256(data)
			return hash[:]
		},
		nodes: make(map[string][]byte),
	}
}

// AddLeaf adds a leaf to the compact tree
func (c *CompactMerkleTree) AddLeaf(data []byte) {
	leafHash := c.hasher(data)
	c.nodes[c.nodeKey(0, c.leafCount)] = leafHash
	c.leafCount++
	c.updateRoot()
}

// nodeKey generates a unique key for a node
func (c *CompactMerkleTree) nodeKey(level, index int) string {
	return fmt.Sprintf("%d:%d", level, index)
}

// updateRoot recalculates the root after adding leaves
func (c *CompactMerkleTree) updateRoot() {
	if c.leafCount == 0 {
		c.root = nil
		return
	}
	
	// Calculate tree height
	height := int(math.Ceil(math.Log2(float64(c.leafCount))))
	if c.leafCount == 1 {
		height = 0
	}
	
	// Build tree level by level
	for level := 0; level < height; level++ {
		levelSize := (c.leafCount + (1 << level) - 1) >> level
		for i := 0; i < levelSize; i += 2 {
			left := c.nodes[c.nodeKey(level, i)]
			right := c.nodes[c.nodeKey(level, i+1)]
			
			if right == nil {
				right = left // Duplicate if odd number
			}
			
			combined := append(left, right...)
			parentHash := c.hasher(combined)
			c.nodes[c.nodeKey(level+1, i/2)] = parentHash
		}
	}
	
	c.root = c.nodes[c.nodeKey(height, 0)]
}

// GetRoot returns the current root
func (c *CompactMerkleTree) GetRoot() []byte {
	return c.root
}

// SparseMerkleTree implements a sparse Merkle tree for large key spaces
type SparseMerkleTree struct {
	depth    int
	root     []byte
	db       map[string][]byte // In production, use persistent storage
	emptyVal []byte
}

// NewSparseMerkleTree creates a sparse Merkle tree
func NewSparseMerkleTree(depth int) *SparseMerkleTree {
	emptyVal := make([]byte, 32) // Zero hash for empty nodes
	return &SparseMerkleTree{
		depth:    depth,
		db:       make(map[string][]byte),
		emptyVal: emptyVal,
		root:     emptyVal,
	}
}

// Update sets a value at a key in the sparse tree
func (s *SparseMerkleTree) Update(key, value []byte) error {
	if len(key) != 32 {
		return errors.New("key must be 32 bytes")
	}
	
	path := s.getPath(key)
	s.root = s.updateNode(s.root, path, 0, value)
	return nil
}

// getPath converts key to bit path
func (s *SparseMerkleTree) getPath(key []byte) []bool {
	path := make([]bool, s.depth)
	for i := 0; i < s.depth; i++ {
		byteIndex := i / 8
		bitIndex := uint(7 - (i % 8))
		path[i] = (key[byteIndex] >> bitIndex) & 1 == 1
	}
	return path
}

// updateNode recursively updates nodes
func (s *SparseMerkleTree) updateNode(nodeHash []byte, path []bool, depth int, value []byte) []byte {
	if depth == s.depth {
		// Leaf level
		hash := sha256.Sum256(value)
		return hash[:]
	}
	
	// Get current node
	node := s.getNode(nodeHash)
	
	// Update appropriate child
	if path[depth] {
		// Update right child
		node.Right = s.updateNode(node.Right, path, depth+1, value)
	} else {
		// Update left child
		node.Left = s.updateNode(node.Left, path, depth+1, value)
	}
	
	// Calculate new hash
	combined := append(node.Left, node.Right...)
	newHash := sha256.Sum256(combined)
	
	// Store updated node
	s.storeNode(newHash[:], node)
	
	return newHash[:]
}

// sparseNode represents a node in the sparse tree
type sparseNode struct {
	Left  []byte
	Right []byte
}

// getNode retrieves a node from storage
func (s *SparseMerkleTree) getNode(hash []byte) *sparseNode {
	if bytes.Equal(hash, s.emptyVal) {
		return &sparseNode{
			Left:  s.emptyVal,
			Right: s.emptyVal,
		}
	}
	
	key := hex.EncodeToString(hash)
	data, exists := s.db[key]
	if !exists {
		return &sparseNode{
			Left:  s.emptyVal,
			Right: s.emptyVal,
		}
	}
	
	// Decode node (simplified)
	if len(data) >= 64 {
		return &sparseNode{
			Left:  data[:32],
			Right: data[32:64],
		}
	}
	
	return &sparseNode{
		Left:  s.emptyVal,
		Right: s.emptyVal,
	}
}

// storeNode saves a node to storage
func (s *SparseMerkleTree) storeNode(hash []byte, node *sparseNode) {
	key := hex.EncodeToString(hash)
	data := append(node.Left, node.Right...)
	s.db[key] = data
}

// Get retrieves a value from the sparse tree
func (s *SparseMerkleTree) Get(key []byte) ([]byte, *SparseMerkleProof, error) {
	if len(key) != 32 {
		return nil, nil, errors.New("key must be 32 bytes")
	}
	
	path := s.getPath(key)
	proof := &SparseMerkleProof{
		Key:      key,
		Siblings: make([][]byte, s.depth),
	}
	
	currentHash := s.root
	for i := 0; i < s.depth; i++ {
		node := s.getNode(currentHash)
		
		if path[i] {
			proof.Siblings[i] = node.Left
			currentHash = node.Right
		} else {
			proof.Siblings[i] = node.Right
			currentHash = node.Left
		}
	}
	
	return currentHash, proof, nil
}

// SparseMerkleProof represents a proof in a sparse Merkle tree
type SparseMerkleProof struct {
	Key      []byte
	Value    []byte
	Siblings [][]byte
}

// Verify checks a sparse Merkle proof
func (p *SparseMerkleProof) Verify(root []byte, depth int) bool {
	path := make([]bool, depth)
	for i := 0; i < depth; i++ {
		byteIndex := i / 8
		bitIndex := uint(7 - (i % 8))
		path[i] = (p.Key[byteIndex] >> bitIndex) & 1 == 1
	}
	
	currentHash := sha256.Sum256(p.Value)
	hash := currentHash[:]
	
	for i := depth - 1; i >= 0; i-- {
		var combined []byte
		if path[i] {
			combined = append(p.Siblings[i], hash...)
		} else {
			combined = append(hash, p.Siblings[i]...)
		}
		
		newHash := sha256.Sum256(combined)
		hash = newHash[:]
	}
	
	return bytes.Equal(hash, root)
}

// GetRoot returns the sparse tree root
func (s *SparseMerkleTree) GetRoot() []byte {
	return s.root
}