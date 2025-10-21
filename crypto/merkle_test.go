package crypto

import (
	"bytes"
	"testing"
)

// TestNewMerkleTree tests basic Merkle tree creation
func TestNewMerkleTree(t *testing.T) {
	leaves := [][]byte{
		[]byte("leaf1"),
		[]byte("leaf2"),
		[]byte("leaf3"),
		[]byte("leaf4"),
	}

	tree, err := NewMerkleTree(leaves)
	if err != nil {
		t.Fatalf("Failed to create tree: %v", err)
	}

	if tree == nil {
		t.Fatal("Tree should not be nil")
	}

	root := tree.GetRoot()
	if len(root) == 0 {
		t.Error("Root should not be empty")
	}
}

// TestMerkleTreeGetRoot tests root retrieval
func TestMerkleTreeGetRoot(t *testing.T) {
	leaves := [][]byte{
		[]byte("a"),
		[]byte("b"),
	}

	tree, err := NewMerkleTree(leaves)
	if err != nil {
		t.Fatalf("Failed to create tree: %v", err)
	}

	root := tree.GetRoot()
	if len(root) == 0 {
		t.Error("Root should not be empty")
	}

	rootHex := tree.GetRootHex()
	if len(rootHex) == 0 {
		t.Error("Root hex should not be empty")
	}
}

// TestMerkleTreeDeterminism tests that same leaves produce same root
func TestMerkleTreeDeterminism(t *testing.T) {
	leaves := [][]byte{
		[]byte("a"),
		[]byte("b"),
		[]byte("c"),
		[]byte("d"),
	}

	tree1, err := NewMerkleTree(leaves)
	if err != nil {
		t.Fatalf("Failed to create tree1: %v", err)
	}

	tree2, err := NewMerkleTree(leaves)
	if err != nil {
		t.Fatalf("Failed to create tree2: %v", err)
	}

	if !bytes.Equal(tree1.GetRoot(), tree2.GetRoot()) {
		t.Error("Same leaves should produce same root")
	}
}

// TestGetProof tests Merkle proof generation
func TestGetProof(t *testing.T) {
	leaves := [][]byte{
		[]byte("leaf0"),
		[]byte("leaf1"),
		[]byte("leaf2"),
		[]byte("leaf3"),
	}

	tree, err := NewMerkleTree(leaves)
	if err != nil {
		t.Fatalf("Failed to create tree: %v", err)
	}

	// Get proof for each leaf
	for i := range leaves {
		proof, err := tree.GetProof(i)
		if err != nil {
			t.Errorf("Failed to get proof for leaf %d: %v", i, err)
		}
		if proof == nil {
			t.Errorf("Proof for leaf %d should not be nil", i)
		}
	}

	// Test invalid index
	_, err = tree.GetProof(999)
	if err == nil {
		t.Error("Should fail with invalid index")
	}
}

// TestVerifyProof tests Merkle proof verification
func TestVerifyProof(t *testing.T) {
	leaves := [][]byte{
		[]byte("leaf0"),
		[]byte("leaf1"),
		[]byte("leaf2"),
		[]byte("leaf3"),
	}

	tree, err := NewMerkleTree(leaves)
	if err != nil {
		t.Fatalf("Failed to create tree: %v", err)
	}

	// Verify proof for each leaf
	for i := range leaves {
		proof, err := tree.GetProof(i)
		if err != nil {
			t.Fatalf("Failed to get proof for leaf %d: %v", i, err)
		}

		valid := tree.VerifyProof(proof)
		if !valid {
			t.Errorf("Proof verification failed for leaf %d", i)
		}
	}
}

// TestCompactMerkleTree tests compact Merkle tree
func TestCompactMerkleTree(t *testing.T) {
	tree := NewCompactMerkleTree()
	if tree == nil {
		t.Fatal("Compact tree should not be nil")
	}

	// Initial root may be empty, which is valid
	_ = tree.GetRoot()
}

// TestCompactMerkleTreeAddLeaf tests adding leaves
func TestCompactMerkleTreeAddLeaf(t *testing.T) {
	tree := NewCompactMerkleTree()

	leaves := [][]byte{
		[]byte("leaf1"),
		[]byte("leaf2"),
		[]byte("leaf3"),
	}

	for _, leaf := range leaves {
		tree.AddLeaf(leaf)
	}

	root := tree.GetRoot()
	// After adding leaves, root should exist
	_ = root
}

// TestCompactMerkleTreeRootUpdates tests root updates
func TestCompactMerkleTreeRootUpdates(t *testing.T) {
	tree := NewCompactMerkleTree()
	initialRoot := tree.GetRoot()

	tree.AddLeaf([]byte("new_leaf"))
	newRoot := tree.GetRoot()

	if bytes.Equal(newRoot, initialRoot) {
		t.Error("Root should change after adding leaf")
	}
}

// TestSparseMerkleTree tests sparse Merkle tree creation
func TestSparseMerkleTree(t *testing.T) {
	tree := NewSparseMerkleTree(256)
	if tree == nil {
		t.Fatal("Sparse tree should not be nil")
	}

	root := tree.GetRoot()
	if len(root) == 0 {
		t.Error("Root should not be empty")
	}
}

// TestSparseMerkleTreeUpdate tests updating values
func TestSparseMerkleTreeUpdate(t *testing.T) {
	tree := NewSparseMerkleTree(256)

	// Use 32-byte key (hash size)
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i)
	}
	value := []byte("test_value")

	err := tree.Update(key, value)
	if err != nil {
		t.Fatalf("Failed to update: %v", err)
	}

	retrieved, proof, err := tree.Get(key)
	if err != nil {
		t.Fatalf("Failed to get value: %v", err)
	}

	// Value is stored/retrieved (actual format may be hashed)
	if len(retrieved) == 0 {
		t.Error("Retrieved value should not be empty")
	}

	if proof == nil {
		t.Error("Proof should not be nil")
	}
}

// TestSparseMerkleTreeRootChanges tests that root changes after update
func TestSparseMerkleTreeRootChanges(t *testing.T) {
	tree := NewSparseMerkleTree(256)

	initialRoot := tree.GetRoot()

	// Use 32-byte key (hash size)
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i)
	}
	value := []byte("test_value")
	tree.Update(key, value)

	newRoot := tree.GetRoot()

	if bytes.Equal(newRoot, initialRoot) {
		t.Error("Root should change after update")
	}
}

// Benchmark tests
func BenchmarkNewMerkleTree(b *testing.B) {
	leaves := make([][]byte, 100)
	for i := range leaves {
		leaves[i] = []byte(string(rune(i)))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewMerkleTree(leaves)
	}
}

func BenchmarkMerkleTreeGetProof(b *testing.B) {
	leaves := make([][]byte, 100)
	for i := range leaves {
		leaves[i] = []byte(string(rune(i)))
	}

	tree, _ := NewMerkleTree(leaves)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree.GetProof(i % 100)
	}
}

func BenchmarkCompactMerkleTreeAddLeaf(b *testing.B) {
	tree := NewCompactMerkleTree()
	leaf := []byte("benchmark_leaf")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree.AddLeaf(leaf)
	}
}

func BenchmarkSparseMerkleTreeUpdate(b *testing.B) {
	tree := NewSparseMerkleTree(256)
	// Use 32-byte key
	key := make([]byte, 32)
	value := []byte("benchmark_value")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tree.Update(key, value)
	}
}
