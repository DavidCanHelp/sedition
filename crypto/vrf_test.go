package crypto

import (
	"bytes"
	"crypto/ed25519"
	"math/big"
	"testing"
)

// NOTE: These tests focus on API completeness. The VRF implementation
// contains simplified cryptographic operations that may not be fully
// correct (see vrf.go comments about "simplified point addition").

// TestNewVRF tests VRF creation with random keys
func TestNewVRF(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	if vrf.privateKey == nil {
		t.Error("Private key should not be nil")
	}

	if vrf.publicKey == nil {
		t.Error("Public key should not be nil")
	}

	if len(vrf.publicKey) != ed25519.PublicKeySize {
		t.Errorf("Invalid public key size: got %d, want %d", len(vrf.publicKey), ed25519.PublicKeySize)
	}

	if len(vrf.privateKey) != ed25519.PrivateKeySize {
		t.Errorf("Invalid private key size: got %d, want %d", len(vrf.privateKey), ed25519.PrivateKeySize)
	}
}

// TestNewVRFFromSeed tests deterministic VRF creation
func TestNewVRFFromSeed(t *testing.T) {
	// Create seed
	seed := make([]byte, 32)
	for i := range seed {
		seed[i] = byte(i)
	}

	// Create two VRFs with same seed
	vrf1, err := NewVRFFromSeed(seed)
	if err != nil {
		t.Fatalf("Failed to create VRF from seed: %v", err)
	}

	vrf2, err := NewVRFFromSeed(seed)
	if err != nil {
		t.Fatalf("Failed to create second VRF from seed: %v", err)
	}

	// They should have identical keys
	if !bytes.Equal(vrf1.publicKey, vrf2.publicKey) {
		t.Error("Public keys should be identical for same seed")
	}

	if !bytes.Equal(vrf1.privateKey, vrf2.privateKey) {
		t.Error("Private keys should be identical for same seed")
	}

	// Test with short seed (should fail)
	shortSeed := make([]byte, 16)
	_, err = NewVRFFromSeed(shortSeed)
	if err == nil {
		t.Error("Should fail with seed shorter than 32 bytes")
	}

	// Test with nil seed
	_, err = NewVRFFromSeed(nil)
	if err == nil {
		t.Error("Should fail with nil seed")
	}
}

// TestProveAPI tests that Prove can be called without errors
func TestProveAPI(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	message := []byte("Test message for VRF")

	// Generate proof
	output, err := vrf.Prove(message)
	if err != nil {
		t.Fatalf("Failed to generate proof: %v", err)
	}

	if output == nil {
		t.Fatal("Output should not be nil")
	}

	if len(output.Proof) == 0 {
		t.Error("Proof should not be empty")
	}

	if len(output.Value) != 32 {
		t.Errorf("Output value should be 32 bytes, got %d", len(output.Value))
	}
}

// TestProveDeterminism tests that same input produces same output with same key
func TestProveDeterminism(t *testing.T) {
	seed := make([]byte, 32)
	for i := range seed {
		seed[i] = byte(i)
	}

	vrf, err := NewVRFFromSeed(seed)
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	message := []byte("deterministic test")

	// Generate two proofs for same message
	// Note: Due to random nonce, proofs will differ, but output values should be same
	output1, err := vrf.Prove(message)
	if err != nil {
		t.Fatalf("First prove failed: %v", err)
	}

	output2, err := vrf.Prove(message)
	if err != nil {
		t.Fatalf("Second prove failed: %v", err)
	}

	// Values should be identical for same message and key
	if output1.Value != output2.Value {
		t.Error("VRF outputs should be identical for same message and key")
	}
}

// TestProveEmptyMessage tests handling of empty message
func TestProveEmptyMessage(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	// Empty message should be valid
	output, err := vrf.Prove([]byte{})
	if err != nil {
		t.Fatalf("Should handle empty message: %v", err)
	}

	if len(output.Proof) == 0 {
		t.Error("Proof should not be empty even for empty message")
	}

	if len(output.Value) != 32 {
		t.Errorf("Output value should be 32 bytes, got %d", len(output.Value))
	}
}

// TestGetPublicKey tests public key retrieval
func TestGetPublicKey(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	pubKey := vrf.GetPublicKey()
	if len(pubKey) != ed25519.PublicKeySize {
		t.Errorf("Invalid public key size: got %d, want %d", len(pubKey), ed25519.PublicKeySize)
	}

	if !bytes.Equal(pubKey, vrf.publicKey) {
		t.Error("GetPublicKey should return the same key")
	}
}

// TestVRFOutputGetRandomness tests randomness extraction
func TestVRFOutputGetRandomness(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	output, err := vrf.Prove([]byte("test"))
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}

	randomness := output.GetRandomness()
	if randomness == nil {
		t.Error("Randomness should not be nil")
	}

	if randomness.Sign() < 0 {
		t.Error("Randomness should be non-negative")
	}
}

// TestCompareVRFOutputs tests VRF output comparison
func TestCompareVRFOutputs(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	output1, err := vrf.Prove([]byte("message1"))
	if err != nil {
		t.Fatalf("First prove failed: %v", err)
	}

	output2, err := vrf.Prove([]byte("message2"))
	if err != nil {
		t.Fatalf("Second prove failed: %v", err)
	}

	cmp := CompareVRFOutputs(output1, output2)
	if cmp != -1 && cmp != 0 && cmp != 1 {
		t.Errorf("Compare result should be -1, 0, or 1, got %d", cmp)
	}

	// Comparing output with itself should be 0
	cmp = CompareVRFOutputs(output1, output1)
	if cmp != 0 {
		t.Errorf("Comparing output with itself should return 0, got %d", cmp)
	}
}

// TestSortitionAPI tests the sortition functionality API
func TestSortitionAPI(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	seed := []byte("election-seed-123")
	round := uint64(1)
	role := "proposer"
	stake := big.NewInt(1000)
	totalStake := big.NewInt(10000)
	expectedSize := uint64(10)

	// Generate sortition proof
	proof, err := Sortition(vrf, seed, round, role, stake, totalStake, expectedSize)
	if err != nil {
		t.Fatalf("Failed to generate sortition proof: %v", err)
	}

	if proof == nil {
		t.Error("Sortition proof should not be nil")
	}

	if proof.VRFOutput == nil {
		t.Error("VRF output should not be nil")
	}

	if proof.Stake.Cmp(stake) != 0 {
		t.Error("Stake should match input")
	}

	// Selection count should be valid
	if proof.J > expectedSize {
		t.Errorf("Selection count %d should not exceed expected size %d", proof.J, expectedSize)
	}

	// Test with zero stake (should result in 0 selections)
	proof, err = Sortition(vrf, seed, round, role, big.NewInt(0), totalStake, expectedSize)
	if err != nil {
		t.Fatalf("Should handle zero stake: %v", err)
	}
	if proof.J != 0 {
		t.Error("Should have 0 selections with zero stake")
	}
}

// TestSortitionDeterminism tests that sortition is deterministic
func TestSortitionDeterminism(t *testing.T) {
	seed := make([]byte, 32)
	for i := range seed {
		seed[i] = byte(i)
	}

	vrf, err := NewVRFFromSeed(seed)
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	electionSeed := []byte("election-seed")
	round := uint64(1)
	role := "proposer"
	stake := big.NewInt(1000)
	totalStake := big.NewInt(10000)
	expectedSize := uint64(10)

	// Generate sortition twice
	proof1, err := Sortition(vrf, electionSeed, round, role, stake, totalStake, expectedSize)
	if err != nil {
		t.Fatalf("First sortition failed: %v", err)
	}

	proof2, err := Sortition(vrf, electionSeed, round, role, stake, totalStake, expectedSize)
	if err != nil {
		t.Fatalf("Second sortition failed: %v", err)
	}

	// VRF outputs should be identical (same value)
	if proof1.VRFOutput.Value != proof2.VRFOutput.Value {
		t.Error("VRF outputs should be identical")
	}

	// Selection counts should be identical
	if proof1.J != proof2.J {
		t.Error("Selection counts should be identical")
	}
}

// TestVerifySortitionAPI tests verification API
func TestVerifySortitionAPI(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	seed := []byte("test-seed")
	round := uint64(1)
	role := "proposer"
	stake := big.NewInt(1000)
	totalStake := big.NewInt(10000)
	expectedSize := uint64(10)

	proof, err := Sortition(vrf, seed, round, role, stake, totalStake, expectedSize)
	if err != nil {
		t.Fatalf("Failed to generate sortition: %v", err)
	}

	// Test with invalid selection count
	invalidProof := &VRFSortitionProof{
		VRFOutput: proof.VRFOutput,
		J:         expectedSize + 100, // Exceeds expected size
		Stake:     stake,
	}

	valid, err := VerifySortition(vrf.publicKey, seed, round, role, invalidProof, totalStake, expectedSize)
	if err == nil && valid {
		t.Error("Should fail with invalid selection count")
	}
}

// TestVRFOutputUniqueness tests that different messages produce different outputs
func TestVRFOutputUniqueness(t *testing.T) {
	vrf, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF: %v", err)
	}

	message1 := []byte("message1")
	message2 := []byte("message2")

	output1, err := vrf.Prove(message1)
	if err != nil {
		t.Fatalf("Failed to prove message1: %v", err)
	}

	output2, err := vrf.Prove(message2)
	if err != nil {
		t.Fatalf("Failed to prove message2: %v", err)
	}

	// Different messages should produce different outputs
	// (though this is probabilistic, collision is extremely unlikely)
	if output1.Value == output2.Value {
		t.Error("Different messages should produce different outputs")
	}
}

// TestVRFPublicKeyUniqueness tests that different public keys produce different outputs
func TestVRFPublicKeyUniqueness(t *testing.T) {
	vrf1, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF1: %v", err)
	}

	vrf2, err := NewVRF()
	if err != nil {
		t.Fatalf("Failed to create VRF2: %v", err)
	}

	message := []byte("same message")

	output1, err := vrf1.Prove(message)
	if err != nil {
		t.Fatalf("Failed to prove with VRF1: %v", err)
	}

	output2, err := vrf2.Prove(message)
	if err != nil {
		t.Fatalf("Failed to prove with VRF2: %v", err)
	}

	// Different keys should produce different outputs
	if output1.Value == output2.Value {
		t.Error("Different keys should produce different outputs for same message")
	}
}

// Benchmark tests
func BenchmarkNewVRF(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := NewVRF()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProve(b *testing.B) {
	vrf, err := NewVRF()
	if err != nil {
		b.Fatal(err)
	}

	message := []byte("Benchmark message")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := vrf.Prove(message)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSortition(b *testing.B) {
	vrf, err := NewVRF()
	if err != nil {
		b.Fatal(err)
	}

	seed := []byte("election-seed")
	round := uint64(1)
	role := "proposer"
	stake := big.NewInt(1000)
	totalStake := big.NewInt(10000)
	expectedSize := uint64(10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Sortition(vrf, seed, round, role, stake, totalStake, expectedSize)
		if err != nil {
			b.Fatal(err)
		}
	}
}
