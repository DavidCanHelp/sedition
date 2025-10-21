package crypto

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

// TestNewSigner tests signer creation with random keys
func TestNewSigner(t *testing.T) {
	signer, err := NewSigner()
	if err != nil {
		t.Fatalf("Failed to create signer: %v", err)
	}

	if signer.privateKey == nil {
		t.Error("Private key should not be nil")
	}

	if signer.publicKey == nil {
		t.Error("Public key should not be nil")
	}

	if signer.address == "" {
		t.Error("Address should not be empty")
	}

	// Address should start with 0x
	if len(signer.address) != 42 || signer.address[:2] != "0x" {
		t.Errorf("Invalid address format: %s", signer.address)
	}
}

// TestNewSignerFromSeed tests deterministic signer creation from seed
func TestNewSignerFromSeed(t *testing.T) {
	// Create seed
	seed := make([]byte, 32)
	for i := range seed {
		seed[i] = byte(i)
	}

	// Create two signers with same seed
	signer1, err := NewSignerFromSeed(seed)
	if err != nil {
		t.Fatalf("Failed to create signer from seed: %v", err)
	}

	signer2, err := NewSignerFromSeed(seed)
	if err != nil {
		t.Fatalf("Failed to create second signer from seed: %v", err)
	}

	// They should have identical keys and addresses
	if !bytes.Equal(signer1.publicKey, signer2.publicKey) {
		t.Error("Public keys should be identical for same seed")
	}

	if signer1.address != signer2.address {
		t.Error("Addresses should be identical for same seed")
	}

	// Test with short seed (should fail)
	shortSeed := make([]byte, 16)
	_, err = NewSignerFromSeed(shortSeed)
	if err == nil {
		t.Error("Should fail with seed shorter than 32 bytes")
	}
}

// TestSignAndVerify tests basic signing and verification
func TestSignAndVerify(t *testing.T) {
	signer, err := NewSigner()
	if err != nil {
		t.Fatalf("Failed to create signer: %v", err)
	}

	message := []byte("Hello, blockchain!")

	// Sign the message
	signature, err := signer.Sign(message)
	if err != nil {
		t.Fatalf("Failed to sign message: %v", err)
	}

	if len(signature) != ed25519.SignatureSize {
		t.Errorf("Invalid signature size: got %d, want %d", len(signature), ed25519.SignatureSize)
	}

	// Verify the signature
	if !signer.Verify(signer.publicKey, message, signature) {
		t.Error("Signature verification failed")
	}

	// Verify with wrong message should fail
	wrongMessage := []byte("Wrong message")
	if signer.Verify(signer.publicKey, wrongMessage, signature) {
		t.Error("Verification should fail with wrong message")
	}

	// Verify with wrong public key should fail
	otherSigner, _ := NewSigner()
	if signer.Verify(otherSigner.publicKey, message, signature) {
		t.Error("Verification should fail with wrong public key")
	}

	// Verify with corrupted signature should fail
	corruptedSig := make([]byte, len(signature))
	copy(corruptedSig, signature)
	corruptedSig[0] ^= 0xFF
	if signer.Verify(signer.publicKey, message, corruptedSig) {
		t.Error("Verification should fail with corrupted signature")
	}
}

// TestDeriveAddress tests address derivation
func TestDeriveAddress(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("Failed to generate key: %v", err)
	}

	address := DeriveAddress(pub)

	// Check format
	if len(address) != 42 {
		t.Errorf("Invalid address length: got %d, want 42", len(address))
	}

	if address[:2] != "0x" {
		t.Error("Address should start with 0x")
	}

	// Same public key should always give same address
	address2 := DeriveAddress(pub)
	if address != address2 {
		t.Error("Address should be deterministic")
	}
}

// TestMultiSignature tests multi-signature functionality
func TestMultiSignature(t *testing.T) {
	message := []byte("Multi-sig test")
	threshold := 3

	multiSig := NewMultiSignature(message, threshold)

	if multiSig.Threshold != threshold {
		t.Errorf("Threshold mismatch: got %d, want %d", multiSig.Threshold, threshold)
	}

	// Create signers and add signatures
	signers := make([]*Signer, 5)
	signerKeys := make(map[string]ed25519.PublicKey)

	for i := 0; i < 5; i++ {
		signer, err := NewSigner()
		if err != nil {
			t.Fatalf("Failed to create signer %d: %v", i, err)
		}
		signers[i] = signer
		signerKeys[signer.address] = signer.publicKey

		err = multiSig.AddSignature(signer)
		if err != nil {
			t.Fatalf("Failed to add signature %d: %v", i, err)
		}
	}

	// Verify we have 5 signatures
	if len(multiSig.Signatures) != 5 {
		t.Errorf("Expected 5 signatures, got %d", len(multiSig.Signatures))
	}

	// Verify multi-signature
	if !multiSig.Verify(signerKeys) {
		t.Error("Multi-signature verification failed")
	}

	// Test with insufficient threshold
	lowThresholdSig := NewMultiSignature(message, 10)
	for i := 0; i < 5; i++ {
		lowThresholdSig.AddSignature(signers[i])
	}

	// Should fail because we need 10 but only have 5
	if lowThresholdSig.Verify(signerKeys) {
		t.Error("Verification should fail when threshold not met")
	}

	// Test HasSufficientSignatures
	if !multiSig.HasSufficientSignatures() {
		t.Error("Should have sufficient signatures (5 >= 3)")
	}

	if lowThresholdSig.HasSufficientSignatures() {
		t.Error("Should not have sufficient signatures (5 < 10)")
	}
}

// TestSignedMessage tests signed message functionality
func TestSignedMessage(t *testing.T) {
	signer, err := NewSigner()
	if err != nil {
		t.Fatalf("Failed to create signer: %v", err)
	}

	message := []byte("Signed message test")

	signedMsg, err := NewSignedMessage(signer, message)
	if err != nil {
		t.Fatalf("Failed to create signed message: %v", err)
	}

	// Verify message matches
	if !bytes.Equal(signedMsg.Message, message) {
		t.Error("Message mismatch")
	}

	// Verify address matches
	if signedMsg.Address != signer.address {
		t.Error("Address mismatch")
	}

	// Verify public key matches
	if !bytes.Equal(signedMsg.PublicKey, signer.publicKey) {
		t.Error("Public key mismatch")
	}

	// Verify signature
	if !signedMsg.Verify() {
		t.Error("Signed message verification failed")
	}

	// Test with tampered message
	signedMsg.Message = []byte("Tampered message")
	if signedMsg.Verify() {
		t.Error("Verification should fail with tampered message")
	}
}

// TestAggregateSignature tests aggregate signature functionality
func TestAggregateSignature(t *testing.T) {
	message := []byte("Aggregate signature test")
	aggSig := NewAggregateSignature(message)

	// Add multiple signers
	numSigners := 5
	for i := 0; i < numSigners; i++ {
		signer, err := NewSigner()
		if err != nil {
			t.Fatalf("Failed to create signer %d: %v", i, err)
		}

		err = aggSig.AddSigner(signer)
		if err != nil {
			t.Fatalf("Failed to add signer %d: %v", i, err)
		}
	}

	// Verify size
	if aggSig.Size() != numSigners {
		t.Errorf("Size mismatch: got %d, want %d", aggSig.Size(), numSigners)
	}

	// Verify aggregate signature
	if !aggSig.Verify() {
		t.Error("Aggregate signature verification failed")
	}

	// Corrupt one signature
	aggSig.Signatures[0][0] ^= 0xFF

	// Verification should fail
	if aggSig.Verify() {
		t.Error("Verification should fail with corrupted signature")
	}
}

// TestThresholdSigner tests threshold signature functionality
func TestThresholdSigner(t *testing.T) {
	message := []byte("Threshold signature test")
	threshold := 3

	thresholdSigner := NewThresholdSigner(threshold, message)

	// Add signatures from multiple signers
	for i := 0; i < 5; i++ {
		signer, err := NewSigner()
		if err != nil {
			t.Fatalf("Failed to create signer %d: %v", i, err)
		}

		err = thresholdSigner.AddShare(i, signer)
		if err != nil {
			t.Fatalf("Failed to add share %d: %v", i, err)
		}
	}

	// Should have threshold
	if !thresholdSigner.HasThreshold() {
		t.Error("Should have met threshold (5 >= 3)")
	}

	// Combine shares
	combined, err := thresholdSigner.CombineShares()
	if err != nil {
		t.Fatalf("Failed to combine shares: %v", err)
	}

	if len(combined) == 0 {
		t.Error("Combined signature should not be empty")
	}

	// Test with insufficient shares
	lowThreshold := NewThresholdSigner(10, message)
	for i := 0; i < 5; i++ {
		signer, _ := NewSigner()
		lowThreshold.AddShare(i, signer)
	}

	if lowThreshold.HasThreshold() {
		t.Error("Should not have met threshold (5 < 10)")
	}

	_, err = lowThreshold.CombineShares()
	if err == nil {
		t.Error("Should fail to combine with insufficient shares")
	}
}

// TestBlindSignature tests blind signature functionality
func TestBlindSignature(t *testing.T) {
	signer, err := NewSigner()
	if err != nil {
		t.Fatalf("Failed to create signer: %v", err)
	}

	blindSig := NewBlindSignature(signer)

	message := []byte("Blind signature test message")
	blindingFactor := make([]byte, len(message))
	rand.Read(blindingFactor)

	// Blind the message
	blinded := blindSig.Blind(message, blindingFactor)

	// Blinded message should be different
	if bytes.Equal(message, blinded) {
		t.Error("Blinded message should differ from original")
	}

	// Sign blinded message
	signature, err := blindSig.SignBlinded(blinded)
	if err != nil {
		t.Fatalf("Failed to sign blinded message: %v", err)
	}

	// Unblind signature
	unblinded := blindSig.Unblind(signature, blindingFactor)

	// Unblinded signature should exist
	if len(unblinded) == 0 {
		t.Error("Unblinded signature should not be empty")
	}
}

// TestRingSignature tests ring signature functionality
func TestRingSignature(t *testing.T) {
	message := []byte("Ring signature test")

	// Create a ring of public keys
	ringSize := 5
	ring := make([]ed25519.PublicKey, ringSize)
	signers := make([]*Signer, ringSize)

	for i := 0; i < ringSize; i++ {
		signer, err := NewSigner()
		if err != nil {
			t.Fatalf("Failed to create signer %d: %v", i, err)
		}
		signers[i] = signer
		ring[i] = signer.publicKey
	}

	// Create ring signature
	ringSig := NewRingSignature(ring, message)

	// Sign with one of the ring members (index 2)
	signature, err := ringSig.Sign(signers[2])
	if err != nil {
		t.Fatalf("Failed to create ring signature: %v", err)
	}

	// Verify ring signature
	if !ringSig.Verify(signature) {
		t.Error("Ring signature verification failed")
	}

	// Test with corrupted signature
	corruptedSig := make([]byte, len(signature))
	copy(corruptedSig, signature)
	if len(corruptedSig) > 1 {
		corruptedSig[1] ^= 0xFF
	}

	if ringSig.Verify(corruptedSig) {
		t.Error("Verification should fail with corrupted signature")
	}

	// Test with invalid position
	if len(signature) > 0 {
		invalidPosSig := make([]byte, len(signature))
		copy(invalidPosSig, signature)
		invalidPosSig[0] = byte(ringSize + 1) // Invalid position

		if ringSig.Verify(invalidPosSig) {
			t.Error("Verification should fail with invalid ring position")
		}
	}
}

// TestGenerateRandom tests random byte generation
func TestGenerateRandom(t *testing.T) {
	// Test various sizes
	sizes := []int{16, 32, 64, 128}

	for _, size := range sizes {
		b := make([]byte, size)
		n, err := GenerateRandom(b)

		if err != nil {
			t.Errorf("Failed to generate %d random bytes: %v", size, err)
		}

		if n != size {
			t.Errorf("Generated %d bytes, wanted %d", n, size)
		}

		// Check that bytes are not all zeros (very unlikely with true randomness)
		allZeros := true
		for _, v := range b {
			if v != 0 {
				allZeros = false
				break
			}
		}

		if allZeros {
			t.Error("Generated bytes are all zeros (extremely unlikely)")
		}
	}
}

// Benchmark tests
func BenchmarkNewSigner(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := NewSigner()
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSign(b *testing.B) {
	signer, _ := NewSigner()
	message := []byte("Benchmark message")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := signer.Sign(message)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVerify(b *testing.B) {
	signer, _ := NewSigner()
	message := []byte("Benchmark message")
	signature, _ := signer.Sign(message)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !signer.Verify(signer.publicKey, message, signature) {
			b.Fatal("Verification failed")
		}
	}
}
