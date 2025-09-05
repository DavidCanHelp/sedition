// Package crypto implements cryptographic primitives for PoC consensus
package crypto

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"

	"golang.org/x/crypto/curve25519"
)

// VRF implements Verifiable Random Functions using Ed25519
// Based on draft-irtf-cfrg-vrf-09
type VRF struct {
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey
}

// VRFOutput represents the output of a VRF computation
type VRFOutput struct {
	Value [32]byte // Random output
	Proof []byte   // Cryptographic proof
}

// NewVRF creates a new VRF instance with a generated key pair
func NewVRF() (*VRF, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate VRF keys: %w", err)
	}

	return &VRF{
		privateKey: priv,
		publicKey:  pub,
	}, nil
}

// NewVRFFromSeed creates a VRF instance from a seed
func NewVRFFromSeed(seed []byte) (*VRF, error) {
	if len(seed) < 32 {
		return nil, errors.New("seed must be at least 32 bytes")
	}

	// Derive private key from seed using HMAC-SHA512
	mac := hmac.New(sha512.New, []byte("VRF-SEED"))
	mac.Write(seed)
	keyMaterial := mac.Sum(nil)

	priv := ed25519.NewKeyFromSeed(keyMaterial[:32])
	pub := priv.Public().(ed25519.PublicKey)

	return &VRF{
		privateKey: priv,
		publicKey:  pub,
	}, nil
}

// Prove generates a VRF proof for the given message
func (v *VRF) Prove(message []byte) (*VRFOutput, error) {
	if v.privateKey == nil {
		return nil, errors.New("private key not set")
	}

	// Hash message to curve point
	h := v.hashToCurve(message)

	// Generate random nonce
	nonce := make([]byte, 32)
	if _, err := rand.Read(nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Compute gamma = h^sk
	gamma := v.scalarMult(h, v.privateKey[:32])

	// Create proof using Schnorr-like construction
	// c = H(g, h, g^sk, h^sk, g^k, h^k)
	// s = k - c*sk
	k := v.generateNonce(v.privateKey, h, nonce)
	gk := v.scalarBaseMult(k)
	hk := v.scalarMult(h, k)

	c := v.challengeHash(v.publicKey, h, gamma, gk, hk)
	s := v.scalarSub(k, v.scalarMult(c, v.privateKey[:32]))

	// Encode proof
	proof := v.encodeProof(gamma, c, s)

	// Compute output value
	output := v.proofToHash(gamma)

	return &VRFOutput{
		Value: output,
		Proof: proof,
	}, nil
}

// Verify checks a VRF proof and returns the output if valid
func (v *VRF) Verify(publicKey ed25519.PublicKey, message []byte, output *VRFOutput) (bool, error) {
	if len(output.Proof) < 80 {
		return false, errors.New("invalid proof length")
	}

	// Decode proof
	gamma, c, s := v.decodeProof(output.Proof)

	// Hash message to curve point
	h := v.hashToCurve(message)

	// Verify proof using public key
	// Check: g^s * y^c = g^k and h^s * gamma^c = h^k
	gs := v.scalarBaseMult(s)
	yc := v.scalarMult(publicKey[:32], c)
	gk := v.pointAdd(gs, yc)

	hs := v.scalarMult(h, s)
	gammac := v.scalarMult(gamma, c)
	hk := v.pointAdd(hs, gammac)

	// Recompute challenge
	cPrime := v.challengeHash(publicKey, h, gamma, gk, hk)

	// Verify challenge matches
	if !hmac.Equal(c, cPrime) {
		return false, nil
	}

	// Verify output matches
	expectedOutput := v.proofToHash(gamma)
	if !hmac.Equal(expectedOutput[:], output.Value[:]) {
		return false, nil
	}

	return true, nil
}

// GetPublicKey returns the VRF public key
func (v *VRF) GetPublicKey() ed25519.PublicKey {
	return v.publicKey
}

// GetRandomness extracts randomness from VRF output for leader election
func (v *VRFOutput) GetRandomness() *big.Int {
	return new(big.Int).SetBytes(v.Value[:])
}

// hashToCurve hashes a message to a curve point (simplified)
func (v *VRF) hashToCurve(message []byte) []byte {
	h := sha512.New()
	h.Write([]byte("VRF-HASH-TO-CURVE"))
	h.Write(message)
	hash := h.Sum(nil)

	// Reduce modulo curve order
	var point [32]byte
	copy(point[:], hash[:32])
	point[31] &= 127 // Clear high bit for valid curve point

	return point[:]
}

// scalarMult performs scalar multiplication on curve25519
func (v *VRF) scalarMult(point, scalar []byte) []byte {
	var p, s [32]byte
	copy(p[:], point[:32])
	copy(s[:], scalar[:32])

	var result [32]byte
	curve25519.ScalarMult(&result, &s, &p)
	return result[:]
}

// scalarBaseMult performs scalar multiplication with base point
func (v *VRF) scalarBaseMult(scalar []byte) []byte {
	var s [32]byte
	copy(s[:], scalar[:32])

	var result [32]byte
	curve25519.ScalarBaseMult(&result, &s)
	return result[:]
}

// pointAdd adds two curve points (simplified)
func (v *VRF) pointAdd(p1, p2 []byte) []byte {
	// Simplified point addition using XOR (not cryptographically correct)
	// In production, use proper elliptic curve addition
	result := make([]byte, 32)
	for i := 0; i < 32; i++ {
		result[i] = p1[i] ^ p2[i]
	}
	return result
}

// scalarSub performs scalar subtraction modulo curve order
func (v *VRF) scalarSub(a, b []byte) []byte {
	// Simplified scalar subtraction
	aInt := new(big.Int).SetBytes(a)
	bInt := new(big.Int).SetBytes(b)

	// Curve25519 order
	order := new(big.Int)
	order.SetString("7237005577332262213973186563042994240857116359379907606001950938285454250989", 10)

	result := new(big.Int).Sub(aInt, bInt)
	result.Mod(result, order)

	resultBytes := result.Bytes()
	if len(resultBytes) < 32 {
		padded := make([]byte, 32)
		copy(padded[32-len(resultBytes):], resultBytes)
		return padded
	}
	return resultBytes[:32]
}

// generateNonce generates a deterministic nonce
func (v *VRF) generateNonce(privateKey ed25519.PrivateKey, message, randomness []byte) []byte {
	h := sha512.New()
	h.Write([]byte("VRF-NONCE"))
	h.Write(privateKey[:32])
	h.Write(message)
	h.Write(randomness)
	hash := h.Sum(nil)
	return hash[:32]
}

// challengeHash computes the challenge hash
func (v *VRF) challengeHash(publicKey ed25519.PublicKey, h, gamma, gk, hk []byte) []byte {
	hash := sha256.New()
	hash.Write([]byte("VRF-CHALLENGE"))
	hash.Write(publicKey)
	hash.Write(h)
	hash.Write(gamma)
	hash.Write(gk)
	hash.Write(hk)
	return hash.Sum(nil)
}

// proofToHash converts proof to output hash
func (v *VRF) proofToHash(gamma []byte) [32]byte {
	h := sha256.New()
	h.Write([]byte("VRF-OUTPUT"))
	h.Write(gamma)
	sum := h.Sum(nil)

	var output [32]byte
	copy(output[:], sum)
	return output
}

// encodeProof encodes the proof components
func (v *VRF) encodeProof(gamma, c, s []byte) []byte {
	proof := make([]byte, 0, 96)
	proof = append(proof, gamma[:32]...)
	proof = append(proof, c[:32]...)
	proof = append(proof, s[:32]...)
	return proof
}

// decodeProof decodes the proof components
func (v *VRF) decodeProof(proof []byte) (gamma, c, s []byte) {
	gamma = proof[:32]
	c = proof[32:64]
	s = proof[64:96]
	return
}

// CompareVRFOutputs compares two VRF outputs for leader election
// Returns -1 if a < b, 0 if a == b, 1 if a > b
func CompareVRFOutputs(a, b *VRFOutput) int {
	aInt := a.GetRandomness()
	bInt := b.GetRandomness()
	return aInt.Cmp(bInt)
}

// VRFSortitionProof represents a proof of selection in sortition
type VRFSortitionProof struct {
	VRFOutput *VRFOutput
	J         uint64 // Number of times selected
	Stake     *big.Int
}

// Sortition performs cryptographic sortition using VRF
// Returns number of times selected and proof
func Sortition(vrf *VRF, seed []byte, round uint64, role string, stake, totalStake *big.Int, expectedSize uint64) (*VRFSortitionProof, error) {
	// Create sortition message
	message := make([]byte, 0, len(seed)+8+len(role))
	message = append(message, seed...)

	roundBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(roundBytes, round)
	message = append(message, roundBytes...)
	message = append(message, []byte(role)...)

	// Generate VRF output
	vrfOut, err := vrf.Prove(message)
	if err != nil {
		return nil, fmt.Errorf("VRF prove failed: %w", err)
	}

	// Calculate selection probability
	p := new(big.Float).SetInt(stake)
	p.Mul(p, new(big.Float).SetUint64(expectedSize))
	p.Quo(p, new(big.Float).SetInt(totalStake))

	// Use VRF output to determine selection
	hash := vrfOut.GetRandomness()
	maxHash := new(big.Int).Lsh(big.NewInt(1), 256)

	// Binomial sampling using VRF
	j := uint64(0)
	hashFloat := new(big.Float).SetInt(hash)
	maxFloat := new(big.Float).SetInt(maxHash)
	ratio := new(big.Float).Quo(hashFloat, maxFloat)

	// Check if selected (simplified - in production use proper binomial CDF)
	if ratio.Cmp(p) <= 0 {
		// Calculate number of selections (simplified)
		selections := new(big.Float).Quo(p, ratio)
		j64, _ := selections.Uint64()
		j = j64
		if j == 0 {
			j = 1
		}
		if j > expectedSize {
			j = expectedSize
		}
	}

	return &VRFSortitionProof{
		VRFOutput: vrfOut,
		J:         j,
		Stake:     stake,
	}, nil
}

// VerifySortition verifies a sortition proof
func VerifySortition(publicKey ed25519.PublicKey, seed []byte, round uint64, role string, proof *VRFSortitionProof, totalStake *big.Int, expectedSize uint64) (bool, error) {
	// Create sortition message
	message := make([]byte, 0, len(seed)+8+len(role))
	message = append(message, seed...)

	roundBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(roundBytes, round)
	message = append(message, roundBytes...)
	message = append(message, []byte(role)...)

	// Verify VRF proof
	vrf := &VRF{publicKey: publicKey}
	valid, err := vrf.Verify(publicKey, message, proof.VRFOutput)
	if err != nil || !valid {
		return false, err
	}

	// Verify selection count
	// (simplified - in production verify against binomial CDF)
	if proof.J > expectedSize {
		return false, errors.New("invalid selection count")
	}

	return true, nil
}
