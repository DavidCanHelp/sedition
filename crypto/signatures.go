package crypto

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
)

// Signer handles digital signatures for consensus messages
type Signer struct {
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey
	address    string // Derived from public key
}

// NewSigner creates a new signer with generated keys
func NewSigner() (*Signer, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate keys: %w", err)
	}
	
	return &Signer{
		privateKey: priv,
		publicKey:  pub,
		address:    DeriveAddress(pub),
	}, nil
}

// NewSignerFromSeed creates a signer from a seed
func NewSignerFromSeed(seed []byte) (*Signer, error) {
	if len(seed) < 32 {
		return nil, errors.New("seed must be at least 32 bytes")
	}
	
	priv := ed25519.NewKeyFromSeed(seed[:32])
	pub := priv.Public().(ed25519.PublicKey)
	
	return &Signer{
		privateKey: priv,
		publicKey:  pub,
		address:    DeriveAddress(pub),
	}, nil
}

// Sign creates a signature for the given message
func (s *Signer) Sign(message []byte) ([]byte, error) {
	if s.privateKey == nil {
		return nil, errors.New("private key not set")
	}
	
	signature := ed25519.Sign(s.privateKey, message)
	return signature, nil
}

// Verify checks a signature against a message
func (s *Signer) Verify(publicKey ed25519.PublicKey, message, signature []byte) bool {
	return ed25519.Verify(publicKey, message, signature)
}

// GetPublicKey returns the signer's public key
func (s *Signer) GetPublicKey() ed25519.PublicKey {
	return s.publicKey
}

// GetAddress returns the signer's address
func (s *Signer) GetAddress() string {
	return s.address
}

// DeriveAddress derives an address from a public key
func DeriveAddress(publicKey ed25519.PublicKey) string {
	hash := sha256.Sum256(publicKey)
	// Take first 20 bytes like Ethereum
	return "0x" + hex.EncodeToString(hash[:20])
}

// MultiSignature represents a threshold signature scheme
type MultiSignature struct {
	Signatures map[string][]byte // address -> signature
	Message    []byte
	Threshold  int
}

// NewMultiSignature creates a new multi-signature collector
func NewMultiSignature(message []byte, threshold int) *MultiSignature {
	return &MultiSignature{
		Signatures: make(map[string][]byte),
		Message:    message,
		Threshold:  threshold,
	}
}

// AddSignature adds a signature to the collection
func (m *MultiSignature) AddSignature(signer *Signer) error {
	sig, err := signer.Sign(m.Message)
	if err != nil {
		return err
	}
	
	m.Signatures[signer.GetAddress()] = sig
	return nil
}

// Verify checks if we have enough valid signatures
func (m *MultiSignature) Verify(signers map[string]ed25519.PublicKey) bool {
	validCount := 0
	
	for address, sig := range m.Signatures {
		pubKey, exists := signers[address]
		if !exists {
			continue
		}
		
		if ed25519.Verify(pubKey, m.Message, sig) {
			validCount++
		}
	}
	
	return validCount >= m.Threshold
}

// HasSufficientSignatures checks if threshold is met
func (m *MultiSignature) HasSufficientSignatures() bool {
	return len(m.Signatures) >= m.Threshold
}

// GenerateRandom generates random bytes
func GenerateRandom(b []byte) (int, error) {
	return io.ReadFull(rand.Reader, b)
}

// SignedMessage represents a message with its signature
type SignedMessage struct {
	Message   []byte
	Signature []byte
	PublicKey ed25519.PublicKey
	Address   string
}

// NewSignedMessage creates and signs a message
func NewSignedMessage(signer *Signer, message []byte) (*SignedMessage, error) {
	sig, err := signer.Sign(message)
	if err != nil {
		return nil, err
	}
	
	return &SignedMessage{
		Message:   message,
		Signature: sig,
		PublicKey: signer.GetPublicKey(),
		Address:   signer.GetAddress(),
	}, nil
}

// Verify checks the signature on a signed message
func (sm *SignedMessage) Verify() bool {
	return ed25519.Verify(sm.PublicKey, sm.Message, sm.Signature)
}

// AggregateSignature represents BLS-style aggregate signatures (simplified)
type AggregateSignature struct {
	Signers    []ed25519.PublicKey
	Signatures [][]byte
	Message    []byte
}

// NewAggregateSignature creates a new aggregate signature
func NewAggregateSignature(message []byte) *AggregateSignature {
	return &AggregateSignature{
		Message:    message,
		Signers:    make([]ed25519.PublicKey, 0),
		Signatures: make([][]byte, 0),
	}
}

// AddSigner adds a signer to the aggregate
func (a *AggregateSignature) AddSigner(signer *Signer) error {
	sig, err := signer.Sign(a.Message)
	if err != nil {
		return err
	}
	
	a.Signers = append(a.Signers, signer.GetPublicKey())
	a.Signatures = append(a.Signatures, sig)
	return nil
}

// Verify checks all signatures in the aggregate
func (a *AggregateSignature) Verify() bool {
	if len(a.Signers) != len(a.Signatures) {
		return false
	}
	
	for i, pubKey := range a.Signers {
		if !ed25519.Verify(pubKey, a.Message, a.Signatures[i]) {
			return false
		}
	}
	
	return true
}

// Size returns the number of signatures
func (a *AggregateSignature) Size() int {
	return len(a.Signatures)
}

// SignatureShare represents a threshold signature share
type SignatureShare struct {
	Index     int
	Share     []byte
	PublicKey ed25519.PublicKey
}

// ThresholdSigner implements threshold signatures (simplified)
type ThresholdSigner struct {
	threshold int
	shares    map[int]*SignatureShare
	message   []byte
}

// NewThresholdSigner creates a threshold signer
func NewThresholdSigner(threshold int, message []byte) *ThresholdSigner {
	return &ThresholdSigner{
		threshold: threshold,
		shares:    make(map[int]*SignatureShare),
		message:   message,
	}
}

// AddShare adds a signature share
func (t *ThresholdSigner) AddShare(index int, signer *Signer) error {
	sig, err := signer.Sign(t.message)
	if err != nil {
		return err
	}
	
	t.shares[index] = &SignatureShare{
		Index:     index,
		Share:     sig,
		PublicKey: signer.GetPublicKey(),
	}
	
	return nil
}

// HasThreshold checks if we have enough shares
func (t *ThresholdSigner) HasThreshold() bool {
	return len(t.shares) >= t.threshold
}

// CombineShares combines shares into final signature (simplified)
// In production, use proper threshold signature scheme like BLS
func (t *ThresholdSigner) CombineShares() ([]byte, error) {
	if !t.HasThreshold() {
		return nil, errors.New("insufficient shares")
	}
	
	// Simplified: just concatenate first threshold signatures
	// Real implementation would use Shamir secret sharing or BLS
	combined := make([]byte, 0)
	count := 0
	for _, share := range t.shares {
		if count >= t.threshold {
			break
		}
		combined = append(combined, share.Share...)
		count++
	}
	
	return combined, nil
}

// BlindSignature implements blind signatures (simplified)
type BlindSignature struct {
	signer *Signer
}

// NewBlindSignature creates a blind signature scheme
func NewBlindSignature(signer *Signer) *BlindSignature {
	return &BlindSignature{
		signer: signer,
	}
}

// Blind blinds a message (simplified - use RSA or proper blind signature in production)
func (b *BlindSignature) Blind(message []byte, blindingFactor []byte) []byte {
	blinded := make([]byte, len(message))
	for i := range message {
		if i < len(blindingFactor) {
			blinded[i] = message[i] ^ blindingFactor[i]
		} else {
			blinded[i] = message[i]
		}
	}
	return blinded
}

// SignBlinded signs a blinded message
func (b *BlindSignature) SignBlinded(blindedMessage []byte) ([]byte, error) {
	return b.signer.Sign(blindedMessage)
}

// Unblind removes blinding from signature
func (b *BlindSignature) Unblind(blindedSig []byte, blindingFactor []byte) []byte {
	// Simplified - real implementation needs proper unblinding
	return blindedSig
}

// RingSignature implements ring signatures (simplified)
type RingSignature struct {
	ring    []ed25519.PublicKey
	message []byte
}

// NewRingSignature creates a ring signature
func NewRingSignature(ring []ed25519.PublicKey, message []byte) *RingSignature {
	return &RingSignature{
		ring:    ring,
		message: message,
	}
}

// Sign creates a ring signature (simplified)
func (r *RingSignature) Sign(signer *Signer) ([]byte, error) {
	// Simplified - real implementation needs proper ring signature
	// For now, just sign with included public key proof
	sig, err := signer.Sign(r.message)
	if err != nil {
		return nil, err
	}
	
	// Include ring position (simplified)
	for i, pubKey := range r.ring {
		if string(pubKey) == string(signer.GetPublicKey()) {
			sig = append([]byte{byte(i)}, sig...)
			break
		}
	}
	
	return sig, nil
}

// Verify checks a ring signature
func (r *RingSignature) Verify(signature []byte) bool {
	if len(signature) < 65 {
		return false
	}
	
	// Extract ring position
	position := int(signature[0])
	if position >= len(r.ring) {
		return false
	}
	
	// Verify with public key at position
	actualSig := signature[1:]
	return ed25519.Verify(r.ring[position], r.message, actualSig)
}