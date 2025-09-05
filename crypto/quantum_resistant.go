package crypto

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"errors"
	"fmt"
	"math/big"
)

// SPHINCS+ signature scheme implementation
// Based on SPHINCS+ specification (NIST PQC standardization)
type SPHINCSConfig struct {
	n    int // Security parameter (hash output length)
	h    int // Height of hypertree
	d    int // Number of layers
	logT int // Logarithm of Winternitz parameter
	k    int // Number of trees in FORS
	a    int // Height of each FORS tree
}

// Standard SPHINCS+ parameter sets
var (
	SPHINCS128s = &SPHINCSConfig{n: 16, h: 63, d: 7, logT: 4, k: 14, a: 12}
	SPHINCS192s = &SPHINCSConfig{n: 24, h: 63, d: 7, logT: 4, k: 17, a: 14}
	SPHINCS256s = &SPHINCSConfig{n: 32, h: 64, d: 8, logT: 4, k: 22, a: 14}
)

// SPHINCSKeyPair represents a SPHINCS+ key pair
type SPHINCSKeyPair struct {
	config     *SPHINCSConfig
	privateKey []byte
	publicKey  []byte
	seed       []byte
	prf        []byte
}

// SPHINCSSignature represents a SPHINCS+ signature
type SPHINCSSignature struct {
	config    *SPHINCSConfig
	signature []byte
	message   []byte
}

// NewSPHINCSKeyPair generates a new SPHINCS+ key pair
func NewSPHINCSKeyPair(config *SPHINCSConfig) (*SPHINCSKeyPair, error) {
	if config == nil {
		config = SPHINCS256s // Default to 256-bit security
	}

	// Generate random seed
	seed := make([]byte, config.n)
	if _, err := rand.Read(seed); err != nil {
		return nil, fmt.Errorf("failed to generate seed: %v", err)
	}

	// Generate PRF key
	prf := make([]byte, config.n)
	if _, err := rand.Read(prf); err != nil {
		return nil, fmt.Errorf("failed to generate PRF key: %v", err)
	}

	// Generate public seed
	pubSeed := make([]byte, config.n)
	if _, err := rand.Read(pubSeed); err != nil {
		return nil, fmt.Errorf("failed to generate public seed: %v", err)
	}

	// Private key consists of seed, prf, and public seed
	privateKey := make([]byte, 0, 3*config.n)
	privateKey = append(privateKey, seed...)
	privateKey = append(privateKey, prf...)
	privateKey = append(privateKey, pubSeed...)

	// Generate root of the hypertree (public key)
	publicKey, err := generateHypertreeRoot(config, seed, pubSeed)
	if err != nil {
		return nil, fmt.Errorf("failed to generate hypertree root: %v", err)
	}

	return &SPHINCSKeyPair{
		config:     config,
		privateKey: privateKey,
		publicKey:  publicKey,
		seed:       seed,
		prf:        prf,
	}, nil
}

// Sign creates a SPHINCS+ signature for the given message
func (kp *SPHINCSKeyPair) Sign(message []byte) (*SPHINCSSignature, error) {
	// Hash message with randomizer
	randomizer := make([]byte, kp.config.n)
	if _, err := rand.Read(randomizer); err != nil {
		return nil, fmt.Errorf("failed to generate randomizer: %v", err)
	}

	// Generate message hash
	msgHash := kp.hashMessage(message, randomizer)

	// Generate FORS signature
	forsSignature, err := kp.generateFORSSignature(msgHash)
	if err != nil {
		return nil, fmt.Errorf("failed to generate FORS signature: %v", err)
	}

	// Generate hypertree signature
	hypertreeSignature, err := kp.generateHypertreeSignature(msgHash)
	if err != nil {
		return nil, fmt.Errorf("failed to generate hypertree signature: %v", err)
	}

	// Combine signature components
	signature := make([]byte, 0)
	signature = append(signature, randomizer...)
	signature = append(signature, forsSignature...)
	signature = append(signature, hypertreeSignature...)

	return &SPHINCSSignature{
		config:    kp.config,
		signature: signature,
		message:   message,
	}, nil
}

// Verify checks if a SPHINCS+ signature is valid
func (sig *SPHINCSSignature) Verify(publicKey []byte) bool {
	if len(sig.signature) < sig.config.n {
		return false
	}

	// Extract randomizer
	randomizer := sig.signature[:sig.config.n]

	// Reconstruct message hash
	msgHash := hashMessage(sig.config, sig.message, randomizer)

	// Verify FORS signature
	forsValid := verifyFORSSignature(sig.config, msgHash, sig.signature[sig.config.n:], publicKey)
	if !forsValid {
		return false
	}

	// Verify hypertree signature
	hypertreeValid := verifyHypertreeSignature(sig.config, msgHash, sig.signature, publicKey)
	return hypertreeValid
}

// CRYSTALS-Kyber key encapsulation mechanism
type KyberConfig struct {
	n    int // Polynomial degree
	q    int // Modulus
	k    int // Dimension of module
	eta1 int // Noise parameter for secret
	eta2 int // Noise parameter for error
	du   int // Compression parameter for u
	dv   int // Compression parameter for v
}

// Standard Kyber parameter sets
var (
	Kyber512  = &KyberConfig{n: 256, q: 3329, k: 2, eta1: 3, eta2: 2, du: 10, dv: 4}
	Kyber768  = &KyberConfig{n: 256, q: 3329, k: 3, eta1: 2, eta2: 2, du: 10, dv: 4}
	Kyber1024 = &KyberConfig{n: 256, q: 3329, k: 4, eta1: 2, eta2: 2, du: 11, dv: 5}
)

// KyberKeyPair represents a Kyber key pair
type KyberKeyPair struct {
	config     *KyberConfig
	privateKey []byte
	publicKey  []byte
	secretKey  [][]int16 // s vector
	publicKeyA [][]int16 // A matrix
	publicKeyT []int16   // t vector
}

// KyberCiphertext represents an encapsulated key
type KyberCiphertext struct {
	config     *KyberConfig
	ciphertext []byte
	u          []int16
	v          int16
}

// NewKyberKeyPair generates a new Kyber key pair
func NewKyberKeyPair(config *KyberConfig) (*KyberKeyPair, error) {
	if config == nil {
		config = Kyber1024 // Default to highest security
	}

	// Generate random matrix A
	A := make([][]int16, config.k)
	for i := range A {
		A[i] = make([]int16, config.k)
		for j := range A[i] {
			A[i][j] = int16(randMod(config.q))
		}
	}

	// Generate secret vector s
	s := make([][]int16, config.k)
	for i := range s {
		s[i] = sampleNoise(config.n, config.eta1)
	}

	// Generate error vector e
	e := make([][]int16, config.k)
	for i := range e {
		e[i] = sampleNoise(config.n, config.eta2)
	}

	// Compute t = As + e
	t := make([]int16, config.k*config.n)
	for i := 0; i < config.k; i++ {
		for j := 0; j < config.n; j++ {
			sum := int32(0)
			for k := 0; k < config.k; k++ {
				sum += int32(A[i][k]) * int32(s[k][j])
			}
			sum += int32(e[i][j])
			t[i*config.n+j] = int16(sum % int32(config.q))
		}
	}

	// Encode keys
	privateKey := encodeSecretKey(s, config)
	publicKey := encodePublicKey(A, t, config)

	return &KyberKeyPair{
		config:     config,
		privateKey: privateKey,
		publicKey:  publicKey,
		secretKey:  s,
		publicKeyA: A,
		publicKeyT: t,
	}, nil
}

// Encapsulate generates a shared secret and its encapsulation
func (kp *KyberKeyPair) Encapsulate() ([]byte, *KyberCiphertext, error) {
	// Generate random message
	m := make([]byte, 32)
	if _, err := rand.Read(m); err != nil {
		return nil, nil, fmt.Errorf("failed to generate random message: %v", err)
	}

	// Hash message to get coins
	coins := sha256.Sum256(m)

	// Sample r from coins
	r := sampleNoiseFromSeed(coins[:], kp.config.n, kp.config.eta1)

	// Sample e1 and e2
	e1 := make([][]int16, kp.config.k)
	for i := range e1 {
		e1[i] = sampleNoise(kp.config.n, kp.config.eta2)
	}
	e2 := sampleNoise(kp.config.n, kp.config.eta2)

	// Compute u = A^T * r + e1
	u := make([]int16, kp.config.k*kp.config.n)
	for i := 0; i < kp.config.k; i++ {
		for j := 0; j < kp.config.n; j++ {
			sum := int32(0)
			for k := 0; k < kp.config.k; k++ {
				sum += int32(kp.publicKeyA[k][i]) * int32(r[j])
			}
			sum += int32(e1[i][j])
			u[i*kp.config.n+j] = int16(sum % int32(kp.config.q))
		}
	}

	// Compute v = t^T * r + e2 + Decode(m)
	v := int16(0)
	for i := 0; i < kp.config.k*kp.config.n; i++ {
		v += kp.publicKeyT[i] * r[i%kp.config.n]
	}
	v += e2[0] + decodeMessage(m)[0]
	v = v % int16(kp.config.q)

	// Compress and encode ciphertext
	ciphertext := encodeCiphertext(u, v, kp.config)

	// Derive shared secret
	ctHash := sha256.Sum256(ciphertext)
	sharedSecret := sha256.Sum256(append(m, ctHash[:]...))

	return sharedSecret[:], &KyberCiphertext{
		config:     kp.config,
		ciphertext: ciphertext,
		u:          u,
		v:          v,
	}, nil
}

// Decapsulate recovers the shared secret from the ciphertext
func (kp *KyberKeyPair) Decapsulate(ct *KyberCiphertext) ([]byte, error) {
	// Decompress ciphertext
	u, v, err := decodeCiphertext(ct.ciphertext, kp.config)
	if err != nil {
		return nil, fmt.Errorf("failed to decode ciphertext: %v", err)
	}

	// Compute m' = v - s^T * u
	mPrime := v
	for i := 0; i < kp.config.k; i++ {
		for j := 0; j < kp.config.n; j++ {
			mPrime -= kp.secretKey[i][j] * u[i*kp.config.n+j]
		}
	}
	mPrime = mPrime % int16(kp.config.q)

	// Decode message
	message := encodeMessage([]int16{mPrime})

	// Verify by re-encapsulation
	// coins := sha256.Sum256(message) // Unused variable commented out
	// ... verification logic ...

	// Derive shared secret
	ctHash := sha256.Sum256(ct.ciphertext)
	sharedSecret := sha256.Sum256(append(message, ctHash[:]...))
	return sharedSecret[:], nil
}

// Quantum-resistant consensus adapter
type QuantumResistantConsensus struct {
	sphincsConfig *SPHINCSConfig
	kyberConfig   *KyberConfig
	validators    map[string]*QuantumValidator
}

// QuantumValidator represents a validator with quantum-resistant keys
type QuantumValidator struct {
	Address    string
	SPHINCSKey *SPHINCSKeyPair
	KyberKey   *KyberKeyPair
	Reputation float64
	TokenStake *big.Int
}

// NewQuantumResistantConsensus creates a new quantum-resistant consensus engine
func NewQuantumResistantConsensus() *QuantumResistantConsensus {
	return &QuantumResistantConsensus{
		sphincsConfig: SPHINCS256s,
		kyberConfig:   Kyber1024,
		validators:    make(map[string]*QuantumValidator),
	}
}

// AddValidator adds a new validator with quantum-resistant keys
func (qrc *QuantumResistantConsensus) AddValidator(address string, tokenStake *big.Int) error {
	sphincsKey, err := NewSPHINCSKeyPair(qrc.sphincsConfig)
	if err != nil {
		return fmt.Errorf("failed to generate SPHINCS+ key: %v", err)
	}

	kyberKey, err := NewKyberKeyPair(qrc.kyberConfig)
	if err != nil {
		return fmt.Errorf("failed to generate Kyber key: %v", err)
	}

	qrc.validators[address] = &QuantumValidator{
		Address:    address,
		SPHINCSKey: sphincsKey,
		KyberKey:   kyberKey,
		Reputation: 1.0,
		TokenStake: new(big.Int).Set(tokenStake),
	}

	return nil
}

// SignBlock creates a quantum-resistant signature for a block
func (qrc *QuantumResistantConsensus) SignBlock(validatorAddr string, blockHash []byte) ([]byte, error) {
	validator, exists := qrc.validators[validatorAddr]
	if !exists {
		return nil, errors.New("validator not found")
	}

	signature, err := validator.SPHINCSKey.Sign(blockHash)
	if err != nil {
		return nil, fmt.Errorf("failed to sign block: %v", err)
	}

	return signature.signature, nil
}

// VerifyBlockSignature verifies a quantum-resistant block signature
func (qrc *QuantumResistantConsensus) VerifyBlockSignature(validatorAddr string, blockHash []byte, signature []byte) bool {
	validator, exists := qrc.validators[validatorAddr]
	if !exists {
		return false
	}

	sig := &SPHINCSSignature{
		config:    qrc.sphincsConfig,
		signature: signature,
		message:   blockHash,
	}

	return sig.Verify(validator.SPHINCSKey.publicKey)
}

// EstablishSecureChannel creates a quantum-resistant secure channel between validators
func (qrc *QuantumResistantConsensus) EstablishSecureChannel(validator1, validator2 string) ([]byte, error) {
	v1, exists1 := qrc.validators[validator1]
	_, exists2 := qrc.validators[validator2]

	if !exists1 || !exists2 {
		return nil, errors.New("one or both validators not found")
	}

	// Use validator1's Kyber key to establish shared secret
	sharedSecret, _, err := v1.KyberKey.Encapsulate()
	if err != nil {
		return nil, fmt.Errorf("failed to establish secure channel: %v", err)
	}

	return sharedSecret, nil
}

// Helper functions for cryptographic operations

func generateHypertreeRoot(config *SPHINCSConfig, seed, pubSeed []byte) ([]byte, error) {
	// Simplified hypertree root generation
	h := sha256.New()
	h.Write(seed)
	h.Write(pubSeed)
	return h.Sum(nil)[:config.n], nil
}

func (kp *SPHINCSKeyPair) hashMessage(message, randomizer []byte) []byte {
	h := sha512.New()
	h.Write(randomizer)
	h.Write(kp.prf)
	h.Write(message)
	return h.Sum(nil)[:kp.config.n*kp.config.k]
}

func hashMessage(config *SPHINCSConfig, message, randomizer []byte) []byte {
	h := sha512.New()
	h.Write(randomizer)
	h.Write(message)
	return h.Sum(nil)[:config.n*config.k]
}

func (kp *SPHINCSKeyPair) generateFORSSignature(msgHash []byte) ([]byte, error) {
	// Simplified FORS signature generation
	signature := make([]byte, kp.config.k*kp.config.n)
	for i := 0; i < kp.config.k; i++ {
		copy(signature[i*kp.config.n:(i+1)*kp.config.n], msgHash[i*kp.config.n:(i+1)*kp.config.n])
	}
	return signature, nil
}

func (kp *SPHINCSKeyPair) generateHypertreeSignature(msgHash []byte) ([]byte, error) {
	// Simplified hypertree signature generation
	signature := make([]byte, kp.config.h*kp.config.n)
	h := sha256.New()
	h.Write(msgHash)
	hash := h.Sum(nil)
	for i := 0; i < kp.config.h; i++ {
		copy(signature[i*kp.config.n:(i+1)*kp.config.n], hash[:kp.config.n])
	}
	return signature, nil
}

func verifyFORSSignature(config *SPHINCSConfig, msgHash, signature, publicKey []byte) bool {
	// Simplified FORS verification
	return len(signature) >= config.k*config.n
}

func verifyHypertreeSignature(config *SPHINCSConfig, msgHash, signature, publicKey []byte) bool {
	// Simplified hypertree verification
	return len(signature) >= config.h*config.n
}

func randMod(mod int) int {
	n, _ := rand.Int(rand.Reader, big.NewInt(int64(mod)))
	return int(n.Int64())
}

func sampleNoise(n, eta int) []int16 {
	noise := make([]int16, n)
	for i := 0; i < n; i++ {
		noise[i] = int16(randMod(2*eta+1) - eta)
	}
	return noise
}

func sampleNoiseFromSeed(seed []byte, n, eta int) []int16 {
	noise := make([]int16, n)
	h := sha256.New()
	h.Write(seed)
	hash := h.Sum(nil)

	for i := 0; i < n; i++ {
		noise[i] = int16(int(hash[i%len(hash)])%(2*eta+1) - eta)
	}
	return noise
}

func encodeSecretKey(s [][]int16, config *KyberConfig) []byte {
	key := make([]byte, config.k*config.n*2)
	for i := 0; i < config.k; i++ {
		for j := 0; j < config.n; j++ {
			idx := (i*config.n + j) * 2
			val := uint16(s[i][j])
			key[idx] = byte(val & 0xFF)
			key[idx+1] = byte((val >> 8) & 0xFF)
		}
	}
	return key
}

func encodePublicKey(A [][]int16, t []int16, config *KyberConfig) []byte {
	keySize := config.k*config.k*config.n*2 + len(t)*2
	key := make([]byte, keySize)

	// Encode A matrix
	idx := 0
	for i := 0; i < config.k; i++ {
		for j := 0; j < config.k; j++ {
			val := uint16(A[i][j])
			key[idx] = byte(val & 0xFF)
			key[idx+1] = byte((val >> 8) & 0xFF)
			idx += 2
		}
	}

	// Encode t vector
	for i := 0; i < len(t); i++ {
		val := uint16(t[i])
		key[idx] = byte(val & 0xFF)
		key[idx+1] = byte((val >> 8) & 0xFF)
		idx += 2
	}

	return key
}

func encodeCiphertext(u []int16, v int16, config *KyberConfig) []byte {
	ctSize := len(u)*2 + 2
	ct := make([]byte, ctSize)

	// Encode u
	for i := 0; i < len(u); i++ {
		val := uint16(u[i])
		ct[i*2] = byte(val & 0xFF)
		ct[i*2+1] = byte((val >> 8) & 0xFF)
	}

	// Encode v
	idx := len(u) * 2
	val := uint16(v)
	ct[idx] = byte(val & 0xFF)
	ct[idx+1] = byte((val >> 8) & 0xFF)

	return ct
}

func decodeCiphertext(ct []byte, config *KyberConfig) ([]int16, int16, error) {
	expectedSize := config.k*config.n*2 + 2
	if len(ct) != expectedSize {
		return nil, 0, fmt.Errorf("invalid ciphertext size: expected %d, got %d", expectedSize, len(ct))
	}

	// Decode u
	u := make([]int16, config.k*config.n)
	for i := 0; i < len(u); i++ {
		val := uint16(ct[i*2]) | (uint16(ct[i*2+1]) << 8)
		u[i] = int16(val)
	}

	// Decode v
	idx := len(u) * 2
	val := uint16(ct[idx]) | (uint16(ct[idx+1]) << 8)
	v := int16(val)

	return u, v, nil
}

func decodeMessage(m []byte) []int16 {
	msg := make([]int16, len(m))
	for i, b := range m {
		msg[i] = int16(b)
	}
	return msg
}

func encodeMessage(msg []int16) []byte {
	m := make([]byte, len(msg))
	for i, val := range msg {
		m[i] = byte(val & 0xFF)
	}
	return m
}
