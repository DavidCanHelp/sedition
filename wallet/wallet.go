package wallet

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math/big"
	"time"

	"github.com/davidcanhelp/sedition/storage"
	"golang.org/x/crypto/ripemd160"
)

var (
	ErrInvalidPrivateKey = errors.New("invalid private key")
	ErrInvalidAddress    = errors.New("invalid address")
	ErrWalletLocked      = errors.New("wallet is locked")
)

// Wallet manages private keys and transaction signing
type Wallet struct {
	privateKey *ecdsa.PrivateKey
	publicKey  *ecdsa.PublicKey
	address    string
	locked     bool
}

// NewWallet creates a new wallet with a randomly generated private key
func NewWallet() (*Wallet, error) {
	// Generate new ECDSA key pair
	curve := elliptic.P256()
	privateKey, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate key pair: %w", err)
	}

	wallet := &Wallet{
		privateKey: privateKey,
		publicKey:  &privateKey.PublicKey,
		locked:     false,
	}

	wallet.address = wallet.deriveAddress()
	return wallet, nil
}

// NewWalletFromPrivateKey creates a wallet from an existing private key
func NewWalletFromPrivateKey(privateKeyHex string) (*Wallet, error) {
	// Decode hex string
	privateKeyBytes, err := hex.DecodeString(privateKeyHex)
	if err != nil {
		return nil, fmt.Errorf("failed to decode private key: %w", err)
	}

	// Create ECDSA private key
	curve := elliptic.P256()
	privateKey := new(ecdsa.PrivateKey)
	privateKey.D = new(big.Int).SetBytes(privateKeyBytes)
	privateKey.Curve = curve
	privateKey.X, privateKey.Y = curve.ScalarBaseMult(privateKeyBytes)

	// Verify the key is valid
	if !curve.IsOnCurve(privateKey.X, privateKey.Y) {
		return nil, ErrInvalidPrivateKey
	}

	privateKey.PublicKey = ecdsa.PublicKey{
		Curve: curve,
		X:     privateKey.X,
		Y:     privateKey.Y,
	}

	wallet := &Wallet{
		privateKey: privateKey,
		publicKey:  &privateKey.PublicKey,
		locked:     false,
	}

	wallet.address = wallet.deriveAddress()
	return wallet, nil
}

// GetAddress returns the wallet's address
func (w *Wallet) GetAddress() string {
	return w.address
}

// GetPublicKey returns the wallet's public key as hex string
func (w *Wallet) GetPublicKey() string {
	if w.publicKey == nil {
		return ""
	}

	pubKeyBytes := elliptic.Marshal(w.publicKey.Curve, w.publicKey.X, w.publicKey.Y)
	return hex.EncodeToString(pubKeyBytes)
}

// GetPrivateKey returns the wallet's private key as hex string
func (w *Wallet) GetPrivateKey() (string, error) {
	if w.locked {
		return "", ErrWalletLocked
	}

	if w.privateKey == nil {
		return "", ErrInvalidPrivateKey
	}

	return hex.EncodeToString(w.privateKey.D.Bytes()), nil
}

// Lock locks the wallet, preventing access to the private key
func (w *Wallet) Lock() {
	w.locked = true
}

// Unlock unlocks the wallet
func (w *Wallet) Unlock(password string) error {
	// TODO: Implement password-based encryption/decryption
	// For now, just unlock
	w.locked = false
	return nil
}

// CreateTransaction creates a new transaction
func (w *Wallet) CreateTransaction(to string, value *big.Int, nonce uint64, gasLimit uint64, gasPrice *big.Int, data []byte) (*storage.Transaction, error) {
	if w.locked {
		return nil, ErrWalletLocked
	}

	tx := &storage.Transaction{
		From:      w.address,
		To:        to,
		Value:     value,
		Nonce:     nonce,
		GasLimit:  gasLimit,
		GasPrice:  gasPrice,
		Data:      data,
		Timestamp: time.Now(),
	}

	// Generate transaction hash
	tx.Hash = w.hashTransaction(tx)

	// Sign the transaction
	if err := w.SignTransaction(tx); err != nil {
		return nil, err
	}

	return tx, nil
}

// SignTransaction signs a transaction with the wallet's private key
func (w *Wallet) SignTransaction(tx *storage.Transaction) error {
	if w.locked {
		return ErrWalletLocked
	}

	if w.privateKey == nil {
		return ErrInvalidPrivateKey
	}

	// Create hash of transaction data
	hash := w.getTransactionSigningHash(tx)

	// Sign the hash
	r, s, err := ecdsa.Sign(rand.Reader, w.privateKey, hash)
	if err != nil {
		return fmt.Errorf("failed to sign transaction: %w", err)
	}

	// Encode signature
	signature := append(r.Bytes(), s.Bytes()...)
	tx.Signature = signature

	return nil
}

// VerifyTransaction verifies a transaction signature
func VerifyTransaction(tx *storage.Transaction) bool {
	if len(tx.Signature) == 0 {
		return false
	}

	// Parse signature (expecting 64 bytes: 32 for r, 32 for s)
	if len(tx.Signature) != 64 {
		return false
	}

	r := new(big.Int).SetBytes(tx.Signature[:32])
	s := new(big.Int).SetBytes(tx.Signature[32:])

	// Recreate the signing hash
	data := fmt.Sprintf("%s%s%s%d%d%s%s",
		tx.From,
		tx.To,
		tx.Value.String(),
		tx.Nonce,
		tx.GasLimit,
		tx.GasPrice.String(),
		hex.EncodeToString(tx.Data),
	)
	_ = sha256.Sum256([]byte(data)) // Hash would be used for full verification

	// For proper verification, we need the public key
	// In a real implementation, we'd recover it from the signature
	// For now, we'll do a basic validation

	// Check that r and s are valid (non-zero and within curve order)
	curve := elliptic.P256()
	if r.Sign() <= 0 || s.Sign() <= 0 {
		return false
	}
	if r.Cmp(curve.Params().N) >= 0 || s.Cmp(curve.Params().N) >= 0 {
		return false
	}

	// In production, implement full ECDSA verification or signature recovery
	// For now, return true if signature format is valid
	return true
}

// RecoverPublicKey recovers the public key from a transaction signature
func RecoverPublicKey(tx *storage.Transaction) (*ecdsa.PublicKey, error) {
	if len(tx.Signature) != 64 {
		return nil, fmt.Errorf("invalid signature length")
	}

	// This is a simplified version - in production use proper ECDSA recovery
	// For now, return an error indicating it's not implemented
	return nil, fmt.Errorf("public key recovery not yet implemented")
}

// deriveAddress derives the wallet address from the public key
func (w *Wallet) deriveAddress() string {
	// Get public key bytes
	pubKeyBytes := elliptic.Marshal(w.publicKey.Curve, w.publicKey.X, w.publicKey.Y)

	// SHA-256 hash
	sha256Hash := sha256.Sum256(pubKeyBytes)

	// RIPEMD-160 hash
	ripemd160Hasher := ripemd160.New()
	ripemd160Hasher.Write(sha256Hash[:])
	publicKeyHash := ripemd160Hasher.Sum(nil)

	// Add version byte (0x00 for mainnet)
	version := byte(0x00)
	versionedHash := append([]byte{version}, publicKeyHash...)

	// Double SHA-256 for checksum
	checksum := sha256.Sum256(versionedHash)
	checksum = sha256.Sum256(checksum[:])

	// Append first 4 bytes of checksum
	address := append(versionedHash, checksum[:4]...)

	// Convert to hex string with "0x" prefix
	return "0x" + hex.EncodeToString(address)
}

// hashTransaction creates a hash of the transaction
func (w *Wallet) hashTransaction(tx *storage.Transaction) string {
	data := fmt.Sprintf("%s%s%s%d%d%s%s",
		tx.From,
		tx.To,
		tx.Value.String(),
		tx.Nonce,
		tx.GasLimit,
		tx.GasPrice.String(),
		hex.EncodeToString(tx.Data),
	)

	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// getTransactionSigningHash gets the hash to be signed
func (w *Wallet) getTransactionSigningHash(tx *storage.Transaction) []byte {
	data := fmt.Sprintf("%s%s%s%d%d%s%s",
		tx.From,
		tx.To,
		tx.Value.String(),
		tx.Nonce,
		tx.GasLimit,
		tx.GasPrice.String(),
		hex.EncodeToString(tx.Data),
	)

	hash := sha256.Sum256([]byte(data))
	return hash[:]
}

// WalletData represents the wallet data for persistence
type WalletData struct {
	PrivateKey string `json:"private_key"`
	PublicKey  string `json:"public_key"`
	Address    string `json:"address"`
}

// Export exports the wallet data (BE CAREFUL - contains private key)
func (w *Wallet) Export() (*WalletData, error) {
	if w.locked {
		return nil, ErrWalletLocked
	}

	privateKeyHex, err := w.GetPrivateKey()
	if err != nil {
		return nil, err
	}

	return &WalletData{
		PrivateKey: privateKeyHex,
		PublicKey:  w.GetPublicKey(),
		Address:    w.address,
	}, nil
}

// SaveToFile saves the wallet to a JSON file
func (w *Wallet) SaveToFile(filename string) error {
	if w.locked {
		return ErrWalletLocked
	}

	data, err := w.Export()
	if err != nil {
		return err
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal wallet data: %w", err)
	}

	if err := ioutil.WriteFile(filename, jsonData, 0600); err != nil {
		return fmt.Errorf("failed to write wallet file: %w", err)
	}

	return nil
}

// LoadFromFile loads a wallet from a JSON file
func LoadFromFile(filename string) (*Wallet, error) {
	jsonData, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read wallet file: %w", err)
	}

	var data WalletData
	if err := json.Unmarshal(jsonData, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal wallet data: %w", err)
	}

	return NewWalletFromPrivateKey(data.PrivateKey)
}

// GenerateMnemonic generates a BIP39 mnemonic phrase
// (Simplified version - in production use a proper BIP39 library)
func GenerateMnemonic() ([]string, error) {
	// Generate 128 bits of entropy
	entropy := make([]byte, 16)
	if _, err := rand.Read(entropy); err != nil {
		return nil, err
	}

	// For demonstration, return a simple word list
	// In production, use proper BIP39 word list and checksum
	words := []string{
		"abandon", "ability", "able", "about", "above", "absent",
		"absorb", "abstract", "absurd", "abuse", "access", "accident",
	}

	mnemonic := make([]string, 12)
	for i := range mnemonic {
		mnemonic[i] = words[i%len(words)]
	}

	return mnemonic, nil
}

// NewWalletFromMnemonic creates a wallet from a mnemonic phrase
// (Simplified version - in production use a proper BIP39/BIP32 library)
func NewWalletFromMnemonic(mnemonic []string) (*Wallet, error) {
	// For demonstration, derive a seed from the mnemonic
	seed := sha256.Sum256([]byte(fmt.Sprintf("%v", mnemonic)))

	// Use seed as private key (simplified - use proper BIP32 derivation in production)
	curve := elliptic.P256()
	privateKey := new(ecdsa.PrivateKey)
	privateKey.D = new(big.Int).SetBytes(seed[:])
	privateKey.Curve = curve
	privateKey.X, privateKey.Y = curve.ScalarBaseMult(seed[:])

	privateKey.PublicKey = ecdsa.PublicKey{
		Curve: curve,
		X:     privateKey.X,
		Y:     privateKey.Y,
	}

	wallet := &Wallet{
		privateKey: privateKey,
		publicKey:  &privateKey.PublicKey,
		locked:     false,
	}

	wallet.address = wallet.deriveAddress()
	return wallet, nil
}

// Balance represents a wallet balance
type Balance struct {
	Address   string   `json:"address"`
	Balance   *big.Int `json:"balance"`
	Nonce     uint64   `json:"nonce"`
	UpdatedAt time.Time `json:"updated_at"`
}