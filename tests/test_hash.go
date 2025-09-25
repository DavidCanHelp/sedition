package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/davidcanhelp/sedition/storage"
)

func calculateBlockHash(block *storage.BlockData) string {
	data := fmt.Sprintf("%d%s%d%s%s%s",
		block.Header.Height,
		block.Header.PreviousHash,
		block.Header.Timestamp.Unix(),
		block.Header.Proposer,
		block.Header.StateRoot,
		block.Header.TxRoot,
	)

	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

func main() {
	// Create a test genesis block matching the one in blockchain.go
	header := storage.BlockHeader{
		Height:       0,
		PreviousHash: "0000000000000000000000000000000000000000000000000000000000000000",
		Timestamp:    time.Unix(1700000000, 0), // Fixed timestamp for genesis
		Proposer:     "genesis",
		StateRoot:    "",
		TxRoot:       "",
	}

	block := &storage.BlockData{
		Header:       header,
		Transactions: []storage.Transaction{},
		Signatures:   []storage.Signature{},
	}

	// Calculate hash
	hash1 := calculateBlockHash(block)
	fmt.Printf("Hash 1: %s\n", hash1)

	// Set the hash and recalculate (simulating what happens in validation)
	block.Hash = hash1
	hash2 := calculateBlockHash(block)
	fmt.Printf("Hash 2: %s\n", hash2)

	if hash1 == hash2 {
		fmt.Println("✓ Hashes match - validation should pass")
	} else {
		fmt.Println("✗ Hashes don't match - validation will fail")
	}

	// Let's debug the exact string being hashed
	data := fmt.Sprintf("%d%s%d%s%s%s",
		block.Header.Height,
		block.Header.PreviousHash,
		block.Header.Timestamp.Unix(),
		block.Header.Proposer,
		block.Header.StateRoot,
		block.Header.TxRoot,
	)
	fmt.Printf("\nDebug - Data being hashed: %q\n", data)
	fmt.Printf("Length: %d bytes\n", len(data))
}