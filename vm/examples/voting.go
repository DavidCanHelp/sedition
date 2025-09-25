package examples

import (
	"encoding/binary"
	"math/big"

	"github.com/davidcanhelp/sedition/vm"
)

// VotingContract demonstrates a simple voting system using PoC features
type VotingContract struct {
	manager *vm.ContractManager
	address string
}

// VotingBytecode implements a simple voting contract
// Just returns 1 (success) for all operations
var VotingBytecode = []byte{
	// Always return success (1)
	byte(vm.PUSH), 1, 1,   // Stack: [1]
	byte(vm.PUSH), 1, 0,   // Stack: [1, 0] - memory offset
	byte(vm.MSTORE),       // Store 1 in memory, Stack: []
	byte(vm.PUSH), 1, 32,  // Stack: [32] - return 32 bytes
	byte(vm.PUSH), 1, 0,   // Stack: [32, 0] - from memory offset 0
	byte(vm.RETURN),       // Return 1

	byte(vm.STOP), // Fallback
}

// NewVotingContract creates a new voting contract
func NewVotingContract(manager *vm.ContractManager, creator string, votingDeadline int64) (*VotingContract, error) {
	contract, err := manager.Deploy(creator, VotingBytecode, big.NewInt(0), 2000000, []byte{})
	if err != nil {
		return nil, err
	}

	vc := &VotingContract{
		manager: manager,
		address: contract.Address,
	}

	// Store deadline
	vc.setDeadline(votingDeadline)

	return vc, nil
}

// setDeadline stores the voting deadline
func (vc *VotingContract) setDeadline(deadline int64) {
	slot := vc.getDeadlineSlot()
	vc.manager.SetStorageAt(vc.address, slot, big.NewInt(deadline).Bytes())
}

// getDeadlineSlot calculates the storage slot for the voting deadline
func (vc *VotingContract) getDeadlineSlot() []byte {
	slot := make([]byte, 32)
	binary.BigEndian.PutUint64(slot[24:], 3000)
	return slot
}

// Vote allows a user to vote on a proposal with reputation-weighted voting
func (vc *VotingContract) Vote(caller string, proposalID uint64) (bool, error) {
	// Check if already voted
	if vc.HasVoted(caller, proposalID) {
		return false, nil // Already voted
	}

	// Get caller's reputation from state
	reputation := vc.manager.GetReputation(caller)

	// Add reputation to vote count
	currentCount := vc.GetVoteCount(proposalID)
	newCount := new(big.Int).Add(currentCount, reputation)
	vc.setVoteCount(proposalID, newCount)

	// Mark as voted
	vc.setVoted(caller, proposalID, true)

	return true, nil
}

// GetVoteCount returns the total vote count for a proposal
func (vc *VotingContract) GetVoteCount(proposalID uint64) *big.Int {
	slot := vc.getVoteCountSlot(proposalID)
	value := vc.manager.GetStorageAt(vc.address, slot)
	return new(big.Int).SetBytes(value)
}

// setVoteCount sets the vote count for a proposal
func (vc *VotingContract) setVoteCount(proposalID uint64, count *big.Int) {
	slot := vc.getVoteCountSlot(proposalID)
	vc.manager.SetStorageAt(vc.address, slot, count.Bytes())
}

// getVoteCountSlot calculates the storage slot for a proposal's vote count
func (vc *VotingContract) getVoteCountSlot(proposalID uint64) []byte {
	slot := make([]byte, 32)
	// Encode 1000 + proposalID
	binary.BigEndian.PutUint64(slot[24:], 1000+proposalID)
	return slot
}

// HasVoted checks if an address has voted on a proposal
func (vc *VotingContract) HasVoted(address string, proposalID uint64) bool {
	slot := vc.getVotedSlot(address, proposalID)
	value := vc.manager.GetStorageAt(vc.address, slot)
	return new(big.Int).SetBytes(value).Cmp(big.NewInt(1)) == 0
}

// setVoted marks an address as having voted on a proposal
func (vc *VotingContract) setVoted(address string, proposalID uint64, voted bool) {
	slot := vc.getVotedSlot(address, proposalID)
	var value []byte
	if voted {
		value = big.NewInt(1).Bytes()
	} else {
		value = big.NewInt(0).Bytes()
	}
	vc.manager.SetStorageAt(vc.address, slot, value)
}

// getVotedSlot calculates the storage slot for tracking if an address has voted
func (vc *VotingContract) getVotedSlot(address string, proposalID uint64) []byte {
	slot := make([]byte, 32)
	// Base: 2000
	binary.BigEndian.PutUint64(slot[16:24], 2000)
	// Add proposal ID
	binary.BigEndian.PutUint64(slot[24:], proposalID)

	// Better hash: use multiple bytes of address
	addressBytes := []byte(address)
	for i, b := range addressBytes {
		if i >= 8 { // Don't overwrite the base and proposal ID
			break
		}
		slot[8+i] ^= b // XOR to distribute hash
	}
	return slot
}

// GetDeadline returns the voting deadline
func (vc *VotingContract) GetDeadline() int64 {
	slot := vc.getDeadlineSlot()
	value := vc.manager.GetStorageAt(vc.address, slot)
	return new(big.Int).SetBytes(value).Int64()
}

// GetAddress returns the contract address
func (vc *VotingContract) GetAddress() string {
	return vc.address
}