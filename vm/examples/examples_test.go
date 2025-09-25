package examples

import (
	"math/big"
	"testing"
	"time"

	"github.com/davidcanhelp/sedition/vm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleStorageContract(t *testing.T) {
	state := vm.NewStateDB()
	manager := vm.NewContractManager(state)

	creator := "creator"
	user := "user"
	state.AddBalance(creator, big.NewInt(10000))
	state.AddBalance(user, big.NewInt(5000))

	t.Run("Deploy and Use Storage Contract", func(t *testing.T) {
		// Deploy contract
		contract, err := NewSimpleStorageContract(manager, creator)
		require.NoError(t, err)
		require.NotNil(t, contract)

		// Set a value
		err = contract.Set(user, big.NewInt(42))
		require.NoError(t, err)

		// Get the value back
		value, err := contract.Get(user)
		require.NoError(t, err)
		assert.Equal(t, big.NewInt(42), value)

		// Set a different value
		err = contract.Set(user, big.NewInt(100))
		require.NoError(t, err)

		// Verify it changed
		value, err = contract.Get(user)
		require.NoError(t, err)
		assert.Equal(t, big.NewInt(100), value)
	})

	t.Run("Multiple Users Same Contract", func(t *testing.T) {
		user2 := "user2"
		state.AddBalance(user2, big.NewInt(5000))

		contract, err := NewSimpleStorageContract(manager, creator)
		require.NoError(t, err)

		// Both users set different values
		err = contract.Set(user, big.NewInt(10))
		require.NoError(t, err)

		err = contract.Set(user2, big.NewInt(20))
		require.NoError(t, err)

		// The last set wins (this is how our simple contract works)
		value, err := contract.Get(user)
		require.NoError(t, err)
		assert.Equal(t, big.NewInt(20), value) // Last value set

		value, err = contract.Get(user2)
		require.NoError(t, err)
		assert.Equal(t, big.NewInt(20), value) // Same for all callers
	})
}

func TestTokenContract(t *testing.T) {
	state := vm.NewStateDB()
	manager := vm.NewContractManager(state)

	creator := "creator"
	alice := "alice"
	bob := "bob"

	state.AddBalance(creator, big.NewInt(10000))
	state.AddBalance(alice, big.NewInt(5000))
	state.AddBalance(bob, big.NewInt(5000))

	t.Run("Deploy Token Contract", func(t *testing.T) {
		totalSupply := big.NewInt(1000000)

		contract, err := NewTokenContract(manager, creator, "TestToken", "TTK", totalSupply)
		require.NoError(t, err)
		require.NotNil(t, contract)

		// Check contract properties
		assert.Equal(t, "TestToken", contract.GetName())
		assert.Equal(t, "TTK", contract.GetSymbol())

		// Check total supply
		supply := contract.TotalSupply()
		assert.Equal(t, totalSupply, supply)

		// Creator should have all initial tokens
		creatorBalance := contract.BalanceOf(creator)
		assert.Equal(t, totalSupply, creatorBalance)

		// Others should have zero balance
		aliceBalance := contract.BalanceOf(alice)
		assert.Equal(t, big.NewInt(0), aliceBalance)
	})

	t.Run("Token Transfer", func(t *testing.T) {
		totalSupply := big.NewInt(1000000)
		contract, err := NewTokenContract(manager, creator, "TestToken", "TTK", totalSupply)
		require.NoError(t, err)

		transferAmount := big.NewInt(1000)

		// Creator transfers tokens to Alice
		success, err := contract.Transfer(creator, alice, transferAmount)
		require.NoError(t, err)
		assert.True(t, success)

		// Check balances after transfer
		creatorBalance := contract.BalanceOf(creator)
		expectedCreatorBalance := new(big.Int).Sub(totalSupply, transferAmount)
		assert.Equal(t, expectedCreatorBalance, creatorBalance)

		aliceBalance := contract.BalanceOf(alice)
		assert.Equal(t, transferAmount, aliceBalance)

		// Alice transfers to Bob
		transferAmount2 := big.NewInt(500)
		success, err = contract.Transfer(alice, bob, transferAmount2)
		require.NoError(t, err)
		assert.True(t, success)

		// Check final balances
		aliceBalance = contract.BalanceOf(alice)
		expectedAliceBalance := new(big.Int).Sub(transferAmount, transferAmount2)
		assert.Equal(t, expectedAliceBalance, aliceBalance)

		bobBalance := contract.BalanceOf(bob)
		assert.Equal(t, transferAmount2, bobBalance)
	})

	t.Run("Transfer More Than Balance", func(t *testing.T) {
		contract, err := NewTokenContract(manager, creator, "TestToken", "TTK", big.NewInt(1000))
		require.NoError(t, err)

		// Try to transfer more than balance
		success, err := contract.Transfer(alice, bob, big.NewInt(2000))
		require.NoError(t, err)
		assert.False(t, success) // Should fail

		// Balances should remain unchanged
		aliceBalance := contract.BalanceOf(alice)
		assert.Equal(t, big.NewInt(0), aliceBalance)

		bobBalance := contract.BalanceOf(bob)
		assert.Equal(t, big.NewInt(0), bobBalance)
	})
}

func TestVotingContract(t *testing.T) {
	state := vm.NewStateDB()
	manager := vm.NewContractManager(state)

	creator := "creator"
	voter1 := "voter1"
	voter2 := "voter2"
	voter3 := "voter3"

	// Set up balances and reputations
	state.AddBalance(creator, big.NewInt(10000))
	state.AddBalance(voter1, big.NewInt(5000))
	state.AddBalance(voter2, big.NewInt(5000))
	state.AddBalance(voter3, big.NewInt(5000))

	// Set different reputation scores
	state.SetReputation(voter1, big.NewInt(10))
	state.SetReputation(voter2, big.NewInt(20))
	state.SetReputation(voter3, big.NewInt(5))

	t.Run("Deploy Voting Contract", func(t *testing.T) {
		deadline := time.Now().Add(time.Hour).Unix()

		contract, err := NewVotingContract(manager, creator, deadline)
		require.NoError(t, err)
		require.NotNil(t, contract)

		// Check deadline
		storedDeadline := contract.GetDeadline()
		assert.Equal(t, deadline, storedDeadline)

		// Initial vote count should be zero
		voteCount := contract.GetVoteCount(1)
		assert.Equal(t, big.NewInt(0), voteCount)
	})

	t.Run("Voting with Reputation Weight", func(t *testing.T) {
		deadline := time.Now().Add(time.Hour).Unix()
		contract, err := NewVotingContract(manager, creator, deadline)
		require.NoError(t, err)

		proposalID := uint64(1)

		// Voter1 votes (reputation = 10)
		success, err := contract.Vote(voter1, proposalID)
		require.NoError(t, err)
		assert.True(t, success)

		// Check vote count
		voteCount := contract.GetVoteCount(proposalID)
		assert.Equal(t, big.NewInt(10), voteCount) // Should equal voter1's reputation

		// Check that voter1 is marked as voted
		hasVoted := contract.HasVoted(voter1, proposalID)
		assert.True(t, hasVoted)

		// Voter2 votes (reputation = 20)
		success, err = contract.Vote(voter2, proposalID)
		require.NoError(t, err)
		assert.True(t, success)

		// Vote count should now be 10 + 20 = 30
		voteCount = contract.GetVoteCount(proposalID)
		assert.Equal(t, big.NewInt(30), voteCount)

		// Voter3 votes (reputation = 5)
		success, err = contract.Vote(voter3, proposalID)
		require.NoError(t, err)
		assert.True(t, success)

		// Final vote count should be 30 + 5 = 35
		voteCount = contract.GetVoteCount(proposalID)
		assert.Equal(t, big.NewInt(35), voteCount)
	})

	t.Run("Prevent Double Voting", func(t *testing.T) {
		deadline := time.Now().Add(time.Hour).Unix()
		contract, err := NewVotingContract(manager, creator, deadline)
		require.NoError(t, err)

		proposalID := uint64(1)

		// Voter1 votes first time
		success, err := contract.Vote(voter1, proposalID)
		require.NoError(t, err)
		assert.True(t, success)

		voteCount1 := contract.GetVoteCount(proposalID)

		// Voter1 tries to vote again
		success, err = contract.Vote(voter1, proposalID)
		require.NoError(t, err)
		assert.False(t, success) // Should fail

		// Vote count should remain the same
		voteCount2 := contract.GetVoteCount(proposalID)
		assert.Equal(t, voteCount1, voteCount2)
	})

	t.Run("Multiple Proposals", func(t *testing.T) {
		deadline := time.Now().Add(time.Hour).Unix()
		contract, err := NewVotingContract(manager, creator, deadline)
		require.NoError(t, err)

		// Vote on different proposals
		success, err := contract.Vote(voter1, 1)
		require.NoError(t, err)
		assert.True(t, success)

		success, err = contract.Vote(voter1, 2)
		require.NoError(t, err)
		assert.True(t, success) // Should succeed - different proposal

		// Check vote counts
		voteCount1 := contract.GetVoteCount(1)
		voteCount2 := contract.GetVoteCount(2)

		assert.Equal(t, big.NewInt(10), voteCount1) // voter1's reputation
		assert.Equal(t, big.NewInt(10), voteCount2) // voter1's reputation

		// Check voting status
		assert.True(t, contract.HasVoted(voter1, 1))
		assert.True(t, contract.HasVoted(voter1, 2))
		assert.False(t, contract.HasVoted(voter2, 1))
	})

	// Note: Testing expired voting would require manipulating the VM's timestamp
	// which is more complex in this simplified implementation
}

func TestContractInteraction(t *testing.T) {
	state := vm.NewStateDB()
	manager := vm.NewContractManager(state)

	creator := "creator"
	user := "user"

	state.AddBalance(creator, big.NewInt(10000))
	state.AddBalance(user, big.NewInt(5000))
	state.SetReputation(user, big.NewInt(15))

	t.Run("Token and Voting Integration", func(t *testing.T) {
		// Deploy token contract
		tokenContract, err := NewTokenContract(manager, creator, "VoteToken", "VTK", big.NewInt(1000))
		require.NoError(t, err)

		// Deploy voting contract
		deadline := time.Now().Add(time.Hour).Unix()
		votingContract, err := NewVotingContract(manager, creator, deadline)
		require.NoError(t, err)

		// Give user some tokens
		success, err := tokenContract.Transfer(creator, user, big.NewInt(100))
		require.NoError(t, err)
		assert.True(t, success)

		// User can vote (using their reputation, not token balance)
		success, err = votingContract.Vote(user, 1)
		require.NoError(t, err)
		assert.True(t, success)

		// Vote should be weighted by reputation (15)
		voteCount := votingContract.GetVoteCount(1)
		assert.Equal(t, big.NewInt(15), voteCount)

		// User should still have their tokens
		balance := tokenContract.BalanceOf(user)
		assert.Equal(t, big.NewInt(100), balance)
	})
}

// Benchmark tests
func BenchmarkSimpleStorageContract(b *testing.B) {
	state := vm.NewStateDB()
	manager := vm.NewContractManager(state)
	creator := "creator"
	state.AddBalance(creator, big.NewInt(1000000))

	contract, err := NewSimpleStorageContract(manager, creator)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := contract.Set(creator, big.NewInt(int64(i)))
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTokenTransfer(b *testing.B) {
	state := vm.NewStateDB()
	manager := vm.NewContractManager(state)
	creator := "creator"
	user := "user"

	state.AddBalance(creator, big.NewInt(1000000))
	state.AddBalance(user, big.NewInt(1000000))

	contract, err := NewTokenContract(manager, creator, "BenchToken", "BTK", big.NewInt(1000000000))
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := contract.Transfer(creator, user, big.NewInt(1))
		if err != nil {
			b.Fatal(err)
		}
	}
}