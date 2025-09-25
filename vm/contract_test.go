package vm

import (
	"math/big"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestContractDeployment(t *testing.T) {
	state := NewStateDB()
	cm := NewContractManager(state)

	// Set up deployer balance
	deployer := "deployer_address"
	state.AddBalance(deployer, big.NewInt(10000))

	t.Run("Basic Contract Deployment", func(t *testing.T) {
		// Simple contract bytecode (just returns)
		code := []byte{
			byte(PUSH), 1, 42, // PUSH 42
			byte(PUSH), 1, 0,  // PUSH 0 (offset)
			byte(MSTORE),      // Store 42 at memory offset 0
			byte(PUSH), 1, 32, // PUSH 32 (size)
			byte(PUSH), 1, 0,  // PUSH 0 (offset)
			byte(RETURN),      // RETURN
		}

		value := big.NewInt(100)
		gasLimit := uint64(1000000)

		contract, err := cm.Deploy(deployer, code, value, gasLimit, []byte{})
		require.NoError(t, err)
		require.NotNil(t, contract)

		// Verify contract properties
		assert.NotEmpty(t, contract.Address)
		assert.Equal(t, deployer, contract.Creator)
		assert.Equal(t, code, contract.Code)
		assert.Equal(t, value, contract.Balance)
		assert.False(t, contract.Destroyed)

		// Verify state changes
		storedCode := state.GetCode(contract.Address)
		assert.Equal(t, code, storedCode)

		contractBalance := state.GetBalance(contract.Address)
		assert.Equal(t, value, contractBalance)

		deployerBalance := state.GetBalance(deployer)
		assert.Equal(t, big.NewInt(9900), deployerBalance) // 10000 - 100

		// Verify nonce increment
		assert.Equal(t, uint64(1), state.GetNonce(deployer))
	})

	t.Run("Deploy with Insufficient Balance", func(t *testing.T) {
		state := NewStateDB()
		cm := NewContractManager(state)

		// Set insufficient balance
		state.AddBalance(deployer, big.NewInt(50))

		code := []byte{byte(STOP)}
		value := big.NewInt(100) // More than balance

		contract, err := cm.Deploy(deployer, code, value, 1000000, []byte{})
		assert.Error(t, err)
		assert.Nil(t, contract)
		assert.Contains(t, err.Error(), "insufficient balance")
	})

	t.Run("Deploy Multiple Contracts", func(t *testing.T) {
		state := NewStateDB()
		cm := NewContractManager(state)
		state.AddBalance(deployer, big.NewInt(10000))

		code1 := []byte{byte(PUSH), 1, 1, byte(STOP)}
		code2 := []byte{byte(PUSH), 1, 2, byte(STOP)}

		contract1, err := cm.Deploy(deployer, code1, big.NewInt(100), 1000000, []byte{})
		require.NoError(t, err)

		contract2, err := cm.Deploy(deployer, code2, big.NewInt(200), 1000000, []byte{})
		require.NoError(t, err)

		// Contracts should have different addresses
		assert.NotEqual(t, contract1.Address, contract2.Address)

		// Deployer nonce should increment
		assert.Equal(t, uint64(2), state.GetNonce(deployer))
	})
}

func TestContractExecution(t *testing.T) {
	state := NewStateDB()
	cm := NewContractManager(state)

	// Set up accounts
	deployer := "deployer"
	caller := "caller"
	state.AddBalance(deployer, big.NewInt(10000))
	state.AddBalance(caller, big.NewInt(5000))

	t.Run("Simple Contract Call", func(t *testing.T) {
		// Contract that doubles the input value and returns it
		code := []byte{
			byte(PUSH), 1, 0,   // PUSH 0 (calldata offset)
			byte(CALLDATALOAD), // Load first 32 bytes of calldata
			byte(PUSH), 1, 2,   // PUSH 2 (multiplier)
			byte(MUL),          // Multiply
			byte(PUSH), 1, 0,   // PUSH 0 (memory offset)
			byte(MSTORE),       // Store result in memory
			byte(PUSH), 1, 32,  // PUSH 32 (return size)
			byte(PUSH), 1, 0,   // PUSH 0 (return offset)
			byte(RETURN),       // Return result
		}

		// Deploy contract
		contract, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
		require.NoError(t, err)

		// Prepare call data (input value 5)
		input := make([]byte, 32)
		big.NewInt(5).FillBytes(input)

		// Call contract
		returnData, gasUsed, err := cm.Call(caller, contract.Address, input, big.NewInt(0), 1000000)
		require.NoError(t, err)
		assert.Greater(t, gasUsed, uint64(0))

		// Check return value (should be 10)
		result := new(big.Int).SetBytes(returnData)
		assert.Equal(t, big.NewInt(10), result)
	})

	t.Run("Contract Call with Value Transfer", func(t *testing.T) {
		// Simple contract that stores sent value
		code := []byte{
			byte(CALLVALUE),    // Get sent value
			byte(PUSH), 1, 0,   // PUSH 0 (storage key)
			byte(SSTORE),       // Store value
			byte(STOP),         // Stop
		}

		contract, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
		require.NoError(t, err)

		value := big.NewInt(500)

		// Call with value
		_, gasUsed, err := cm.Call(caller, contract.Address, []byte{}, value, 1000000)
		require.NoError(t, err)
		assert.Greater(t, gasUsed, uint64(0))

		// Check balances
		callerBalance := state.GetBalance(caller)
		assert.Equal(t, big.NewInt(4500), callerBalance) // 5000 - 500

		contractBalance := state.GetBalance(contract.Address)
		assert.Equal(t, value, contractBalance)

		// Check stored value
		storedValue := state.GetState(contract.Address, []byte{0})
		assert.Equal(t, value, new(big.Int).SetBytes(storedValue))
	})

	t.Run("Contract Call Insufficient Balance", func(t *testing.T) {
		code := []byte{byte(STOP)}
		contract, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
		require.NoError(t, err)

		value := big.NewInt(10000) // More than caller balance

		_, _, err = cm.Call(caller, contract.Address, []byte{}, value, 1000000)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "insufficient balance")
	})

	t.Run("Call Non-existent Contract", func(t *testing.T) {
		_, _, err := cm.Call(caller, "non_existent", []byte{}, big.NewInt(0), 1000000)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "contract not found")
	})

	t.Run("Static Call", func(t *testing.T) {
		// Contract that returns a constant value
		code := []byte{
			byte(PUSH), 1, 123, // PUSH 123
			byte(PUSH), 1, 0,   // PUSH 0 (memory offset)
			byte(MSTORE),       // Store in memory
			byte(PUSH), 1, 32,  // PUSH 32 (return size)
			byte(PUSH), 1, 0,   // PUSH 0 (return offset)
			byte(RETURN),       // Return
		}

		contract, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
		require.NoError(t, err)

		// Static call (read-only)
		returnData, gasUsed, err := cm.StaticCall(caller, contract.Address, []byte{}, 1000000)
		require.NoError(t, err)
		assert.Greater(t, gasUsed, uint64(0))

		result := new(big.Int).SetBytes(returnData)
		assert.Equal(t, big.NewInt(123), result)
	})
}

func TestContractStorage(t *testing.T) {
	state := NewStateDB()
	cm := NewContractManager(state)

	deployer := "deployer"
	state.AddBalance(deployer, big.NewInt(10000))

	t.Run("Storage Operations", func(t *testing.T) {
		// Contract that stores multiple values
		code := []byte{
			// Store 100 at key 1
			byte(PUSH), 1, 1,   // PUSH 1 (key)
			byte(PUSH), 1, 100, // PUSH 100 (value)
			byte(SSTORE),       // SSTORE

			// Store 200 at key 2
			byte(PUSH), 1, 2,   // PUSH 2 (key)
			byte(PUSH), 1, 200, // PUSH 200 (value)
			byte(SSTORE),       // SSTORE

			// Load and return value at key 1
			byte(PUSH), 1, 1,   // PUSH 1 (key)
			byte(SLOAD),        // SLOAD
			byte(PUSH), 1, 0,   // PUSH 0 (memory offset)
			byte(MSTORE),       // Store in memory
			byte(PUSH), 1, 32,  // PUSH 32 (return size)
			byte(PUSH), 1, 0,   // PUSH 0 (return offset)
			byte(RETURN),       // Return
		}

		contract, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
		require.NoError(t, err)

		// Call contract
		returnData, _, err := cm.Call(deployer, contract.Address, []byte{}, big.NewInt(0), 1000000)
		require.NoError(t, err)

		// Should return 100
		result := new(big.Int).SetBytes(returnData)
		assert.Equal(t, big.NewInt(100), result)

		// Check storage directly
		value1 := cm.GetStorageAt(contract.Address, []byte{1})
		assert.Equal(t, big.NewInt(100), new(big.Int).SetBytes(value1))

		value2 := cm.GetStorageAt(contract.Address, []byte{2})
		assert.Equal(t, big.NewInt(200), new(big.Int).SetBytes(value2))
	})
}

func TestContractEvents(t *testing.T) {
	state := NewStateDB()
	cm := NewContractManager(state)

	deployer := "deployer"
	caller := "caller"
	state.AddBalance(deployer, big.NewInt(10000))
	state.AddBalance(caller, big.NewInt(5000))

	t.Run("Event Emission", func(t *testing.T) {
		// Contract that emits events
		eventData := []byte("Hello, Event!")

		// Store event data in memory first
		code := []byte{
			// Store event data in memory
			byte(PUSH), 1, byte(len(eventData)), // Data size
			byte(PUSH), 1, 0,                    // Memory offset
		}

		// Manually create contract and emit event
		contract, err := cm.Deploy(deployer, []byte{byte(STOP)}, big.NewInt(0), 1000000, []byte{})
		require.NoError(t, err)

		// Create context and VM for event emission
		context := &Context{
			Address: contract.Address,
			Caller:  caller,
		}
		config := DefaultConfig()
		vm := NewVM(context, state, config)

		// Store data in memory and emit LOG0
		vm.memory.Store(0, eventData)
		vm.stack.Push(big.NewInt(int64(len(eventData))))
		vm.stack.Push(big.NewInt(0))

		err = vm.opLog(LOG0)
		require.NoError(t, err)

		// Check events
		events, err := cm.GetEvents(contract.Address)
		require.NoError(t, err)
		require.Len(t, events, 1)

		assert.Equal(t, contract.Address, events[0].Address)
		assert.Len(t, events[0].Topics, 0)
		assert.Equal(t, eventData, events[0].Data)
	})
}

func TestContractDestruction(t *testing.T) {
	state := NewStateDB()
	cm := NewContractManager(state)

	deployer := "deployer"
	beneficiary := "beneficiary"
	state.AddBalance(deployer, big.NewInt(10000))

	t.Run("Self Destruct", func(t *testing.T) {
		// Deploy contract with some balance
		code := []byte{byte(STOP)}
		contract, err := cm.Deploy(deployer, code, big.NewInt(1000), 1000000, []byte{})
		require.NoError(t, err)

		initialBeneficiaryBalance := state.GetBalance(beneficiary)

		// Destroy contract
		err = cm.DestroyContract(contract.Address, beneficiary)
		require.NoError(t, err)

		// Check that contract is marked for deletion
		assert.True(t, state.IsDeleted(contract.Address))

		// Check that balance was transferred
		contractBalance := state.GetBalance(contract.Address)
		assert.Equal(t, big.NewInt(0), contractBalance)

		beneficiaryBalance := state.GetBalance(beneficiary)
		expectedBalance := new(big.Int).Add(initialBeneficiaryBalance, big.NewInt(1000))
		assert.Equal(t, expectedBalance, beneficiaryBalance)
	})

	t.Run("Destroy Non-existent Contract", func(t *testing.T) {
		err := cm.DestroyContract("non_existent", beneficiary)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "contract not found")
	})
}

func TestGasEstimation(t *testing.T) {
	state := NewStateDB()
	cm := NewContractManager(state)

	deployer := "deployer"
	caller := "caller"
	state.AddBalance(deployer, big.NewInt(10000))
	state.AddBalance(caller, big.NewInt(5000))

	t.Run("Gas Estimation", func(t *testing.T) {
		// Simple arithmetic contract
		code := []byte{
			byte(PUSH), 1, 10, // PUSH 10
			byte(PUSH), 1, 20, // PUSH 20
			byte(ADD),         // ADD
			byte(STOP),        // STOP
		}

		contract, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
		require.NoError(t, err)

		// Estimate gas
		estimatedGas, err := cm.EstimateGas(caller, contract.Address, []byte{}, big.NewInt(0))
		require.NoError(t, err)
		assert.Greater(t, estimatedGas, uint64(0))

		// Actual execution should use less than or equal to estimated gas
		_, actualGas, err := cm.Call(caller, contract.Address, []byte{}, big.NewInt(0), estimatedGas)
		require.NoError(t, err)
		assert.LessOrEqual(t, actualGas, estimatedGas)
	})
}

func TestContractRegistry(t *testing.T) {
	registry := NewContractRegistry()

	contract1 := &Contract{
		Address: "contract1",
		Name:    "TestContract",
	}

	contract2 := &Contract{
		Address: "contract2",
		Name:    "AnotherContract",
	}

	t.Run("Register and Get", func(t *testing.T) {
		registry.Register(contract1)
		registry.Register(contract2)

		retrieved, exists := registry.Get("contract1")
		assert.True(t, exists)
		assert.Equal(t, contract1, retrieved)

		_, exists = registry.Get("non_existent")
		assert.False(t, exists)
	})

	t.Run("List Contracts", func(t *testing.T) {
		contracts := registry.List()
		assert.Len(t, contracts, 2)
	})

	t.Run("Find by Name", func(t *testing.T) {
		found := registry.FindByName("TestContract")
		require.Len(t, found, 1)
		assert.Equal(t, contract1, found[0])

		found = registry.FindByName("NonExistent")
		assert.Len(t, found, 0)
	})

	t.Run("Remove Contract", func(t *testing.T) {
		registry.Remove("contract1")

		_, exists := registry.Get("contract1")
		assert.False(t, exists)

		contracts := registry.List()
		assert.Len(t, contracts, 1)
	})
}

// Benchmark tests for contract operations
func BenchmarkContractDeployment(b *testing.B) {
	state := NewStateDB()
	cm := NewContractManager(state)

	deployer := "deployer"
	state.AddBalance(deployer, big.NewInt(1000000))

	code := []byte{
		byte(PUSH), 1, 42,
		byte(STOP),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkContractCall(b *testing.B) {
	state := NewStateDB()
	cm := NewContractManager(state)

	deployer := "deployer"
	caller := "caller"
	state.AddBalance(deployer, big.NewInt(10000))
	state.AddBalance(caller, big.NewInt(10000))

	// Simple contract
	code := []byte{
		byte(PUSH), 1, 42,
		byte(PUSH), 1, 0,
		byte(MSTORE),
		byte(PUSH), 1, 32,
		byte(PUSH), 1, 0,
		byte(RETURN),
	}

	contract, err := cm.Deploy(deployer, code, big.NewInt(0), 1000000, []byte{})
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := cm.Call(caller, contract.Address, []byte{}, big.NewInt(0), 1000000)
		if err != nil {
			b.Fatal(err)
		}
	}
}