package vm

import (
	"math/big"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVMBasicOperations(t *testing.T) {
	state := NewStateDB()
	context := &Context{
		Address: "test_contract",
		Caller:  "test_caller",
		Origin:  "test_origin",
		Value:   big.NewInt(0),
		Input:   []byte{},
	}

	config := DefaultConfig()
	vm := NewVM(context, state, config)

	t.Run("Stack Operations", func(t *testing.T) {
		// Test PUSH and POP
		code := []byte{
			byte(PUSH), 1, 42,  // PUSH 42
			byte(POP),          // POP
			byte(STOP),         // STOP
		}

		_, gasUsed, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)
		assert.Greater(t, gasUsed, uint64(0))
	})

	t.Run("Arithmetic Operations", func(t *testing.T) {
		// Test ADD: 10 + 20 = 30
		code := []byte{
			byte(PUSH), 1, 10,  // PUSH 10
			byte(PUSH), 1, 20,  // PUSH 20
			byte(ADD),          // ADD
			byte(STOP),         // STOP
		}

		vm = NewVM(context, state, config) // Reset VM
		_, gasUsed, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		// Check that result (30) is on stack
		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(30), result.Int64())
	})

	t.Run("Memory Operations", func(t *testing.T) {
		// Store value in memory and load it back
		code := []byte{
			byte(PUSH), 1, 0,   // PUSH 0 (offset)
			byte(PUSH), 1, 100, // PUSH 100 (value)
			byte(MSTORE),       // MSTORE
			byte(PUSH), 1, 0,   // PUSH 0 (offset)
			byte(MLOAD),        // MLOAD
			byte(STOP),         // STOP
		}

		vm = NewVM(context, state, config) // Reset VM
		_, gasUsed, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		// Check that loaded value is on stack
		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(100), result.Int64())
	})

	t.Run("Storage Operations", func(t *testing.T) {
		// Store value in contract storage and load it back
		code := []byte{
			byte(PUSH), 1, 1,   // PUSH 1 (key)
			byte(PUSH), 1, 200, // PUSH 200 (value)
			byte(SSTORE),       // SSTORE
			byte(PUSH), 1, 1,   // PUSH 1 (key)
			byte(SLOAD),        // SLOAD
			byte(STOP),         // STOP
		}

		vm = NewVM(context, state, config) // Reset VM
		_, gasUsed, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		// Check that loaded value is on stack
		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(200), result.Int64())

		// Check that value was stored in state
		stored := state.GetState("test_contract", []byte{1})
		assert.Equal(t, big.NewInt(200), new(big.Int).SetBytes(stored))
	})

	t.Run("Comparison Operations", func(t *testing.T) {
		// Test EQ: 5 == 5 should return 1
		code := []byte{
			byte(PUSH), 1, 5,   // PUSH 5
			byte(PUSH), 1, 5,   // PUSH 5
			byte(EQ),           // EQ
			byte(STOP),         // STOP
		}

		vm = NewVM(context, state, config) // Reset VM
		_, gasUsed, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(1), result.Int64()) // True
	})

	t.Run("Jump Operations", func(t *testing.T) {
		// Test JUMP to JUMPDEST
		code := []byte{
			byte(PUSH), 1, 5,   // PUSH 5 (jump destination)
			byte(JUMP),         // JUMP
			byte(PUSH), 1, 99,  // This should be skipped
			byte(JUMPDEST),     // Jump destination (PC = 5)
			byte(PUSH), 1, 77,  // This should execute
			byte(STOP),         // STOP
		}

		vm = NewVM(context, state, config) // Reset VM
		_, gasUsed, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		// Only 77 should be on stack (99 was skipped)
		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(77), result.Int64())
	})
}

func TestVMSystemOperations(t *testing.T) {
	state := NewStateDB()

	// Set up initial balances
	state.AddBalance("caller", big.NewInt(1000))
	state.AddBalance("contract", big.NewInt(500))

	context := &Context{
		Address:     "contract",
		Caller:      "caller",
		Origin:      "origin",
		Value:       big.NewInt(100),
		Input:       []byte{1, 2, 3, 4},
		BlockHeight: 12345,
		Timestamp:   1640995200,
	}

	config := DefaultConfig()
	vm := NewVM(context, state, config)

	t.Run("System Information", func(t *testing.T) {
		// Test ADDRESS opcode
		code := []byte{
			byte(ADDRESS), // Get contract address
			byte(STOP),
		}

		_, _, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		result, err := vm.stack.Pop()
		require.NoError(t, err)

		// Address should be encoded as bytes
		addressBytes := []byte("contract")
		expectedValue := new(big.Int).SetBytes(addressBytes)
		assert.Equal(t, expectedValue, result)
	})

	t.Run("Balance Check", func(t *testing.T) {
		// Test BALANCE opcode
		callerBytes := []byte("caller")
		code := []byte{
			byte(PUSH), byte(len(callerBytes)), // Push address length
		}
		// Add address bytes
		code = append(code, callerBytes...)
		code = append(code, []byte{
			byte(BALANCE), // Get balance
			byte(STOP),
		}...)

		// Create address value properly
		vm = NewVM(context, state, config) // Reset VM

		// Manually push caller address and get balance
		vm.stack.Push(new(big.Int).SetBytes(callerBytes))
		err := vm.opBalance()
		require.NoError(t, err)

		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, big.NewInt(1000), result)
	})

	t.Run("Call Data Operations", func(t *testing.T) {
		// Test CALLDATASIZE
		code := []byte{
			byte(CALLDATASIZE), // Get call data size
			byte(STOP),
		}

		vm = NewVM(context, state, config) // Reset VM
		_, _, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(4), result.Int64()) // Input has 4 bytes
	})
}

func TestVMGasMetering(t *testing.T) {
	state := NewStateDB()
	context := &Context{
		Address: "test_contract",
		Caller:  "test_caller",
		Value:   big.NewInt(0),
	}

	config := DefaultConfig()
	config.EnableGasMetering = true
	vm := NewVM(context, state, config)

	t.Run("Out of Gas", func(t *testing.T) {
		// Simple operation that should run out of gas
		code := []byte{
			byte(PUSH), 1, 42, // PUSH costs 3 gas
			byte(PUSH), 1, 43, // Another 3 gas
			byte(ADD),         // ADD costs 3 gas
			byte(STOP),        // STOP costs 0 gas
		}

		// Set gas limit too low (total cost is 9, set limit to 8)
		_, gasUsed, err := vm.Execute(code, []byte{}, 8)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "out of gas")
		assert.Equal(t, uint64(8), gasUsed) // Should use all available gas
	})

	t.Run("Exact Gas Usage", func(t *testing.T) {
		code := []byte{
			byte(PUSH), 1, 42, // 3 gas
			byte(STOP),        // 0 gas
		}

		vm = NewVM(context, state, config) // Reset VM
		_, gasUsed, err := vm.Execute(code, []byte{}, 3) // Exact gas needed

		require.NoError(t, err)
		assert.Equal(t, uint64(3), gasUsed)
	})
}

func TestVMErrorHandling(t *testing.T) {
	state := NewStateDB()
	context := &Context{
		Address: "test_contract",
		Caller:  "test_caller",
		Value:   big.NewInt(0),
	}

	config := DefaultConfig()
	vm := NewVM(context, state, config)

	t.Run("Stack Underflow", func(t *testing.T) {
		// Try to ADD with empty stack
		code := []byte{
			byte(ADD), // This should fail - no values on stack
			byte(STOP),
		}

		_, _, err := vm.Execute(code, []byte{}, 1000000)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "stack underflow")
	})

	t.Run("Invalid Jump", func(t *testing.T) {
		// Jump to invalid destination
		code := []byte{
			byte(PUSH), 1, 10,  // Jump to PC 10 (invalid)
			byte(JUMP),         // JUMP
			byte(STOP),
		}

		vm = NewVM(context, state, config) // Reset VM
		_, _, err := vm.Execute(code, []byte{}, 1000000)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "jump")
	})

	t.Run("Division by Zero", func(t *testing.T) {
		// Divide by zero should return 0 (EVM behavior)
		code := []byte{
			byte(PUSH), 1, 10, // PUSH 10
			byte(PUSH), 1, 0,  // PUSH 0
			byte(DIV),         // DIV 10/0
			byte(STOP),
		}

		vm = NewVM(context, state, config) // Reset VM
		_, _, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err) // Should not error

		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(0), result.Int64()) // Should return 0
	})
}

func TestVMCustomOperations(t *testing.T) {
	state := NewStateDB()

	// Set up initial state
	state.AddBalance("validator", big.NewInt(10000))
	state.SetReputation("validator", big.NewInt(50))

	context := &Context{
		Address: "test_contract",
		Caller:  "validator",
		Value:   big.NewInt(0),
	}

	config := DefaultConfig()
	vm := NewVM(context, state, config)

	t.Run("Stake Operation", func(t *testing.T) {
		// Test STAKE operation
		code := []byte{
			byte(PUSH), 1, 100, // PUSH 100 (stake amount)
			byte(STAKE),        // STAKE
			byte(STOP),
		}

		_, _, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		// Check that stake was recorded
		stake := state.GetStake("validator")
		assert.Equal(t, big.NewInt(100), stake)

		// Check that balance was reduced
		balance := state.GetBalance("validator")
		assert.Equal(t, big.NewInt(9900), balance)
	})

	t.Run("Reputation Operation", func(t *testing.T) {
		// Test REPUTATION operation
		validatorBytes := []byte("validator")

		vm = NewVM(context, state, config) // Reset VM

		// Manually push validator address and get reputation
		vm.stack.Push(new(big.Int).SetBytes(validatorBytes))
		err := vm.opReputation()
		require.NoError(t, err)

		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, big.NewInt(50), result)
	})

	t.Run("Contribute Operation", func(t *testing.T) {
		// Test CONTRIBUTE operation
		code := []byte{
			byte(PUSH), 1, 1,   // PUSH 1 (contribution type)
			byte(PUSH), 1, 50,  // PUSH 50 (contribution amount)
			byte(CONTRIBUTE),   // CONTRIBUTE
			byte(STOP),
		}

		vm = NewVM(context, state, config) // Reset VM
		_, _, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		// Check that contribution was recorded
		contribution := state.GetContribution("validator", 1)
		assert.Equal(t, big.NewInt(50), contribution)

		// Check success result on stack
		result, err := vm.stack.Pop()
		require.NoError(t, err)
		assert.Equal(t, int64(1), result.Int64()) // Success
	})
}

func TestVMLogging(t *testing.T) {
	state := NewStateDB()
	context := &Context{
		Address: "test_contract",
		Caller:  "test_caller",
		Value:   big.NewInt(0),
	}

	config := DefaultConfig()
	vm := NewVM(context, state, config)

	t.Run("LOG0 Operation", func(t *testing.T) {
		// Test LOG0 (log with no topics)
		data := []byte("Hello, World!")

		// Store data in memory first
		vm.memory.Store(0, data)

		code := []byte{
			byte(PUSH), 1, byte(len(data)), // PUSH data size
			byte(PUSH), 1, 0,               // PUSH data offset
			byte(LOG0),                     // LOG0
			byte(STOP),
		}

		_, _, err := vm.Execute(code, []byte{}, 1000000)
		require.NoError(t, err)

		// Check that log was created
		logs := state.GetLogs()
		require.Len(t, logs, 1)
		assert.Equal(t, "test_contract", logs[0].Address)
		assert.Len(t, logs[0].Topics, 0)
		assert.Equal(t, data, logs[0].Data)
	})

	t.Run("LOG1 Operation", func(t *testing.T) {
		state.ClearLogs() // Clear previous logs

		data := []byte("Event data")
		topic := []byte("topic1")

		vm = NewVM(context, state, config) // Reset VM
		vm.memory.Store(0, data)

		code := []byte{
			byte(PUSH), 1, byte(len(data)), // PUSH data size
			byte(PUSH), 1, 0,               // PUSH data offset
			// Push topic (manually)
		}

		// Manually execute to test LOG1
		vm.stack.Push(big.NewInt(int64(len(data))))
		vm.stack.Push(big.NewInt(0))
		vm.stack.Push(new(big.Int).SetBytes(topic))

		err := vm.opLog(LOG1)
		require.NoError(t, err)

		logs := state.GetLogs()
		require.Len(t, logs, 1)
		assert.Equal(t, "test_contract", logs[0].Address)
		assert.Len(t, logs[0].Topics, 1)
		assert.Equal(t, topic, logs[0].Topics[0])
		assert.Equal(t, data, logs[0].Data)
	})
}

// Benchmark tests
func BenchmarkVMBasicOperations(b *testing.B) {
	state := NewStateDB()
	context := &Context{
		Address: "test_contract",
		Caller:  "test_caller",
		Value:   big.NewInt(0),
	}

	config := DefaultConfig()
	config.EnableGasMetering = false // Disable for benchmarking

	// Simple arithmetic program
	code := []byte{
		byte(PUSH), 1, 10,  // PUSH 10
		byte(PUSH), 1, 20,  // PUSH 20
		byte(ADD),          // ADD
		byte(PUSH), 1, 30,  // PUSH 30
		byte(MUL),          // MUL
		byte(STOP),         // STOP
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vm := NewVM(context, state, config)
		_, _, err := vm.Execute(code, []byte{}, 1000000)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVMMemoryOperations(b *testing.B) {
	state := NewStateDB()
	context := &Context{
		Address: "test_contract",
		Caller:  "test_caller",
		Value:   big.NewInt(0),
	}

	config := DefaultConfig()
	config.EnableGasMetering = false

	// Memory intensive program
	code := []byte{
		byte(PUSH), 1, 0,   // PUSH 0 (offset)
		byte(PUSH), 1, 100, // PUSH 100 (value)
		byte(MSTORE),       // MSTORE
		byte(PUSH), 1, 0,   // PUSH 0 (offset)
		byte(MLOAD),        // MLOAD
		byte(STOP),         // STOP
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vm := NewVM(context, state, config)
		_, _, err := vm.Execute(code, []byte{}, 1000000)
		if err != nil {
			b.Fatal(err)
		}
	}
}