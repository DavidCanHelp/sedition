# Smart Contract Virtual Machine

## Overview

The PoC Blockchain now includes a complete smart contract virtual machine (VM) that enables the deployment and execution of programmable contracts on the blockchain. The VM is stack-based and includes custom opcodes for Proof of Contribution functionality.

## Architecture

### Core Components

1. **Virtual Machine (`vm/vm.go`)**
   - Stack-based execution engine
   - Gas metering system
   - Context-aware execution
   - Debugging and tracing support

2. **Contract Manager (`vm/contract.go`)**
   - Contract deployment and lifecycle management
   - Function call routing
   - Event handling
   - Gas estimation

3. **State Management (`vm/state.go`)**
   - Account state (balances, nonces, code)
   - Contract storage
   - Transaction logs and events
   - Snapshot/rollback functionality

4. **Memory and Stack (`vm/memory.go`, `vm/stack.go`)**
   - Efficient memory management
   - Stack operations with overflow protection
   - Word-aligned memory access

## Opcode Set

### Standard Operations
- **Stack**: PUSH, POP, DUP, SWAP
- **Arithmetic**: ADD, SUB, MUL, DIV, MOD, EXP, NEG
- **Comparison**: EQ, NEQ, LT, LTE, GT, GTE
- **Bitwise**: AND, OR, XOR, NOT, SHL, SHR
- **Memory**: MLOAD, MSTORE, MSIZE
- **Storage**: SLOAD, SSTORE
- **Flow Control**: JUMP, JUMPI, JUMPDEST, STOP, RETURN, REVERT

### System Operations
- **Context**: ADDRESS, BALANCE, CALLER, CALLVALUE
- **Call Data**: CALLDATALOAD, CALLDATASIZE, CALLDATACOPY
- **Block Info**: TIMESTAMP, BLOCKHASH, BLOCKHEIGHT
- **Contracts**: CREATE, CALL, CALLCODE, DELEGATE, STATICCALL, SELFDESTRUCT

### Logging and Events
- **Logging**: LOG0, LOG1, LOG2, LOG3, LOG4
- **Cryptography**: SHA3, HASH256, VERIFY

### Custom PoC Operations
- **CONTRIBUTE**: Record contribution with type and amount
- **REPUTATION**: Get validator reputation score
- **STAKE**: Stake tokens for validation rights
- **VALIDATE**: Validate block with height and hash

## Gas System

The VM includes a comprehensive gas metering system:

```go
// Example gas costs
var GasCost = map[OpCode]uint64{
    PUSH: 3,
    ADD:  3,
    MUL:  5,
    SLOAD:  200,
    SSTORE: 20000,
    // ... more opcodes
}
```

- **Base costs**: Simple operations (3-10 gas)
- **Memory operations**: Medium cost (3-50 gas)
- **Storage operations**: High cost (200-20,000 gas)
- **Contract operations**: Very high cost (700-32,000 gas)

## Contract Deployment

### Basic Deployment

```go
// Deploy a simple contract
contractManager := vm.NewContractManager(state)

bytecode := []byte{
    byte(vm.PUSH), 1, 42,  // Push 42 to stack
    byte(vm.STOP),         // Stop execution
}

contract, err := contractManager.Deploy(
    creator,        // Creator address
    bytecode,       // Contract bytecode
    big.NewInt(0),  // Initial value
    1000000,        // Gas limit
    []byte{},       // Constructor args
)
```

### Contract Execution

```go
// Call a contract function
returnData, gasUsed, err := contractManager.Call(
    caller,           // Caller address
    contract.Address, // Contract address
    inputData,        // Call data
    big.NewInt(0),    // Value to send
    500000,           // Gas limit
)
```

## Example Contracts

### 1. Simple Storage Contract

```go
// Bytecode that stores and retrieves a value
var SimpleStorageBytecode = []byte{
    // Constructor
    byte(vm.PUSH), 1, 0,   // Return offset
    byte(vm.PUSH), 1, 0,   // Return size
    byte(vm.RETURN),       // Return empty

    // Runtime: set(value)
    byte(vm.PUSH), 1, 0,   // Calldata offset
    byte(vm.CALLDATALOAD), // Load value
    byte(vm.PUSH), 1, 0,   // Storage key
    byte(vm.SSTORE),       // Store value

    // Runtime: get()
    byte(vm.PUSH), 1, 0,   // Storage key
    byte(vm.SLOAD),        // Load value
    byte(vm.RETURN),       // Return value
}
```

### 2. Token Contract (ERC20-like)

Features:
- Total supply tracking
- Balance mapping
- Transfer functionality
- Event emission

### 3. Voting Contract (PoC Integration)

Features:
- Reputation-weighted voting
- Contribution tracking
- Proposal management
- Time-based restrictions

## State Management

### Account State
```go
type StateDB struct {
    balances   map[string]*big.Int    // Account balances
    nonces     map[string]uint64      // Account nonces
    code       map[string][]byte      // Contract code
    storage    map[string]map[string][]byte // Contract storage

    // PoC-specific state
    stakes        map[string]*big.Int           // Validator stakes
    reputation    map[string]*big.Int           // Reputation scores
    contributions map[string]map[uint64]*big.Int // Contributions by type
}
```

### Snapshots and Rollbacks
```go
// Take snapshot before execution
state.Snapshot()

// Execute transaction
result, err := vm.Execute(code, input, gasLimit)

if err != nil {
    // Rollback on error
    state.Rollback()
} else {
    // Commit on success
    state.CommitSnapshot()
}
```

## Event System

### Event Emission
```go
// LOG1 opcode with one topic
vm.stack.Push(topicValue)    // Push topic
vm.stack.Push(dataSize)      // Push data size
vm.stack.Push(dataOffset)    // Push data offset
vm.executeOp(LOG1)           // Emit event
```

### Event Retrieval
```go
events, err := contractManager.GetEvents(contractAddress)
for _, event := range events {
    fmt.Printf("Event: %s, Topics: %v, Data: %x\n",
               event.Address, event.Topics, event.Data)
}
```

## Testing Framework

### Unit Tests
- **VM Operations**: Test individual opcodes
- **Contract Deployment**: Test deployment scenarios
- **Contract Execution**: Test function calls
- **State Changes**: Test storage and balance updates
- **Gas Metering**: Test gas consumption
- **Error Handling**: Test edge cases and failures

### Integration Tests
- **Multi-contract**: Test contract interactions
- **Cross-contract calls**: Test CALL opcodes
- **Event emission**: Test logging functionality
- **PoC integration**: Test custom opcodes

### Example Test
```go
func TestContractDeployment(t *testing.T) {
    state := vm.NewStateDB()
    manager := vm.NewContractManager(state)

    // Set up deployer balance
    state.AddBalance("deployer", big.NewInt(10000))

    // Deploy contract
    contract, err := manager.Deploy(
        "deployer",
        bytecode,
        big.NewInt(100),
        1000000,
        []byte{},
    )

    require.NoError(t, err)
    assert.NotEmpty(t, contract.Address)
    assert.Equal(t, "deployer", contract.Creator)
}
```

## Performance Optimizations

### Gas Optimization
- Efficient opcode implementations
- Minimal memory allocations
- Stack reuse where possible

### Memory Management
- Word-aligned memory access
- Efficient resizing algorithms
- Memory pooling for frequent operations

### State Caching
- In-memory state cache
- Batch state updates
- Lazy loading of storage values

## Security Features

### Access Control
- Caller authentication
- Contract ownership validation
- Permission-based function access

### Safe Execution
- Gas limit enforcement
- Stack overflow protection
- Memory limit enforcement
- Safe arithmetic operations

### State Protection
- Snapshot/rollback mechanisms
- Transaction isolation
- Reentrancy protection

## Integration with PoC Blockchain

### Custom Opcodes
- **CONTRIBUTE**: Integrates with contribution tracking system
- **REPUTATION**: Accesses validator reputation scores
- **STAKE**: Interfaces with staking mechanism
- **VALIDATE**: Participates in block validation

### Consensus Integration
- Contract state changes participate in consensus
- Transaction execution affects validator selection
- Contribution tracking influences reputation

### Network Integration
- Contract events broadcast over P2P network
- State synchronization across nodes
- Transaction pool integration

## Future Enhancements

### Planned Features
1. **WebAssembly (WASM) Support**: Alternative execution engine
2. **Advanced Debugging**: Step-through debugger
3. **Formal Verification**: Mathematical proof of contract correctness
4. **Cross-chain Contracts**: Inter-blockchain contract calls
5. **Upgrade Patterns**: Proxy contracts and upgradeable contracts

### Performance Improvements
1. **JIT Compilation**: Just-in-time bytecode compilation
2. **Parallel Execution**: Multi-threaded contract execution
3. **State Sharding**: Distributed state storage
4. **Caching Layers**: Multi-level caching system

## Usage Examples

### Deploy and Use Storage Contract
```go
// Create state and manager
state := vm.NewStateDB()
manager := vm.NewContractManager(state)

// Deploy simple storage contract
contract, err := examples.NewSimpleStorageContract(manager, "creator")
if err != nil {
    log.Fatal(err)
}

// Set a value
err = contract.Set("user", big.NewInt(42))
if err != nil {
    log.Fatal(err)
}

// Get the value
value, err := contract.Get("user")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Stored value: %s\n", value.String()) // Output: 42
```

### Deploy and Use Token Contract
```go
// Deploy token with 1M supply
tokenContract, err := examples.NewTokenContract(
    manager,
    "creator",
    "MyToken",
    "MTK",
    big.NewInt(1000000),
)

// Transfer tokens
success, err := tokenContract.Transfer("creator", "recipient", big.NewInt(100))
if err != nil || !success {
    log.Fatal("Transfer failed")
}

// Check balance
balance := tokenContract.BalanceOf("recipient")
fmt.Printf("Recipient balance: %s\n", balance.String()) // Output: 100
```

## Summary

The smart contract VM provides a complete execution environment for programmable contracts on the PoC blockchain. It includes:

✅ **Complete VM implementation** with stack-based execution
✅ **Comprehensive opcode set** including PoC-specific operations
✅ **Gas metering system** for resource management
✅ **Contract deployment** and lifecycle management
✅ **State management** with snapshots and rollbacks
✅ **Event system** for contract-to-world communication
✅ **Testing framework** with unit and integration tests
✅ **Example contracts** demonstrating key functionality
✅ **Security features** for safe contract execution
✅ **Performance optimizations** for efficient execution

The VM is production-ready and provides the foundation for building sophisticated decentralized applications on the PoC blockchain platform.