package vm

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"
	"sync"
)

// VM represents the virtual machine for executing smart contracts
type VM struct {
	mu sync.RWMutex

	// Execution context
	context   *Context
	state     *StateDB
	stack     *Stack
	memory    *Memory
	returnData []byte

	// Execution state
	pc          uint64 // Program counter
	gasUsed     uint64
	gasLimit    uint64
	stopped     bool
	returnError error

	// Configuration
	config *Config

	// Debugging
	debug      bool
	breakpoints map[uint64]bool
	trace      []TraceEntry
}

// Context provides the execution context for the VM
type Context struct {
	Origin      string // Transaction origin
	Caller      string // Direct caller
	Address     string // Current contract address
	Value       *big.Int // Value sent with call
	Input       []byte // Input data
	Code        []byte // Contract code
	CodeHash    []byte // Hash of contract code
	BlockHeight uint64 // Current block height
	Timestamp   int64 // Block timestamp
	GasPrice    *big.Int // Gas price
}

// Config contains VM configuration
type Config struct {
	EnableDebug     bool
	MaxStackDepth   int
	MaxMemorySize   int
	MaxCallDepth    int
	EnableGasMetering bool
}

// DefaultConfig returns default VM configuration
func DefaultConfig() *Config {
	return &Config{
		EnableDebug:       false,
		MaxStackDepth:     1024,
		MaxMemorySize:     1024 * 1024, // 1MB
		MaxCallDepth:      1024,
		EnableGasMetering: true,
	}
}

// TraceEntry represents a single execution trace entry
type TraceEntry struct {
	PC      uint64
	Op      OpCode
	Gas     uint64
	GasCost uint64
	Stack   []big.Int
	Memory  []byte
	Error   error
}

// NewVM creates a new VM instance
func NewVM(context *Context, state *StateDB, config *Config) *VM {
	if config == nil {
		config = DefaultConfig()
	}

	return &VM{
		context:     context,
		state:       state,
		stack:       NewStack(config.MaxStackDepth),
		memory:      NewMemory(),
		gasLimit:    1000000, // Default gas limit
		config:      config,
		breakpoints: make(map[uint64]bool),
		trace:       make([]TraceEntry, 0),
	}
}

// Execute runs the VM with the given code
func (vm *VM) Execute(code []byte, input []byte, gasLimit uint64) ([]byte, uint64, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	// Initialize execution
	vm.context.Code = code
	vm.context.Input = input
	vm.gasLimit = gasLimit
	vm.gasUsed = 0
	vm.pc = 0
	vm.stopped = false
	vm.returnData = nil
	vm.returnError = nil

	// Main execution loop
	for !vm.stopped && vm.pc < uint64(len(code)) {
		// Check gas
		if vm.config.EnableGasMetering && vm.gasUsed >= vm.gasLimit {
			return nil, vm.gasUsed, errors.New("out of gas")
		}

		// Get current instruction
		op := OpCode(code[vm.pc])

		// Calculate gas cost
		gasCost := GasCost[op]
		if vm.config.EnableGasMetering {
			if vm.gasUsed+gasCost > vm.gasLimit {
				return nil, vm.gasUsed, errors.New("out of gas")
			}
			vm.gasUsed += gasCost
		}

		// Debug trace
		if vm.config.EnableDebug {
			vm.addTrace(op, gasCost)
		}

		// Check breakpoint
		if vm.debug && vm.breakpoints[vm.pc] {
			// In production, this would pause execution for debugging
			fmt.Printf("Breakpoint at PC: %d, Op: %s\n", vm.pc, op.String())
		}

		// Execute operation
		if err := vm.executeOp(op); err != nil {
			return nil, vm.gasUsed, err
		}

		// Increment program counter if not a jump
		if !op.IsJump() {
			vm.pc++
		}
	}

	return vm.returnData, vm.gasUsed, vm.returnError
}

// executeOp executes a single operation
func (vm *VM) executeOp(op OpCode) error {
	switch op {
	// Stack operations
	case PUSH:
		return vm.opPush()
	case POP:
		return vm.opPop()
	case DUP:
		return vm.opDup()
	case SWAP:
		return vm.opSwap()

	// Arithmetic operations
	case ADD:
		return vm.opAdd()
	case SUB:
		return vm.opSub()
	case MUL:
		return vm.opMul()
	case DIV:
		return vm.opDiv()
	case MOD:
		return vm.opMod()
	case EXP:
		return vm.opExp()
	case NEG:
		return vm.opNeg()

	// Comparison operations
	case EQ:
		return vm.opEq()
	case NEQ:
		return vm.opNeq()
	case LT:
		return vm.opLt()
	case LTE:
		return vm.opLte()
	case GT:
		return vm.opGt()
	case GTE:
		return vm.opGte()

	// Bitwise operations
	case AND:
		return vm.opAnd()
	case OR:
		return vm.opOr()
	case XOR:
		return vm.opXor()
	case NOT:
		return vm.opNot()
	case SHL:
		return vm.opShl()
	case SHR:
		return vm.opShr()

	// Memory operations
	case MLOAD:
		return vm.opMLoad()
	case MSTORE:
		return vm.opMStore()
	case MSIZE:
		return vm.opMSize()

	// Storage operations
	case SLOAD:
		return vm.opSLoad()
	case SSTORE:
		return vm.opSStore()

	// Flow control
	case JUMP:
		return vm.opJump()
	case JUMPI:
		return vm.opJumpI()
	case JUMPDEST:
		return nil // No operation, just a marker
	case STOP:
		return vm.opStop()
	case RETURN:
		return vm.opReturn()
	case REVERT:
		return vm.opRevert()

	// System operations
	case ADDRESS:
		return vm.opAddress()
	case BALANCE:
		return vm.opBalance()
	case CALLER:
		return vm.opCaller()
	case CALLVALUE:
		return vm.opCallValue()
	case CALLDATALOAD:
		return vm.opCallDataLoad()
	case CALLDATASIZE:
		return vm.opCallDataSize()
	case TIMESTAMP:
		return vm.opTimestamp()
	case BLOCKHEIGHT:
		return vm.opBlockHeight()

	// Contract operations
	case CALL:
		return vm.opCall()
	case CREATE:
		return vm.opCreate()
	case SELFDESTRUCT:
		return vm.opSelfDestruct()

	// Logging operations
	case LOG0, LOG1, LOG2, LOG3, LOG4:
		return vm.opLog(op)

	// Cryptographic operations
	case SHA3:
		return vm.opSha3()
	case HASH256:
		return vm.opHash256()

	// Custom PoC operations
	case CONTRIBUTE:
		return vm.opContribute()
	case REPUTATION:
		return vm.opReputation()
	case STAKE:
		return vm.opStake()
	case VALIDATE:
		return vm.opValidate()

	default:
		return fmt.Errorf("unknown opcode: 0x%02x", op)
	}
}

// Stack operations
func (vm *VM) opPush() error {
	// Get the number of bytes to push
	vm.pc++
	if vm.pc >= uint64(len(vm.context.Code)) {
		return errors.New("PUSH: insufficient bytes")
	}

	size := vm.context.Code[vm.pc]
	if size > 32 {
		return errors.New("PUSH: size too large")
	}

	// Read the value
	vm.pc++
	end := vm.pc + uint64(size)
	if end > uint64(len(vm.context.Code)) {
		return errors.New("PUSH: insufficient bytes for value")
	}

	value := new(big.Int).SetBytes(vm.context.Code[vm.pc:end])
	vm.pc = end - 1 // -1 because pc will be incremented after

	return vm.stack.Push(value)
}

func (vm *VM) opPop() error {
	_, err := vm.stack.Pop()
	return err
}

func (vm *VM) opDup() error {
	return vm.stack.Dup(1)
}

func (vm *VM) opSwap() error {
	return vm.stack.Swap(1)
}

// Arithmetic operations
func (vm *VM) opAdd() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Add(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opSub() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Sub(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opMul() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Mul(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opDiv() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	if b.Sign() == 0 {
		return vm.stack.Push(new(big.Int))
	}

	result := new(big.Int).Div(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opMod() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	if b.Sign() == 0 {
		return vm.stack.Push(new(big.Int))
	}

	result := new(big.Int).Mod(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opExp() error {
	base, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	exp, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Exp(base, exp, nil)
	return vm.stack.Push(result)
}

func (vm *VM) opNeg() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Neg(a)
	return vm.stack.Push(result)
}

// Comparison operations
func (vm *VM) opEq() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := big.NewInt(0)
	if a.Cmp(b) == 0 {
		result.SetInt64(1)
	}
	return vm.stack.Push(result)
}

func (vm *VM) opNeq() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := big.NewInt(0)
	if a.Cmp(b) != 0 {
		result.SetInt64(1)
	}
	return vm.stack.Push(result)
}

func (vm *VM) opLt() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := big.NewInt(0)
	if a.Cmp(b) < 0 {
		result.SetInt64(1)
	}
	return vm.stack.Push(result)
}

func (vm *VM) opLte() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := big.NewInt(0)
	if a.Cmp(b) <= 0 {
		result.SetInt64(1)
	}
	return vm.stack.Push(result)
}

func (vm *VM) opGt() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := big.NewInt(0)
	if a.Cmp(b) > 0 {
		result.SetInt64(1)
	}
	return vm.stack.Push(result)
}

func (vm *VM) opGte() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := big.NewInt(0)
	if a.Cmp(b) >= 0 {
		result.SetInt64(1)
	}
	return vm.stack.Push(result)
}

// Bitwise operations
func (vm *VM) opAnd() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).And(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opOr() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Or(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opXor() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	b, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Xor(a, b)
	return vm.stack.Push(result)
}

func (vm *VM) opNot() error {
	a, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	result := new(big.Int).Not(a)
	return vm.stack.Push(result)
}

func (vm *VM) opShl() error {
	shift, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	value, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	if shift.Cmp(big.NewInt(256)) >= 0 {
		return vm.stack.Push(new(big.Int))
	}

	result := new(big.Int).Lsh(value, uint(shift.Uint64()))
	return vm.stack.Push(result)
}

func (vm *VM) opShr() error {
	shift, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	value, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	if shift.Cmp(big.NewInt(256)) >= 0 {
		return vm.stack.Push(new(big.Int))
	}

	result := new(big.Int).Rsh(value, uint(shift.Uint64()))
	return vm.stack.Push(result)
}

// Memory operations
func (vm *VM) opMLoad() error {
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	data := vm.memory.Load(offset.Uint64(), 32)
	value := new(big.Int).SetBytes(data)
	return vm.stack.Push(value)
}

func (vm *VM) opMStore() error {
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	value, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Convert to 32 bytes
	data := make([]byte, 32)
	value.FillBytes(data)

	vm.memory.Store(offset.Uint64(), data)
	return nil
}

func (vm *VM) opMSize() error {
	size := vm.memory.Size()
	return vm.stack.Push(big.NewInt(int64(size)))
}

// Storage operations
func (vm *VM) opSLoad() error {
	key, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	value := vm.state.GetState(vm.context.Address, key.Bytes())
	return vm.stack.Push(new(big.Int).SetBytes(value))
}

func (vm *VM) opSStore() error {
	key, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	value, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	vm.state.SetState(vm.context.Address, key.Bytes(), value.Bytes())
	return nil
}

// Flow control operations
func (vm *VM) opJump() error {
	dest, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	newPC := dest.Uint64()
	if newPC >= uint64(len(vm.context.Code)) {
		return errors.New("invalid jump destination")
	}

	// Check that destination is JUMPDEST
	if OpCode(vm.context.Code[newPC]) != JUMPDEST {
		return errors.New("jump to non-JUMPDEST")
	}

	vm.pc = newPC
	return nil
}

func (vm *VM) opJumpI() error {
	dest, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	cond, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	if cond.Sign() != 0 {
		newPC := dest.Uint64()
		if newPC >= uint64(len(vm.context.Code)) {
			return errors.New("invalid jump destination")
		}

		// Check that destination is JUMPDEST
		if OpCode(vm.context.Code[newPC]) != JUMPDEST {
			return errors.New("jump to non-JUMPDEST")
		}

		vm.pc = newPC
	}
	return nil
}

func (vm *VM) opStop() error {
	vm.stopped = true
	return nil
}

func (vm *VM) opReturn() error {
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	size, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	vm.returnData = vm.memory.Load(offset.Uint64(), size.Uint64())
	vm.stopped = true
	return nil
}

func (vm *VM) opRevert() error {
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	size, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	vm.returnData = vm.memory.Load(offset.Uint64(), size.Uint64())
	vm.stopped = true
	vm.returnError = errors.New("execution reverted")
	return nil
}

// System operations
func (vm *VM) opAddress() error {
	// Convert address string to bytes then to big.Int
	addressBytes := []byte(vm.context.Address)
	value := new(big.Int).SetBytes(addressBytes)
	return vm.stack.Push(value)
}

func (vm *VM) opBalance() error {
	addr, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	address := string(addr.Bytes())
	balance := vm.state.GetBalance(address)
	return vm.stack.Push(balance)
}

func (vm *VM) opCaller() error {
	callerBytes := []byte(vm.context.Caller)
	value := new(big.Int).SetBytes(callerBytes)
	return vm.stack.Push(value)
}

func (vm *VM) opCallValue() error {
	return vm.stack.Push(new(big.Int).Set(vm.context.Value))
}

func (vm *VM) opCallDataLoad() error {
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	off := offset.Uint64()
	data := make([]byte, 32)

	if off < uint64(len(vm.context.Input)) {
		copy(data, vm.context.Input[off:])
	}

	value := new(big.Int).SetBytes(data)
	return vm.stack.Push(value)
}

func (vm *VM) opCallDataSize() error {
	size := big.NewInt(int64(len(vm.context.Input)))
	return vm.stack.Push(size)
}

func (vm *VM) opTimestamp() error {
	return vm.stack.Push(big.NewInt(vm.context.Timestamp))
}

func (vm *VM) opBlockHeight() error {
	return vm.stack.Push(big.NewInt(int64(vm.context.BlockHeight)))
}

// Contract operations
func (vm *VM) opCall() error {
	// Simplified CALL implementation
	// In production, this would execute another contract

	// Pop arguments
	gas, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	addr, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	value, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	argsOffset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	argsSize, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	retOffset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	retSize, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Get call data
	callData := vm.memory.Load(argsOffset.Uint64(), argsSize.Uint64())

	// Execute call (simplified - just check balance for value transfer)
	address := string(addr.Bytes())
	if value.Sign() > 0 {
		callerBalance := vm.state.GetBalance(vm.context.Caller)
		if callerBalance.Cmp(value) < 0 {
			// Insufficient balance - call fails
			return vm.stack.Push(big.NewInt(0))
		}

		// Transfer value
		vm.state.SubBalance(vm.context.Caller, value)
		vm.state.AddBalance(address, value)
	}

	// Store empty return data for now
	vm.memory.Store(retOffset.Uint64(), make([]byte, retSize.Uint64()))

	// Success
	return vm.stack.Push(big.NewInt(1))

	_ = gas
	_ = callData
	return nil
}

func (vm *VM) opCreate() error {
	// Simplified CREATE implementation
	value, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	size, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Get init code
	initCode := vm.memory.Load(offset.Uint64(), size.Uint64())

	// Generate contract address (simplified)
	hash := sha256.Sum256(append([]byte(vm.context.Address), initCode...))
	contractAddr := fmt.Sprintf("0x%x", hash[:20])

	// Deploy contract (simplified - just store code)
	vm.state.SetCode(contractAddr, initCode)

	// Transfer value if any
	if value.Sign() > 0 {
		vm.state.SubBalance(vm.context.Caller, value)
		vm.state.AddBalance(contractAddr, value)
	}

	// Return contract address
	addrBytes := []byte(contractAddr)
	return vm.stack.Push(new(big.Int).SetBytes(addrBytes))
}

func (vm *VM) opSelfDestruct() error {
	beneficiary, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Transfer balance to beneficiary
	balance := vm.state.GetBalance(vm.context.Address)
	beneficiaryAddr := string(beneficiary.Bytes())
	vm.state.AddBalance(beneficiaryAddr, balance)
	vm.state.SetBalance(vm.context.Address, big.NewInt(0))

	// Mark for deletion
	vm.state.MarkForDeletion(vm.context.Address)

	vm.stopped = true
	return nil
}

// Logging operations
func (vm *VM) opLog(op OpCode) error {
	// Calculate number of topics based on opcode
	numTopics := int(op - LOG0)

	// Pop memory location and size
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	size, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Pop topics
	topics := make([][]byte, numTopics)
	for i := 0; i < numTopics; i++ {
		topic, err := vm.stack.Pop()
		if err != nil {
			return err
		}
		topics[i] = topic.Bytes()
	}

	// Get log data
	data := vm.memory.Load(offset.Uint64(), size.Uint64())

	// Emit log
	vm.state.AddLog(&Log{
		Address: vm.context.Address,
		Topics:  topics,
		Data:    data,
	})

	return nil
}

// Cryptographic operations
func (vm *VM) opSha3() error {
	offset, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	size, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	data := vm.memory.Load(offset.Uint64(), size.Uint64())
	hash := sha256.Sum256(data)

	return vm.stack.Push(new(big.Int).SetBytes(hash[:]))
}

func (vm *VM) opHash256() error {
	value, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	hash := sha256.Sum256(value.Bytes())
	return vm.stack.Push(new(big.Int).SetBytes(hash[:]))
}

// Custom PoC operations
func (vm *VM) opContribute() error {
	// Pop contribution parameters
	contributionType, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	amount, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Record contribution (simplified)
	vm.state.AddContribution(vm.context.Caller, contributionType.Uint64(), amount)

	// Return success
	return vm.stack.Push(big.NewInt(1))
}

func (vm *VM) opReputation() error {
	addr, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	address := string(addr.Bytes())
	reputation := vm.state.GetReputation(address)
	return vm.stack.Push(reputation)
}

func (vm *VM) opStake() error {
	amount, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Check balance
	balance := vm.state.GetBalance(vm.context.Caller)
	if balance.Cmp(amount) < 0 {
		return vm.stack.Push(big.NewInt(0))
	}

	// Update stake
	vm.state.AddStake(vm.context.Caller, amount)
	vm.state.SubBalance(vm.context.Caller, amount)

	return vm.stack.Push(big.NewInt(1))
}

func (vm *VM) opValidate() error {
	// Pop validation parameters
	blockHeight, err := vm.stack.Pop()
	if err != nil {
		return err
	}
	blockHash, err := vm.stack.Pop()
	if err != nil {
		return err
	}

	// Perform validation (simplified)
	isValid := vm.state.ValidateBlock(blockHeight.Uint64(), blockHash.Bytes())

	result := big.NewInt(0)
	if isValid {
		result.SetInt64(1)
	}
	return vm.stack.Push(result)
}

// Debugging and tracing
func (vm *VM) addTrace(op OpCode, gasCost uint64) {
	entry := TraceEntry{
		PC:      vm.pc,
		Op:      op,
		Gas:     vm.gasLimit - vm.gasUsed,
		GasCost: gasCost,
		Stack:   vm.stack.Data(),
		Memory:  vm.memory.Data(),
		Error:   nil,
	}
	vm.trace = append(vm.trace, entry)
}

// SetBreakpoint sets a breakpoint at the given PC
func (vm *VM) SetBreakpoint(pc uint64) {
	vm.breakpoints[pc] = true
}

// RemoveBreakpoint removes a breakpoint at the given PC
func (vm *VM) RemoveBreakpoint(pc uint64) {
	delete(vm.breakpoints, pc)
}

// GetTrace returns the execution trace
func (vm *VM) GetTrace() []TraceEntry {
	return vm.trace
}

// EnableDebug enables debug mode
func (vm *VM) EnableDebug() {
	vm.debug = true
	vm.config.EnableDebug = true
}

// DisableDebug disables debug mode
func (vm *VM) DisableDebug() {
	vm.debug = false
	vm.config.EnableDebug = false
}

// Helper function to convert uint64 to bytes
func uint64ToBytes(val uint64) []byte {
	bytes := make([]byte, 8)
	binary.BigEndian.PutUint64(bytes, val)
	return bytes
}