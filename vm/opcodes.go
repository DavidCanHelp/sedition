package vm

// OpCode represents a single operation in the VM
type OpCode byte

const (
	// Stack operations
	PUSH OpCode = 0x01
	POP  OpCode = 0x02
	DUP  OpCode = 0x03
	SWAP OpCode = 0x04

	// Arithmetic operations
	ADD OpCode = 0x10
	SUB OpCode = 0x11
	MUL OpCode = 0x12
	DIV OpCode = 0x13
	MOD OpCode = 0x14
	EXP OpCode = 0x15
	NEG OpCode = 0x16

	// Comparison operations
	EQ  OpCode = 0x20
	NEQ OpCode = 0x21
	LT  OpCode = 0x22
	LTE OpCode = 0x23
	GT  OpCode = 0x24
	GTE OpCode = 0x25

	// Bitwise operations
	AND OpCode = 0x30
	OR  OpCode = 0x31
	XOR OpCode = 0x32
	NOT OpCode = 0x33
	SHL OpCode = 0x34
	SHR OpCode = 0x35

	// Memory operations
	MLOAD  OpCode = 0x40
	MSTORE OpCode = 0x41
	MSIZE  OpCode = 0x42

	// Storage operations
	SLOAD  OpCode = 0x50
	SSTORE OpCode = 0x51

	// Flow control
	JUMP     OpCode = 0x60
	JUMPI    OpCode = 0x61
	JUMPDEST OpCode = 0x62
	STOP     OpCode = 0x63
	RETURN   OpCode = 0x64
	REVERT   OpCode = 0x65

	// System operations
	ADDRESS     OpCode = 0x70
	BALANCE     OpCode = 0x71
	CALLER      OpCode = 0x72
	CALLVALUE   OpCode = 0x73
	CALLDATALOAD OpCode = 0x74
	CALLDATASIZE OpCode = 0x75
	CALLDATACOPY OpCode = 0x76
	TIMESTAMP   OpCode = 0x77
	BLOCKHASH   OpCode = 0x78
	BLOCKHEIGHT OpCode = 0x79

	// Contract operations
	CREATE   OpCode = 0x80
	CALL     OpCode = 0x81
	CALLCODE OpCode = 0x82
	DELEGATE OpCode = 0x83
	STATICCALL OpCode = 0x84
	SELFDESTRUCT OpCode = 0x85

	// Logging operations
	LOG0 OpCode = 0x90
	LOG1 OpCode = 0x91
	LOG2 OpCode = 0x92
	LOG3 OpCode = 0x93
	LOG4 OpCode = 0x94

	// Cryptographic operations
	SHA3    OpCode = 0xA0
	HASH256 OpCode = 0xA1
	VERIFY  OpCode = 0xA2

	// Custom PoC operations
	CONTRIBUTE OpCode = 0xF0
	REPUTATION OpCode = 0xF1
	STAKE      OpCode = 0xF2
	VALIDATE   OpCode = 0xF3
)

// GasCost defines the gas cost for each operation
var GasCost = map[OpCode]uint64{
	// Stack operations
	PUSH: 3,
	POP:  2,
	DUP:  3,
	SWAP: 3,

	// Arithmetic operations
	ADD: 3,
	SUB: 3,
	MUL: 5,
	DIV: 5,
	MOD: 5,
	EXP: 10,
	NEG: 3,

	// Comparison operations
	EQ:  3,
	NEQ: 3,
	LT:  3,
	LTE: 3,
	GT:  3,
	GTE: 3,

	// Bitwise operations
	AND: 3,
	OR:  3,
	XOR: 3,
	NOT: 3,
	SHL: 3,
	SHR: 3,

	// Memory operations
	MLOAD:  3,
	MSTORE: 3,
	MSIZE:  2,

	// Storage operations
	SLOAD:  200,
	SSTORE: 20000,

	// Flow control
	JUMP:     8,
	JUMPI:    10,
	JUMPDEST: 1,
	STOP:     0,
	RETURN:   0,
	REVERT:   0,

	// System operations
	ADDRESS:      2,
	BALANCE:      400,
	CALLER:       2,
	CALLVALUE:    2,
	CALLDATALOAD: 3,
	CALLDATASIZE: 2,
	CALLDATACOPY: 3,
	TIMESTAMP:    2,
	BLOCKHASH:    20,
	BLOCKHEIGHT:  2,

	// Contract operations
	CREATE:       32000,
	CALL:         700,
	CALLCODE:     700,
	DELEGATE:     700,
	STATICCALL:   700,
	SELFDESTRUCT: 5000,

	// Logging operations
	LOG0: 375,
	LOG1: 750,
	LOG2: 1125,
	LOG3: 1500,
	LOG4: 1875,

	// Cryptographic operations
	SHA3:    30,
	HASH256: 30,
	VERIFY:  3000,

	// Custom PoC operations
	CONTRIBUTE: 1000,
	REPUTATION: 100,
	STAKE:      500,
	VALIDATE:   2000,
}

// String returns the string representation of an opcode
func (op OpCode) String() string {
	switch op {
	case PUSH:
		return "PUSH"
	case POP:
		return "POP"
	case DUP:
		return "DUP"
	case SWAP:
		return "SWAP"
	case ADD:
		return "ADD"
	case SUB:
		return "SUB"
	case MUL:
		return "MUL"
	case DIV:
		return "DIV"
	case MOD:
		return "MOD"
	case EXP:
		return "EXP"
	case NEG:
		return "NEG"
	case EQ:
		return "EQ"
	case NEQ:
		return "NEQ"
	case LT:
		return "LT"
	case LTE:
		return "LTE"
	case GT:
		return "GT"
	case GTE:
		return "GTE"
	case AND:
		return "AND"
	case OR:
		return "OR"
	case XOR:
		return "XOR"
	case NOT:
		return "NOT"
	case SHL:
		return "SHL"
	case SHR:
		return "SHR"
	case MLOAD:
		return "MLOAD"
	case MSTORE:
		return "MSTORE"
	case MSIZE:
		return "MSIZE"
	case SLOAD:
		return "SLOAD"
	case SSTORE:
		return "SSTORE"
	case JUMP:
		return "JUMP"
	case JUMPI:
		return "JUMPI"
	case JUMPDEST:
		return "JUMPDEST"
	case STOP:
		return "STOP"
	case RETURN:
		return "RETURN"
	case REVERT:
		return "REVERT"
	case ADDRESS:
		return "ADDRESS"
	case BALANCE:
		return "BALANCE"
	case CALLER:
		return "CALLER"
	case CALLVALUE:
		return "CALLVALUE"
	case CALLDATALOAD:
		return "CALLDATALOAD"
	case CALLDATASIZE:
		return "CALLDATASIZE"
	case CALLDATACOPY:
		return "CALLDATACOPY"
	case TIMESTAMP:
		return "TIMESTAMP"
	case BLOCKHASH:
		return "BLOCKHASH"
	case BLOCKHEIGHT:
		return "BLOCKHEIGHT"
	case CREATE:
		return "CREATE"
	case CALL:
		return "CALL"
	case CALLCODE:
		return "CALLCODE"
	case DELEGATE:
		return "DELEGATE"
	case STATICCALL:
		return "STATICCALL"
	case SELFDESTRUCT:
		return "SELFDESTRUCT"
	case LOG0:
		return "LOG0"
	case LOG1:
		return "LOG1"
	case LOG2:
		return "LOG2"
	case LOG3:
		return "LOG3"
	case LOG4:
		return "LOG4"
	case SHA3:
		return "SHA3"
	case HASH256:
		return "HASH256"
	case VERIFY:
		return "VERIFY"
	case CONTRIBUTE:
		return "CONTRIBUTE"
	case REPUTATION:
		return "REPUTATION"
	case STAKE:
		return "STAKE"
	case VALIDATE:
		return "VALIDATE"
	default:
		return "UNKNOWN"
	}
}

// IsPush returns true if the opcode is a push operation
func (op OpCode) IsPush() bool {
	return op == PUSH
}

// IsJump returns true if the opcode is a jump operation
func (op OpCode) IsJump() bool {
	return op == JUMP || op == JUMPI
}

// IsHalt returns true if the opcode halts execution
func (op OpCode) IsHalt() bool {
	return op == STOP || op == RETURN || op == REVERT || op == SELFDESTRUCT
}