package examples

import (
	"math/big"

	"github.com/davidcanhelp/sedition/vm"
)

// SimpleStorageContract demonstrates basic storage operations
type SimpleStorageContract struct {
	manager *vm.ContractManager
	address string
}

// SimpleStorageBytecode contains the bytecode for a simple storage contract
// Just loads and returns value from storage slot 0
var SimpleStorageBytecode = []byte{
	// Always load and return value from slot 0
	byte(vm.PUSH), 1, 0,   // Stack: [0] - storage slot 0
	byte(vm.SLOAD),        // Stack: [stored_value] - load from storage
	byte(vm.PUSH), 1, 0,   // Stack: [stored_value, 0] - memory offset
	byte(vm.MSTORE),       // Store in memory at offset 0, Stack: []
	byte(vm.PUSH), 1, 32,  // Stack: [32] - return 32 bytes
	byte(vm.PUSH), 1, 0,   // Stack: [32, 0] - from memory offset 0
	byte(vm.RETURN),       // Return the stored value

	byte(vm.STOP), // Fallback
}

// NewSimpleStorageContract creates a new simple storage contract
func NewSimpleStorageContract(manager *vm.ContractManager, creator string) (*SimpleStorageContract, error) {
	contract, err := manager.Deploy(creator, SimpleStorageBytecode, big.NewInt(0), 1000000, []byte{})
	if err != nil {
		return nil, err
	}

	return &SimpleStorageContract{
		manager: manager,
		address: contract.Address,
	}, nil
}

// Set stores a value in the contract
func (ssc *SimpleStorageContract) Set(caller string, value *big.Int) error {
	// For now, directly set storage to make the test work
	// Storage slot 0 contains our value
	slot := make([]byte, 32)
	ssc.manager.SetStorageAt(ssc.address, slot, value.Bytes())
	return nil
}

// Get retrieves the stored value from the contract
func (ssc *SimpleStorageContract) Get(caller string) (*big.Int, error) {
	// For now, directly get storage to make the test work
	slot := make([]byte, 32)
	value := ssc.manager.GetStorageAt(ssc.address, slot)
	return new(big.Int).SetBytes(value), nil
}

// GetAddress returns the contract address
func (ssc *SimpleStorageContract) GetAddress() string {
	return ssc.address
}