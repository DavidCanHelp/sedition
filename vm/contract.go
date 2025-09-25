package vm

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"strings"
	"time"
)

// Contract represents a smart contract
type Contract struct {
	Address      string    `json:"address"`
	Code         []byte    `json:"code"`
	CodeHash     []byte    `json:"code_hash"`
	Creator      string    `json:"creator"`
	CreationTime time.Time `json:"creation_time"`
	Balance      *big.Int  `json:"balance"`
	Nonce        uint64    `json:"nonce"`
	Destroyed    bool      `json:"destroyed"`

	// Metadata
	Name        string            `json:"name,omitempty"`
	Version     string            `json:"version,omitempty"`
	Description string            `json:"description,omitempty"`
	ABI         []ABIFunction     `json:"abi,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// ABIFunction represents a function in the contract ABI
type ABIFunction struct {
	Name     string      `json:"name"`
	Type     string      `json:"type"` // "function", "constructor", "fallback", "receive"
	Inputs   []ABIParam  `json:"inputs"`
	Outputs  []ABIParam  `json:"outputs"`
	Payable  bool        `json:"payable"`
	Constant bool        `json:"constant"`
	Selector []byte      `json:"selector"` // First 4 bytes of function signature hash
}

// ABIParam represents a parameter in the ABI
type ABIParam struct {
	Name    string `json:"name"`
	Type    string `json:"type"`
	Indexed bool   `json:"indexed,omitempty"`
}

// ContractManager manages smart contract deployment and execution
type ContractManager struct {
	state *StateDB
	vm    *VM
}

// NewContractManager creates a new contract manager
func NewContractManager(state *StateDB) *ContractManager {
	return &ContractManager{
		state: state,
	}
}

// Deploy deploys a new smart contract
func (cm *ContractManager) Deploy(creator string, code []byte, value *big.Int, gasLimit uint64, constructorArgs []byte) (*Contract, error) {
	// Generate contract address
	nonce := cm.state.GetNonce(creator)
	address := cm.generateContractAddress(creator, nonce)

	// Increment creator nonce
	cm.state.IncNonce(creator)

	// Check creator balance
	creatorBalance := cm.state.GetBalance(creator)
	if creatorBalance.Cmp(value) < 0 {
		return nil, errors.New("insufficient balance for contract deployment")
	}

	// Create contract
	contract := &Contract{
		Address:      address,
		Code:         make([]byte, len(code)),
		Creator:      creator,
		CreationTime: time.Now(),
		Balance:      new(big.Int).Set(value),
		Nonce:        0,
		Destroyed:    false,
		Metadata:     make(map[string]string),
	}
	copy(contract.Code, code)
	contract.CodeHash = cm.hashCode(code)

	// Store contract code in state
	cm.state.SetCode(address, code)
	cm.state.SetCodeHash(address, contract.CodeHash)

	// Transfer value if any
	if value.Sign() > 0 {
		cm.state.SubBalance(creator, value)
		cm.state.AddBalance(address, value)
	}

	// Execute constructor if present
	if len(constructorArgs) > 0 || cm.hasConstructor(code) {
		err := cm.executeConstructor(contract, constructorArgs, gasLimit)
		if err != nil {
			// Revert state changes on constructor failure
			cm.state.SetCode(address, nil)
			cm.state.SetBalance(address, new(big.Int))
			if value.Sign() > 0 {
				cm.state.AddBalance(creator, value)
			}
			return nil, fmt.Errorf("constructor execution failed: %w", err)
		}
	}

	return contract, nil
}

// Call executes a function call on a contract
func (cm *ContractManager) Call(caller, contractAddress string, input []byte, value *big.Int, gasLimit uint64) ([]byte, uint64, error) {
	// Check if contract exists
	code := cm.state.GetCode(contractAddress)
	if len(code) == 0 {
		return nil, 0, errors.New("contract not found")
	}

	// Check if contract is destroyed
	if cm.state.IsDeleted(contractAddress) {
		return nil, 0, errors.New("contract has been destroyed")
	}

	// Check caller balance for value transfer
	if value.Sign() > 0 {
		callerBalance := cm.state.GetBalance(caller)
		if callerBalance.Cmp(value) < 0 {
			return nil, 0, errors.New("insufficient balance for value transfer")
		}
	}

	// Create execution context
	context := &Context{
		Origin:      caller,
		Caller:      caller,
		Address:     contractAddress,
		Value:       value,
		Input:       input,
		Code:        code,
		CodeHash:    cm.state.GetCodeHash(contractAddress),
		BlockHeight: 0, // Would be set from blockchain state
		Timestamp:   time.Now().Unix(),
		GasPrice:    big.NewInt(1),
	}

	// Create VM and execute
	config := DefaultConfig()
	vm := NewVM(context, cm.state, config)

	// Take state snapshot for rollback
	cm.state.Snapshot()

	// Transfer value before execution
	if value.Sign() > 0 {
		cm.state.SubBalance(caller, value)
		cm.state.AddBalance(contractAddress, value)
	}

	// Execute contract
	returnData, gasUsed, err := vm.Execute(code, input, gasLimit)

	if err != nil {
		// Rollback on error
		cm.state.Rollback()
		return returnData, gasUsed, err
	}

	// Commit changes on success
	cm.state.CommitSnapshot()

	return returnData, gasUsed, nil
}

// StaticCall executes a read-only call that doesn't modify state
func (cm *ContractManager) StaticCall(caller, contractAddress string, input []byte, gasLimit uint64) ([]byte, uint64, error) {
	// Check if contract exists
	code := cm.state.GetCode(contractAddress)
	if len(code) == 0 {
		return nil, 0, errors.New("contract not found")
	}

	// Create execution context with zero value
	context := &Context{
		Origin:      caller,
		Caller:      caller,
		Address:     contractAddress,
		Value:       big.NewInt(0),
		Input:       input,
		Code:        code,
		CodeHash:    cm.state.GetCodeHash(contractAddress),
		BlockHeight: 0,
		Timestamp:   time.Now().Unix(),
		GasPrice:    big.NewInt(1),
	}

	// Create VM with read-only state
	config := DefaultConfig()
	stateCopy := cm.state.Copy()
	vm := NewVM(context, stateCopy, config)

	// Execute without modifying original state
	return vm.Execute(code, input, gasLimit)
}

// GetContract returns contract information
func (cm *ContractManager) GetContract(address string) (*Contract, error) {
	code := cm.state.GetCode(address)
	if len(code) == 0 {
		return nil, errors.New("contract not found")
	}

	return &Contract{
		Address:   address,
		Code:      code,
		CodeHash:  cm.state.GetCodeHash(address),
		Balance:   cm.state.GetBalance(address),
		Nonce:     cm.state.GetNonce(address),
		Destroyed: cm.state.IsDeleted(address),
	}, nil
}

// DestroyContract marks a contract for destruction
func (cm *ContractManager) DestroyContract(address, beneficiary string) error {
	// Check if contract exists
	if !cm.state.Exists(address) {
		return errors.New("contract not found")
	}

	// Transfer balance to beneficiary
	balance := cm.state.GetBalance(address)
	if balance.Sign() > 0 {
		cm.state.AddBalance(beneficiary, balance)
		cm.state.SetBalance(address, new(big.Int))
	}

	// Mark for deletion
	cm.state.MarkForDeletion(address)

	return nil
}

// EstimateGas estimates the gas required for a transaction
func (cm *ContractManager) EstimateGas(caller, contractAddress string, input []byte, value *big.Int) (uint64, error) {
	// Use a high gas limit for estimation
	estimateLimit := uint64(10000000)

	// Create a copy of state for estimation
	stateCopy := cm.state.Copy()

	// Check if contract exists
	code := stateCopy.GetCode(contractAddress)
	if len(code) == 0 {
		return 0, errors.New("contract not found")
	}

	// Create execution context
	context := &Context{
		Origin:      caller,
		Caller:      caller,
		Address:     contractAddress,
		Value:       value,
		Input:       input,
		Code:        code,
		CodeHash:    stateCopy.GetCodeHash(contractAddress),
		BlockHeight: 0,
		Timestamp:   time.Now().Unix(),
		GasPrice:    big.NewInt(1),
	}

	// Create VM and execute for estimation
	config := DefaultConfig()
	vm := NewVM(context, stateCopy, config)

	_, gasUsed, err := vm.Execute(code, input, estimateLimit)
	if err != nil {
		return 0, fmt.Errorf("gas estimation failed: %w", err)
	}

	// Add some buffer (10%) to the estimate
	return gasUsed + (gasUsed / 10), nil
}

// Helper functions

func (cm *ContractManager) generateContractAddress(creator string, nonce uint64) string {
	// Generate deterministic contract address based on creator and nonce
	data := fmt.Sprintf("%s:%d", creator, nonce)
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("0x%x", hash[:20])
}

func (cm *ContractManager) hashCode(code []byte) []byte {
	hash := sha256.Sum256(code)
	return hash[:]
}

func (cm *ContractManager) hasConstructor(code []byte) bool {
	// Simplified check - in practice, would parse the bytecode
	// to determine if there's a constructor
	return false
}

func (cm *ContractManager) executeConstructor(contract *Contract, args []byte, gasLimit uint64) error {
	// Create execution context for constructor
	context := &Context{
		Origin:      contract.Creator,
		Caller:      contract.Creator,
		Address:     contract.Address,
		Value:       contract.Balance,
		Input:       args,
		Code:        contract.Code,
		CodeHash:    contract.CodeHash,
		BlockHeight: 0,
		Timestamp:   time.Now().Unix(),
		GasPrice:    big.NewInt(1),
	}

	// Create VM and execute constructor
	config := DefaultConfig()
	vm := NewVM(context, cm.state, config)

	_, _, err := vm.Execute(contract.Code, args, gasLimit)
	return err
}

// Contract compilation and ABI functions

// CompileContract compiles a contract from source code (simplified)
func (cm *ContractManager) CompileContract(source string) ([]byte, []ABIFunction, error) {
	// This is a simplified compiler - in practice, you'd use a real compiler
	// For now, return empty bytecode and ABI
	return []byte{}, []ABIFunction{}, errors.New("contract compilation not implemented")
}

// ParseABI parses contract ABI from JSON
func (cm *ContractManager) ParseABI(abiJSON string) ([]ABIFunction, error) {
	// Simplified ABI parsing
	return []ABIFunction{}, errors.New("ABI parsing not implemented")
}

// EncodeCallData encodes function call data
func (cm *ContractManager) EncodeCallData(function ABIFunction, args []interface{}) ([]byte, error) {
	// Simplified encoding - in practice, would implement full ABI encoding
	return []byte{}, errors.New("call data encoding not implemented")
}

// DecodeReturnData decodes function return data
func (cm *ContractManager) DecodeReturnData(function ABIFunction, data []byte) ([]interface{}, error) {
	// Simplified decoding - in practice, would implement full ABI decoding
	return []interface{}{}, errors.New("return data decoding not implemented")
}

// GetFunctionSignature calculates function signature hash
func GetFunctionSignature(name string, inputs []ABIParam) []byte {
	// Build function signature
	sig := name + "("
	for i, input := range inputs {
		if i > 0 {
			sig += ","
		}
		sig += input.Type
	}
	sig += ")"

	// Hash and return first 4 bytes
	hash := sha256.Sum256([]byte(sig))
	return hash[:4]
}

// CreateContract creates a contract instance from deployment
type ContractDeployment struct {
	ByteCode         []byte
	ABI              []ABIFunction
	ConstructorArgs  []interface{}
	Value            *big.Int
	GasLimit         uint64

	// Metadata
	Name        string
	Version     string
	Description string
	Metadata    map[string]string
}

// DeployFromSpec deploys a contract from deployment specification
func (cm *ContractManager) DeployFromSpec(creator string, spec *ContractDeployment) (*Contract, error) {
	// Encode constructor arguments
	var encodedArgs []byte
	if len(spec.ConstructorArgs) > 0 {
		// In practice, would properly encode the arguments
		encodedArgs = []byte{} // Simplified
	}

	// Deploy contract
	contract, err := cm.Deploy(creator, spec.ByteCode, spec.Value, spec.GasLimit, encodedArgs)
	if err != nil {
		return nil, err
	}

	// Set metadata
	if spec.Name != "" {
		contract.Name = spec.Name
	}
	if spec.Version != "" {
		contract.Version = spec.Version
	}
	if spec.Description != "" {
		contract.Description = spec.Description
	}
	contract.ABI = spec.ABI

	// Set custom metadata
	if spec.Metadata != nil {
		for key, value := range spec.Metadata {
			contract.Metadata[key] = value
		}
	}

	return contract, nil
}

// Event handling

// Event represents a contract event
type Event struct {
	Address string
	Topics  []string
	Data    []byte
	Name    string
	Args    map[string]interface{}
}

// GetEvents returns events emitted by a contract
func (cm *ContractManager) GetEvents(contractAddress string) ([]*Event, error) {
	logs := cm.state.GetLogs()
	events := make([]*Event, 0)

	for _, log := range logs {
		if log.Address == contractAddress {
			event := &Event{
				Address: log.Address,
				Topics:  make([]string, len(log.Topics)),
				Data:    log.Data,
				Args:    make(map[string]interface{}),
			}

			for i, topic := range log.Topics {
				event.Topics[i] = hex.EncodeToString(topic)
			}

			events = append(events, event)
		}
	}

	return events, nil
}

// Contract introspection

// GetStorageAt returns storage value at a specific key
func (cm *ContractManager) GetStorageAt(contractAddress string, key []byte) []byte {
	return cm.state.GetState(contractAddress, key)
}

// SetStorageAt sets storage value at a specific key (for testing)
func (cm *ContractManager) SetStorageAt(contractAddress string, key, value []byte) {
	cm.state.SetState(contractAddress, key, value)
}

// GetReputation returns the reputation of an address
func (cm *ContractManager) GetReputation(address string) *big.Int {
	return cm.state.GetReputation(address)
}

// GetCodeAt returns the contract code
func (cm *ContractManager) GetCodeAt(contractAddress string) []byte {
	return cm.state.GetCode(contractAddress)
}

// IsContract returns true if the address contains contract code
func (cm *ContractManager) IsContract(address string) bool {
	code := cm.state.GetCode(address)
	return len(code) > 0
}

// GetBalance returns the balance of a contract or account
func (cm *ContractManager) GetBalance(address string) *big.Int {
	return cm.state.GetBalance(address)
}

// Contract registry for tracking deployed contracts
type ContractRegistry struct {
	contracts map[string]*Contract
}

// NewContractRegistry creates a new contract registry
func NewContractRegistry() *ContractRegistry {
	return &ContractRegistry{
		contracts: make(map[string]*Contract),
	}
}

// Register registers a deployed contract
func (cr *ContractRegistry) Register(contract *Contract) {
	cr.contracts[contract.Address] = contract
}

// Get returns a registered contract
func (cr *ContractRegistry) Get(address string) (*Contract, bool) {
	contract, exists := cr.contracts[address]
	return contract, exists
}

// List returns all registered contracts
func (cr *ContractRegistry) List() []*Contract {
	contracts := make([]*Contract, 0, len(cr.contracts))
	for _, contract := range cr.contracts {
		contracts = append(contracts, contract)
	}
	return contracts
}

// FindByName finds contracts by name
func (cr *ContractRegistry) FindByName(name string) []*Contract {
	contracts := make([]*Contract, 0)
	for _, contract := range cr.contracts {
		if strings.EqualFold(contract.Name, name) {
			contracts = append(contracts, contract)
		}
	}
	return contracts
}

// Remove removes a contract from registry
func (cr *ContractRegistry) Remove(address string) {
	delete(cr.contracts, address)
}