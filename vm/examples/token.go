package examples

import (
	"math/big"

	"github.com/davidcanhelp/sedition/vm"
)

// TokenContract demonstrates a simple ERC20-like token
type TokenContract struct {
	manager *vm.ContractManager
	address string
	name    string
	symbol  string
	totalSupply *big.Int
}

// TokenBytecode contains bytecode for a simple token contract
// Just returns 1 (success) for all operations
var TokenBytecode = []byte{
	// Always return success (1)
	byte(vm.PUSH), 1, 1,   // Stack: [1]
	byte(vm.PUSH), 1, 0,   // Stack: [1, 0] - memory offset
	byte(vm.MSTORE),       // Store 1 in memory, Stack: []
	byte(vm.PUSH), 1, 32,  // Stack: [32] - return 32 bytes
	byte(vm.PUSH), 1, 0,   // Stack: [32, 0] - from memory offset 0
	byte(vm.RETURN),       // Return 1

	byte(vm.STOP), // Fallback
}

// NewTokenContract creates a new token contract
func NewTokenContract(manager *vm.ContractManager, creator, name, symbol string, totalSupply *big.Int) (*TokenContract, error) {
	contract, err := manager.Deploy(creator, TokenBytecode, big.NewInt(0), 2000000, []byte{})
	if err != nil {
		return nil, err
	}

	token := &TokenContract{
		manager:     manager,
		address:     contract.Address,
		name:        name,
		symbol:      symbol,
		totalSupply: new(big.Int).Set(totalSupply),
	}

	// Initialize creator balance
	token.setBalance(creator, totalSupply)

	// Store total supply at slot 0
	totalSupplySlot := make([]byte, 32)
	manager.SetStorageAt(contract.Address, totalSupplySlot, totalSupply.Bytes())

	return token, nil
}

// Transfer sends tokens from caller to recipient
func (tc *TokenContract) Transfer(caller, to string, amount *big.Int) (bool, error) {
	// Get sender balance
	senderBalance := tc.BalanceOf(caller)

	// Check if sender has enough balance
	if senderBalance.Cmp(amount) < 0 {
		return false, nil // Insufficient balance
	}

	// Get recipient balance
	recipientBalance := tc.BalanceOf(to)

	// Calculate new balances
	newSenderBalance := new(big.Int).Sub(senderBalance, amount)
	newRecipientBalance := new(big.Int).Add(recipientBalance, amount)

	// Update balances in storage
	tc.setBalance(caller, newSenderBalance)
	tc.setBalance(to, newRecipientBalance)

	return true, nil
}

// BalanceOf returns the balance of an address
func (tc *TokenContract) BalanceOf(address string) *big.Int {
	slot := tc.getBalanceSlot(address)
	value := tc.manager.GetStorageAt(tc.address, slot)
	return new(big.Int).SetBytes(value)
}

// setBalance sets the balance for an address
func (tc *TokenContract) setBalance(address string, balance *big.Int) {
	slot := tc.getBalanceSlot(address)
	tc.manager.SetStorageAt(tc.address, slot, balance.Bytes())
}

// getBalanceSlot calculates the storage slot for an address balance
func (tc *TokenContract) getBalanceSlot(address string) []byte {
	slot := make([]byte, 32)
	// Encode 1000000 as big-endian bytes
	slot[25] = 15   // 0x0F
	slot[26] = 66   // 0x42
	slot[27] = 64   // 0x40 (1000000 = 0x0F4240)

	// Add simple hash of address
	addressBytes := []byte(address)
	if len(addressBytes) > 0 {
		slot[31] = addressBytes[0]
	}
	return slot
}

// TotalSupply returns the total supply of tokens
func (tc *TokenContract) TotalSupply() *big.Int {
	// For simplicity, return the stored total supply
	return new(big.Int).Set(tc.totalSupply)
}

// GetAddress returns the contract address
func (tc *TokenContract) GetAddress() string {
	return tc.address
}

// GetName returns the token name
func (tc *TokenContract) GetName() string {
	return tc.name
}

// GetSymbol returns the token symbol
func (tc *TokenContract) GetSymbol() string {
	return tc.symbol
}