package vm

import (
	"errors"
	"fmt"
	"math/big"
)

// Stack represents the VM stack
type Stack struct {
	data     []*big.Int
	maxDepth int
}

// NewStack creates a new stack
func NewStack(maxDepth int) *Stack {
	return &Stack{
		data:     make([]*big.Int, 0, 1024),
		maxDepth: maxDepth,
	}
}

// Push adds a value to the stack
func (s *Stack) Push(value *big.Int) error {
	if len(s.data) >= s.maxDepth {
		return errors.New("stack overflow")
	}

	// Make a copy to avoid external modifications
	s.data = append(s.data, new(big.Int).Set(value))
	return nil
}

// Pop removes and returns the top value from the stack
func (s *Stack) Pop() (*big.Int, error) {
	if len(s.data) == 0 {
		return nil, errors.New("stack underflow")
	}

	value := s.data[len(s.data)-1]
	s.data = s.data[:len(s.data)-1]
	return value, nil
}

// Peek returns the value at position n from the top (0 = top)
func (s *Stack) Peek(n int) (*big.Int, error) {
	if n < 0 {
		return nil, errors.New("negative peek position")
	}

	index := len(s.data) - 1 - n
	if index < 0 {
		return nil, errors.New("peek underflow")
	}

	return new(big.Int).Set(s.data[index]), nil
}

// Dup duplicates the nth value from the top
func (s *Stack) Dup(n int) error {
	value, err := s.Peek(n - 1)
	if err != nil {
		return err
	}
	return s.Push(value)
}

// Swap swaps the top value with the nth value from the top
func (s *Stack) Swap(n int) error {
	if n < 1 {
		return errors.New("invalid swap position")
	}

	topIndex := len(s.data) - 1
	swapIndex := topIndex - n

	if swapIndex < 0 {
		return errors.New("swap underflow")
	}

	s.data[topIndex], s.data[swapIndex] = s.data[swapIndex], s.data[topIndex]
	return nil
}

// Len returns the number of items on the stack
func (s *Stack) Len() int {
	return len(s.data)
}

// Clear removes all items from the stack
func (s *Stack) Clear() {
	s.data = s.data[:0]
}

// Data returns a copy of the stack data
func (s *Stack) Data() []big.Int {
	result := make([]big.Int, len(s.data))
	for i, v := range s.data {
		result[i] = *v
	}
	return result
}

// String returns a string representation of the stack
func (s *Stack) String() string {
	if len(s.data) == 0 {
		return "[]"
	}

	result := "["
	for i := len(s.data) - 1; i >= 0; i-- {
		if i != len(s.data)-1 {
			result += ", "
		}
		result += s.data[i].String()
	}
	result += "]"
	return result
}

// Print prints the stack for debugging
func (s *Stack) Print() {
	fmt.Println("Stack (top to bottom):")
	for i := len(s.data) - 1; i >= 0; i-- {
		fmt.Printf("  [%d]: %s\n", len(s.data)-1-i, s.data[i].String())
	}
}