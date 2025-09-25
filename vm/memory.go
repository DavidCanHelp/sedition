package vm

import (
	"fmt"
)

// Memory represents the VM memory
type Memory struct {
	data []byte
}

// NewMemory creates a new memory instance
func NewMemory() *Memory {
	return &Memory{
		data: make([]byte, 0),
	}
}

// Load reads data from memory
func (m *Memory) Load(offset, size uint64) []byte {
	if offset+size > uint64(len(m.data)) {
		m.resize(offset + size)
	}

	result := make([]byte, size)
	copy(result, m.data[offset:offset+size])
	return result
}

// Store writes data to memory
func (m *Memory) Store(offset uint64, data []byte) {
	if offset+uint64(len(data)) > uint64(len(m.data)) {
		m.resize(offset + uint64(len(data)))
	}

	copy(m.data[offset:], data)
}

// Set sets a single byte in memory
func (m *Memory) Set(offset uint64, value byte) {
	if offset >= uint64(len(m.data)) {
		m.resize(offset + 1)
	}
	m.data[offset] = value
}

// Get gets a single byte from memory
func (m *Memory) Get(offset uint64) byte {
	if offset >= uint64(len(m.data)) {
		return 0
	}
	return m.data[offset]
}

// Size returns the current size of memory
func (m *Memory) Size() uint64 {
	return uint64(len(m.data))
}

// Resize resizes the memory to the given size
func (m *Memory) resize(size uint64) {
	if size <= uint64(len(m.data)) {
		return
	}

	// Grow in chunks of 32 bytes (word size)
	newSize := ((size + 31) / 32) * 32

	newData := make([]byte, newSize)
	copy(newData, m.data)
	m.data = newData
}

// Clear clears all memory
func (m *Memory) Clear() {
	m.data = m.data[:0]
}

// Data returns a copy of the memory data
func (m *Memory) Data() []byte {
	result := make([]byte, len(m.data))
	copy(result, m.data)
	return result
}

// LoadWord loads a 32-byte word from memory
func (m *Memory) LoadWord(offset uint64) []byte {
	return m.Load(offset, 32)
}

// StoreWord stores a 32-byte word to memory
func (m *Memory) StoreWord(offset uint64, word []byte) {
	if len(word) > 32 {
		word = word[:32]
	} else if len(word) < 32 {
		// Pad with zeros
		padded := make([]byte, 32)
		copy(padded[32-len(word):], word)
		word = padded
	}
	m.Store(offset, word)
}

// LoadByte loads a single byte from memory
func (m *Memory) LoadByte(offset uint64) byte {
	return m.Get(offset)
}

// StoreByte stores a single byte to memory
func (m *Memory) StoreByte(offset uint64, value byte) {
	m.Set(offset, value)
}

// Copy copies data from one location to another within memory
func (m *Memory) Copy(destOffset, srcOffset, size uint64) {
	if srcOffset+size > uint64(len(m.data)) || destOffset+size > uint64(len(m.data)) {
		maxOffset := srcOffset + size
		if destOffset+size > maxOffset {
			maxOffset = destOffset + size
		}
		m.resize(maxOffset)
	}

	copy(m.data[destOffset:destOffset+size], m.data[srcOffset:srcOffset+size])
}

// String returns a hex representation of memory
func (m *Memory) String() string {
	if len(m.data) == 0 {
		return "0x"
	}
	return fmt.Sprintf("0x%x", m.data)
}

// Print prints memory contents for debugging
func (m *Memory) Print(wordsPerLine int) {
	if wordsPerLine <= 0 {
		wordsPerLine = 4
	}

	fmt.Printf("Memory (%d bytes):\n", len(m.data))

	for i := 0; i < len(m.data); i += 32 * wordsPerLine {
		end := i + 32*wordsPerLine
		if end > len(m.data) {
			end = len(m.data)
		}

		fmt.Printf("  0x%04x: ", i)
		for j := i; j < end; j += 32 {
			wordEnd := j + 32
			if wordEnd > len(m.data) {
				wordEnd = len(m.data)
				// Pad with spaces for incomplete words
				word := make([]byte, 32)
				copy(word, m.data[j:wordEnd])
				fmt.Printf("%064x ", word)
			} else {
				fmt.Printf("%064x ", m.data[j:wordEnd])
			}
		}
		fmt.Println()
	}
}

// GetMemoryRange returns memory within the specified range
func (m *Memory) GetMemoryRange(offset, size uint64) []byte {
	if offset >= uint64(len(m.data)) {
		return make([]byte, size)
	}

	end := offset + size
	if end > uint64(len(m.data)) {
		result := make([]byte, size)
		copy(result, m.data[offset:])
		return result
	}

	return m.Load(offset, size)
}

// SetMemoryRange sets memory within the specified range
func (m *Memory) SetMemoryRange(offset uint64, data []byte) {
	m.Store(offset, data)
}