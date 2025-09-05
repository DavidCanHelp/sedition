package benchmarks

import "sync/atomic"

// Stub implementation of ConsensusAlgorithm interface
type StubConsensusAlgorithm struct {
	initialized bool
	nodeCount   int
	txCount     int64
}

func (s *StubConsensusAlgorithm) Initialize(config map[string]interface{}) error {
	if nodes, ok := config["nodes"].(int); ok {
		s.nodeCount = nodes
	}
	s.initialized = true
	return nil
}

func (s *StubConsensusAlgorithm) ProposeBlock(data []byte) ([]byte, error) {
	return data, nil
}

func (s *StubConsensusAlgorithm) ValidateBlock(block []byte) (bool, error) {
	return true, nil
}

func (s *StubConsensusAlgorithm) GetMetrics() map[string]float64 {
	return map[string]float64{
		"throughput": float64(atomic.LoadInt64(&s.txCount)),
		"nodes":      float64(s.nodeCount),
	}
}

func (s *StubConsensusAlgorithm) ProcessTransaction(tx []byte) error {
	atomic.AddInt64(&s.txCount, 1)
	return nil
}