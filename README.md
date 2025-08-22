# Proof of Contribution Consensus

A research implementation of a Byzantine fault tolerant consensus algorithm that explores reputation-based validator selection and stake weighting mechanisms.

## Project Overview

This project implements a consensus algorithm that combines traditional BFT properties with a reputation system for validator selection. It serves as a research platform for exploring consensus mechanisms, cryptographic techniques, and optimization approaches in distributed systems.

**Note**: This is a research project and experimental implementation. It should not be used in production without extensive testing and security auditing.

## Core Features

### Core Implementation
- **Byzantine Fault Tolerance**: Handles up to f < n/3 malicious nodes
- **Cryptographic Security**: Ed25519 signatures and VRF-based leader selection
- **Stake-Weighted Voting**: Combines economic stake with reputation metrics
- **Slashing Mechanisms**: Penalties for malicious or incorrect behavior
- **Network Protocol**: P2P communication with message authentication

### Experimental Features
- **Post-Quantum Readiness**: Implementations of NIST candidate algorithms (research stage)
- **Neuromorphic Optimization**: Simulation of spiking neural networks for consensus optimization (experimental)

## Quick Start

```bash
# Clone repository
git clone https://github.com/davidcanhelp/sedition.git
cd sedition

# Run demo
go run demo/simple_poc_demo.go

# Run tests
go test ./...

# Benchmark performance
go run benchmarks/ultimate_benchmark_suite.go
```

## Overview

The Proof of Contribution (PoC) consensus algorithm combines economic incentives with cryptographic security to create a practical Byzantine fault tolerant system that rewards quality contributions.

## Mathematical Foundation

The core formula for validator selection is:

```
TotalStake = TokenStake × ReputationMultiplier × ContributionBonus

Where:
- ReputationMultiplier = max(0.5, min(2.0, ReputationScore/5.0))
- ContributionBonus = QualityBonus × FrequencyBonus
- Selection Probability ∝ TotalStake / Sum(AllStakes)
```

## Architecture

The PoC system consists of four main components:

### 1. Main Consensus Engine (`poc.go`)
- **Purpose**: Core consensus logic and validator management
- **Key Features**:
  - Validator registration and stake management
  - Block proposer selection using weighted random selection
  - Slashing mechanisms for malicious behavior
  - Epoch management and cleanup
- **Slashing Conditions**:
  - Malicious Code: 50% reputation slash
  - False Contribution: 30% reputation slash
  - Double Proposal: 40% reputation slash
  - Network Attack: 70% reputation slash
  - Quality Violation: 20% reputation slash

### 2. Quality Analyzer (`quality.go`)
- **Purpose**: Comprehensive code quality assessment
- **Metrics Analyzed**:
  - Cyclomatic complexity (weight: 25%)
  - Test coverage (weight: 30%)
  - Documentation completeness (weight: 20%)
  - Code style compliance (weight: 15%)
  - Security vulnerability assessment (weight: 10%)
- **Security Checks**:
  - SQL injection vulnerabilities
  - Code injection risks
  - Weak cryptographic practices
  - Hardcoded credentials
  - Buffer overflow potential

### 3. Reputation Tracker (`reputation.go`)
- **Purpose**: Long-term contributor reputation management
- **Features**:
  - Exponential decay over time (0.5% daily)
  - Recovery mechanisms for slashed contributors
  - Peer review integration
  - Consistency scoring
  - Historical performance tracking
- **Reputation Range**: 0.5 (minimum) to 10.0 (maximum)

### 4. Metrics Calculator (`metrics.go`)
- **Purpose**: Comprehensive contributor performance analysis
- **Categories**:
  - Productivity (25%): Volume and frequency of contributions
  - Quality (30%): Code quality standards and testing
  - Collaboration (20%): Peer review and community engagement
  - Impact (15%): Bug fixes and system improvements
  - Innovation (10%): Novel approaches and architecture

## Usage Examples

### Basic Setup

```go
import "path/to/poc"

// Create consensus engine
minStake := big.NewInt(1000000) // 1M tokens minimum
blockTime := time.Second * 10   // 10-second blocks
engine := poc.NewConsensusEngine(minStake, blockTime)

// Register validators
err := engine.RegisterValidator("validator1", big.NewInt(5000000))
if err != nil {
    // Handle error
}
```

### Submitting Contributions

```go
contribution := poc.Contribution{
    ID:            "unique-contribution-id",
    Timestamp:     time.Now(),
    Type:          poc.CodeCommit,
    LinesAdded:    150,
    LinesModified: 50,
    TestCoverage:  85.0,
    Complexity:    5.2,
    Documentation: 80.0,
    PeerReviews:   2,
    ReviewScore:   4.5,
}

err := engine.SubmitContribution("validator1", contribution)
```

### Block Proposer Selection

```go
proposer, err := engine.SelectBlockProposer()
if err != nil {
    // Handle error
}
fmt.Printf("Selected proposer: %s\n", proposer)
```

### Reputation Management

```go
// Get current reputation
reputation := engine.reputationTracker.GetReputation("validator1")

// Apply slashing for malicious behavior
engine.SlashValidator("malicious-validator", poc.MaliciousCode, "evidence")

// Record peer review
engine.reputationTracker.RecordPeerReview("validator1", true, 4.5, poc.CodeReviewType)
```

## Configuration Parameters

### Consensus Engine
- `minStakeRequired`: Minimum tokens required to become validator
- `blockTime`: Target time between blocks
- `epochLength`: Number of blocks per epoch (default: 100)
- `slashingRate`: Base slashing percentage (default: 10%)

### Quality Analyzer
- `maxComplexity`: Maximum acceptable cyclomatic complexity (default: 10.0)
- `minTestCoverage`: Minimum test coverage percentage (default: 80.0)
- `minDocumentation`: Minimum documentation coverage (default: 70.0)

### Reputation Tracker
- `baseReputation`: Starting reputation for new contributors (default: 5.0)
- `maxReputation`: Maximum achievable reputation (default: 10.0)
- `minReputation`: Minimum reputation to prevent exclusion (default: 0.5)
- `decayRate`: Daily reputation decay rate (default: 0.5%)

## Performance Characteristics

- **Validator Selection**: O(n) where n = number of active validators
- **Benchmark Results**: ~205μs to select from 1000 validators
- **Memory Usage**: Scales linearly with validator count and contribution history
- **Fairness**: Weighted random selection prevents centralization

## Security Features

1. **Multi-layered Validation**: Combines tokens, reputation, and recent contributions
2. **Slashing Mechanisms**: Economic penalties for malicious behavior  
3. **Recovery Systems**: Paths for rehabilitation after slashing
4. **Anti-centralization**: Fairness checks prevent repeated proposer selection
5. **Code Analysis**: Automated detection of security vulnerabilities

## Integration Testing

The system includes comprehensive tests covering:
- End-to-end consensus functionality
- Quality analysis accuracy
- Reputation tracking over time
- Metrics calculation correctness
- Slashing condition enforcement
- Performance benchmarks

Run tests with:
```bash
go test -v ./...
go test -bench=. -v
```

## Performance Characteristics

| Metric | Measured Value |
|--------|-------|
| Leader Selection | ~206μs per operation |
| Theoretical Throughput | 5,000-10,000 TPS (unoptimized) |
| Finality | Sub-second (network dependent) |
| Byzantine Tolerance | f < n/3 (standard) |
| Memory Usage | Linear with validator count |

## Security

### Security Properties
- **Digital Signatures**: Ed25519 for authentication
- **Leader Selection**: VRF-based to prevent prediction/manipulation
- **Post-Quantum Research**: Experimental implementations of quantum-resistant algorithms
- **Network Security**: Message authentication and integrity checks

### Byzantine Fault Model
- **Safety**: Never finalizes conflicting blocks
- **Liveness**: Always makes progress with ≥2f+1 honest nodes
- **Recovery**: Handles network partitions and rejoining
- **Audit Trail**: Complete cryptographic verification chain

## Potential Applications

### Research & Academic
- Study of Byzantine fault tolerance mechanisms
- Experimentation with consensus algorithms
- Testing of reputation-based systems
- Exploration of post-quantum cryptography

### Possible Use Cases
- Small to medium-scale distributed systems
- Private blockchain networks
- Consensus for collaborative platforms
- Educational demonstrations of BFT consensus

## Development

### Prerequisites
```bash
# Go 1.21+
go version

# For neuromorphic research (optional)
pip install nxsdk  # Intel Loihi SDK
```

### Testing
```bash
# Unit tests
go test ./consensus -v

# Cryptographic validation
go test ./crypto -v

# Performance benchmarks
go test -bench=. ./benchmarks

# Adversarial testing
go run validation/adversarial_testing.go
```

### Deployment
```bash
# Local cluster
docker-compose up -d

# Kubernetes
kubectl apply -f k8s/consensus-cluster.yaml

# Production monitoring
prometheus --config.file=monitoring/prometheus.yml
```

## Contributing

Contributions are welcome in the following areas:
- Bug fixes and performance improvements
- Additional test coverage
- Documentation improvements
- Security audits and reviews

For experimental features:
- Post-quantum cryptography research
- Consensus optimization techniques
- Alternative reputation mechanisms

## Mathematical Proofs

### Fairness Guarantee
The weighted random selection ensures that validators with higher stakes have proportionally higher selection probability, but no validator can guarantee selection, preventing centralization.

### Incentive Alignment
The multi-factor stake calculation (tokens × reputation × contributions) ensures that:
1. Economic stake prevents Sybil attacks
2. Reputation rewards long-term good behavior
3. Recent contributions encourage active participation

### Recovery Mechanism
The reputation recovery system ensures that contributors can rehabilitate after mistakes, preventing permanent exclusion while maintaining security.

## License

Apache 2.0 - See LICENSE file for details.

## Status

**Version**: 1.0.0-substance  
**Stage**: Research implementation  
**Production Readiness**: Not recommended for production use without thorough testing and audit  
**Test Coverage**: ~90% for core components

## Acknowledgments

This implementation builds upon established research in:
- Byzantine fault tolerance (Lamport, Shostak, Pease, 1982)
- Verifiable Random Functions (Micali, Rabin, Vadhan, 1999)
- Post-quantum cryptography (NIST standardization efforts)
- Ed25519 signatures (Bernstein et al., 2012)

## Disclaimer

This is a research implementation intended for educational and experimental purposes. It has not been audited for security and should not be used in production systems without extensive testing and review.