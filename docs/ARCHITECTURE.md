# Architecture Documentation

## Overview

The Sedition project implements a Proof of Contribution (PoC) consensus mechanism designed for collaborative software development. The architecture is organized into modular packages that separate concerns and promote maintainability.

## Package Structure

```
sedition/
├── config/           # Configuration management
├── consensus/        # Core consensus engine
├── contribution/     # Contribution tracking and quality analysis
├── validator/        # Validator management and reputation
├── errors/          # Custom error types
├── crypto/          # Cryptographic utilities
├── network/         # P2P networking
└── storage/         # Data persistence
```

## Core Components

### 1. Consensus Engine (`consensus/`)

The heart of the PoC mechanism, responsible for:
- Block proposer selection using weighted random sampling
- Validator stake calculation combining tokens, reputation, and contributions
- Epoch management and state transitions
- Slashing mechanism for malicious behavior

**Key Files:**
- `engine.go`: Main consensus engine implementation

**Key Types:**
- `Engine`: Core consensus engine struct
- `NetworkStats`: Network-wide statistics

### 2. Validator Management (`validator/`)

Manages validator state, reputation, and slashing:
- Validator registration and activation
- Reputation score tracking and decay
- Slashing event management
- Historical performance tracking

**Key Files:**
- `validator.go`: Validator state management
- `reputation.go`: Reputation tracking system

**Key Types:**
- `Validator`: Individual validator state
- `ReputationTracker`: Reputation management system
- `SlashingEvent`: Records of slashing incidents

### 3. Contribution System (`contribution/`)

Analyzes and tracks code contributions:
- Quality scoring algorithm
- Contribution type classification
- Metrics calculation and aggregation
- Impact assessment

**Key Files:**
- `contribution.go`: Core contribution types
- `quality.go`: Quality analysis algorithms
- `metrics.go`: Metrics calculation

**Key Types:**
- `Contribution`: Individual contribution record
- `QualityAnalyzer`: Analyzes contribution quality
- `MetricsCalculator`: Calculates aggregate metrics

### 4. Configuration (`config/`)

Centralized configuration management:
- Consensus parameters
- Validation thresholds
- Network settings
- Zero-knowledge proof parameters

**Key Files:**
- `config.go`: Configuration structures and defaults

**Key Types:**
- `ConsensusConfig`: Consensus engine configuration
- `ValidationConfig`: Validation framework configuration
- `NetworkConfig`: Network layer configuration

### 5. Error Handling (`errors/`)

Structured error management:
- Error codes for different subsystems
- Context-rich error information
- Error wrapping and unwrapping

**Key Files:**
- `errors.go`: Custom error types

**Key Types:**
- `ConsensusError`: Consensus-related errors
- `NetworkError`: Network-related errors
- `ValidationError`: Validation errors

## Data Flow

### 1. Validator Registration
```
User → RegisterValidator() → ConsensusEngine → ValidatorStore
                                     ↓
                              ReputationTracker
```

### 2. Contribution Submission
```
Contributor → SubmitContribution() → QualityAnalyzer
                 ↓                         ↓
           ValidatorState ← ReputationTracker
                 ↓
           StakeCalculation
```

### 3. Block Proposer Selection
```
ConsensusEngine → GetActiveValidators()
        ↓
  CalculateTotalStakes()
        ↓
  WeightedRandomSelection()
        ↓
  FairnessCheck()
        ↓
  SelectedProposer
```

## Consensus Algorithm

### Stake Calculation Formula

```
TotalStake = TokenStake × ReputationMultiplier × ContributionBonus

Where:
- ReputationMultiplier: 0.1 to 2.0 (based on 0-10 reputation score)
- ContributionBonus: 0.8 to 1.5 (based on recent contribution quality)
```

### Selection Probability

```
P(validator) = ValidatorTotalStake / ΣAllValidatorStakes
```

### Quality Score Components

1. **Test Coverage** (25%): Percentage of test coverage added
2. **Documentation** (20%): Documentation completeness
3. **Complexity** (25%): Inverse of cyclomatic complexity
4. **Type Bonus** (15%): Based on contribution type
5. **Review Score** (15%): Peer review ratings

## Security Considerations

### Slashing Conditions

Validators can be slashed for:
1. **Malicious Code**: Submitting harmful code
2. **False Contributions**: Claiming false work
3. **Double Proposal**: Proposing multiple blocks
4. **Network Attacks**: Attempting to disrupt the network
5. **Quality Violations**: Consistently poor quality contributions

### Slashing Penalties

- Token slash: 10% of staked tokens (configurable)
- Reputation penalty: 0.5 to 3.0 points depending on severity
- Potential deactivation if stake falls below minimum

## Performance Optimization

### Caching Strategy
- Validator stakes cached and recalculated only on changes
- Reputation scores updated incrementally
- Contribution metrics aggregated periodically

### Concurrency
- Read-write locks for validator state
- Atomic operations for stake updates
- Channel-based communication for network events

## Configuration Management

All hardcoded values have been extracted to the `config` package:

### Key Configuration Parameters
- `MinStakeRequired`: Minimum stake to become a validator
- `SlashingRate`: Percentage of stake slashed for violations
- `ReputationDecayRate`: Rate of reputation decay over time
- `QualityThreshold`: Minimum quality score for contributions
- `ProposerHistorySize`: Number of recent proposers to track

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock dependencies
- Edge case coverage

### Integration Tests
- End-to-end consensus flow
- Network simulation
- Slashing scenarios

### Benchmarks
- Proposer selection performance
- Stake calculation efficiency
- Quality analysis speed

## Future Enhancements

### Planned Improvements
1. **Sharding**: Horizontal scaling for large networks
2. **Zero-Knowledge Proofs**: Private contribution verification
3. **Cross-Chain Interoperability**: Bridge to other blockchains
4. **Machine Learning**: Advanced quality prediction
5. **Formal Verification**: Mathematical proof of consensus properties

### Scalability Considerations
- Current design supports ~1000 validators efficiently
- Sharding implementation planned for 10,000+ validators
- Optimistic rollups for contribution aggregation

## Development Workflow

### Adding New Features
1. Update configuration in `config/`
2. Implement logic in appropriate package
3. Add unit tests
4. Update integration tests
5. Document changes

### Code Quality Standards
- All exported functions must have godoc comments
- Test coverage minimum: 70%
- Linting must pass (golangci-lint)
- No magic numbers (use config)
- Error handling with context

## Deployment

### Requirements
- Go 1.23+
- LevelDB for storage
- Prometheus for metrics

### Environment Variables
- `SEDITION_CONFIG`: Path to configuration file
- `SEDITION_DATA_DIR`: Data directory path
- `SEDITION_LOG_LEVEL`: Logging verbosity

### Monitoring
- Prometheus metrics exposed on `/metrics`
- Health check endpoint on `/health`
- Grafana dashboards for visualization

## Support

For questions or issues:
- GitHub Issues: [github.com/davidcanhelp/sedition/issues](https://github.com/davidcanhelp/sedition/issues)
- Documentation: See `/docs` directory
- Contributing: See `CONTRIBUTING.md`