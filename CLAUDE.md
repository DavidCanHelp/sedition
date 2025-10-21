# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of a Byzantine fault tolerant consensus algorithm with reputation-based validator selection and stake weighting mechanisms. It's written in Go and serves as a research platform for distributed systems consensus mechanisms.

**Important**: This is an experimental research project, not production-ready code. It explores consensus algorithms, cryptographic techniques, and optimization approaches.

## Core Architecture

The system implements a Proof of Contribution (PoC) consensus with these key components:

1. **consensus/engine.go**: Core consensus engine managing validator registration, block proposer selection using weighted random selection, slashing mechanisms, and epoch management. Uses formula: `TotalStake = TokenStake × ReputationMultiplier × ContributionBonus`

2. **validator/**: Validator state management and reputation tracking with exponential decay (0.5% daily), recovery mechanisms, and peer review integration. Reputation ranges from 0.5 to 10.0.

3. **contribution/**: Contribution tracking system with quality analysis (cyclomatic complexity, test coverage, documentation, security checks) and comprehensive metrics calculation across productivity, quality, collaboration, impact, and innovation categories. **97.0% test coverage** with 49+ test functions covering all quality analysis algorithms, metrics calculations, and edge cases.

4. **config/**: Centralized configuration management (100% test coverage)

5. **errors/**: Custom error types with context (40.9% test coverage)

## Build and Test Commands

```bash
# Quick validation (fastest check)
./quick_test.sh
# or
make quick

# Run comprehensive test suite with coverage
make test
# or
./run_tests.sh

# Run specific test
make test-specific TEST=TestFunctionName

# Build the demo binary
make build

# Run the demo
go run demo/simple_poc_demo.go

# Generate coverage report
make coverage

# Run benchmarks
make benchmark

# Run linter (golangci-lint)
make lint

# Format code
make fmt

# Run security checks
make security

# Full CI simulation
make ci
```

## Testing Options

The test runner (`./run_tests.sh`) supports:
- `--verbose` or `-v`: Detailed output
- `--benchmark` or `-b`: Include benchmark tests
- `--no-race`: Disable race detection
- `--coverage-threshold N`: Set coverage threshold (default: 70)

Environment variables:
- `VERBOSE=true`: Enable verbose output
- `BENCHMARK=true`: Run benchmark tests
- `RACE=false`: Disable race detection

## Linting Configuration

Uses golangci-lint with extensive checks configured in `.golangci.yml`:
- Standard checks: gofmt, goimports, govet, errcheck, staticcheck
- Additional: gosec, gocritic, revive, errorlint, wrapcheck
- Security scanning with gosec (medium severity)
- Custom rules for error handling and code quality

## Key Technical Details

- **Byzantine Fault Tolerance**: Handles f < n/3 malicious nodes
- **Cryptography**: Ed25519 signatures, VRF-based leader selection
- **Slashing Rates**: Malicious Code (50%), False Contribution (30%), Double Proposal (40%), Network Attack (70%), Quality Violation (20%)
- **Performance**: ~205μs for leader selection from 1000 validators
- **Dependencies**: Prometheus metrics, LevelDB storage, golang.org/x/crypto

## Development Workflow

1. Before making changes, run `make fmt` and `make lint`
2. After implementing features, run `make test` to ensure tests pass
3. For performance-critical changes, run `make benchmark`
4. Check security implications with `make security`
5. The project follows standard Go project structure with package-level organization

## Demo and Production Deployment

### Running the Demo
```bash
# Single node with web UI
go run demo/server.go --validator --name alice --stake 10000

# Multi-node network
./demo/launch.sh

# Access web UI at http://localhost:8080
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Monitoring stack included:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Alertmanager: http://localhost:9093
```

### Key API Endpoints
- `/api/status` - Node status
- `/api/blocks` - Recent blocks
- `/api/transaction` - Submit transaction
- `/metrics` - Prometheus metrics