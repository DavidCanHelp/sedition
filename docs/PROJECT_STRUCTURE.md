# Sedition Blockchain Project Structure

## 📁 Directory Organization

```
sedition/
├── cmd/                    # Command-line applications
│   └── node/              # Main blockchain node executable
├── config/                # Configuration packages
├── consensus/             # Consensus engine (Proof of Stake)
├── crypto/                # Cryptographic utilities
├── demo/                  # Demo applications
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md   # System architecture
│   ├── CURRENCY.md       # SedCoin (SED) specification
│   └── ...               # Various documentation files
├── internal/              # Internal packages
│   └── poc/              # Proof of concept code
├── mempool/               # Transaction pool management
├── mining/                # Block production and mining
├── network/               # P2P networking layer
├── rpc/                   # JSON-RPC server
├── scripts/               # Build and test scripts
├── storage/               # Blockchain storage and state
├── tests/                 # Test files
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── benchmarks/       # Performance benchmarks
├── validator/             # Validator management
├── vm/                    # Virtual machine for smart contracts
└── wallet/                # Wallet and key management
```

## 🚀 Quick Start

```bash
# Build the node
cd cmd/node && go build -o sedition-node

# Run a node with mining
./sedition-node --mine --datadir=/path/to/data

# Run tests
go test ./...
```

## 💰 Currency

- **Name**: SedCoin (SED)
- **Block Reward**: 2 SED
- **Block Time**: 10 seconds
- **Smallest Unit**: 1 wei (10^-18 SED)

## 🛠️ Core Components

- **Blockchain**: Full blockchain with genesis block, state management, and persistence
- **Consensus**: Proof of Stake consensus mechanism
- **Networking**: P2P gossip protocol for block/transaction propagation
- **RPC**: Ethereum-compatible JSON-RPC interface
- **Mining**: Block production with configurable parameters
- **Wallet**: ECDSA key generation and transaction signing
- **Transaction Pool**: Mempool with gas price prioritization

## 📝 Configuration Files

- `docker-compose.yml`: Docker deployment configuration
- `Dockerfile`: Container build specification
- `Makefile`: Build automation
- `.golangci.yml`: Linting configuration
- `go.mod/go.sum`: Go module dependencies