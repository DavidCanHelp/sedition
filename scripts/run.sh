#!/bin/bash

# Build the node and client
echo "Building Sedition blockchain node..."
go build -o sedition-node ./cmd/node
go build -o sedition-client ./cmd/client

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"

# Parse command line arguments
COMMAND=${1:-node}

case $COMMAND in
    node)
        echo "Starting Sedition blockchain node..."
        ./sedition-node \
            --datadir ./data \
            --rpc-host 127.0.0.1 \
            --rpc-port 8545 \
            --p2p-port 30303 \
            --mine \
            --wallet ./wallet.json
        ;;

    node-genesis)
        echo "Starting Sedition blockchain node with genesis block..."
        ./sedition-node \
            --datadir ./data \
            --rpc-host 127.0.0.1 \
            --rpc-port 8545 \
            --p2p-port 30303 \
            --mine \
            --wallet ./wallet.json \
            --genesis
        ;;

    node2)
        echo "Starting second Sedition blockchain node..."
        ./sedition-node \
            --datadir ./data2 \
            --rpc-host 127.0.0.1 \
            --rpc-port 8546 \
            --p2p-port 30304 \
            --wallet ./wallet2.json \
            --bootnodes "localhost:30303"
        ;;

    client)
        shift
        echo "Running Sedition client..."
        ./sedition-client "$@"
        ;;

    clean)
        echo "Cleaning up data directories..."
        rm -rf ./data ./data2
        rm -f ./wallet.json ./wallet2.json
        rm -f ./sedition-node ./sedition-client
        echo "Cleanup complete!"
        ;;

    demo)
        echo "Running Sedition PoC demo..."
        ./sedition --validator --name demo-validator --stake 10000
        ;;

    test)
        echo "Running tests..."
        go test ./... -v
        ;;

    *)
        echo "Sedition Blockchain"
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  demo          Run the PoC demo (interactive blockchain demonstration)"
        echo "  node          Start the main node with mining"
        echo "  node-genesis  Start node and initialize genesis block"
        echo "  node2         Start a second node (for testing P2P)"
        echo "  client        Run the client CLI"
        echo "  clean         Clean up data directories and binaries"
        echo "  test          Run all tests"
        echo ""
        echo "Client usage:"
        echo "  ./run.sh client accounts"
        echo "  ./run.sh client balance <address>"
        echo "  ./run.sh client send <from> <to> <amount>"
        echo "  ./run.sh client block latest"
        echo "  ./run.sh client mining"
        ;;
esac