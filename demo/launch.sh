#!/bin/bash

# Launch script for PoC Blockchain demo network

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Launching PoC Blockchain Demo Network${NC}"
echo "========================================="

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down network...${NC}"
    pkill -f "demo/server.go" 2>/dev/null || true
    echo -e "${GREEN}Network stopped${NC}"
}

trap cleanup EXIT

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${YELLOW}Go is not installed. Please install Go 1.21+ first.${NC}"
    exit 1
fi

# Build the demo server
echo -e "${BLUE}Building demo server...${NC}"
go build -o demo/server demo/server.go || {
    echo -e "${YELLOW}Build failed. Running with go run instead...${NC}"
}

# Create data directories
mkdir -p data/node1 data/node2 data/node3

# Launch bootstrap validator (Alice)
echo -e "${GREEN}Starting Node 1 (Alice - Bootstrap Validator)...${NC}"
go run demo/server.go \
    --validator \
    --name alice \
    --stake 10000 \
    --http :8080 \
    --p2p :8545 \
    --data ./data/node1 &

sleep 2

# Launch second validator (Bob)
echo -e "${GREEN}Starting Node 2 (Bob - Validator)...${NC}"
go run demo/server.go \
    --validator \
    --name bob \
    --stake 20000 \
    --http :8081 \
    --p2p :8546 \
    --bootstrap localhost:8545 \
    --data ./data/node2 &

sleep 2

# Launch observer node
echo -e "${GREEN}Starting Node 3 (Observer)...${NC}"
go run demo/server.go \
    --http :8082 \
    --p2p :8547 \
    --bootstrap localhost:8545 \
    --data ./data/node3 &

sleep 2

echo ""
echo -e "${GREEN}✅ Network is running!${NC}"
echo ""
echo "Access the nodes at:"
echo "  Node 1 (Alice): http://localhost:8080"
echo "  Node 2 (Bob):   http://localhost:8081"
echo "  Node 3 (Observer): http://localhost:8082"
echo ""
echo "Metrics available at:"
echo "  http://localhost:8080/metrics"
echo "  http://localhost:8081/metrics"
echo "  http://localhost:8082/metrics"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the network${NC}"

# Keep script running
wait