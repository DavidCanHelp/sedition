#!/bin/bash

# Quick Test - Simple one-command validation
# Run this for a fast check that everything works

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🚀 Running Quick Test Suite..."
echo "=============================="
echo ""

# Simple function to run tests and show results
test_step() {
    local name="$1"
    local cmd="$2"
    
    printf "%-40s" "$name..."
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Track results
FAILED=0

# Core tests
test_step "Building packages" "go build ./..." || FAILED=$((FAILED + 1))
test_step "Running unit tests" "go test -short ./consensus ./validator ./contribution 2>/dev/null" || FAILED=$((FAILED + 1))
test_step "Checking formatting" "test -z \$(gofmt -l .)" || FAILED=$((FAILED + 1))
test_step "Running go vet" "go vet ./... 2>/dev/null" || FAILED=$((FAILED + 1))
test_step "Verifying structure" "[ -d config ] && [ -d consensus ] && [ -d validator ]" || FAILED=$((FAILED + 1))

echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All quick tests passed!${NC}"
    echo "Run './test_all.sh' for comprehensive testing"
    exit 0
else
    echo -e "${RED}❌ $FAILED test(s) failed${NC}"
    echo "Run './test_all.sh' for detailed results"
    exit 1
fi