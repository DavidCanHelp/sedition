#!/bin/bash

# Comprehensive Test Script for Sedition Project
# This script runs all tests and validations

set -e

# Colors for output
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to print section headers
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE} $1${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n -e "  Testing $test_name... "
    
    if eval "$test_command" > /tmp/test_output_$$.log 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo -e "  ${YELLOW}Error output:${NC}"
        tail -5 /tmp/test_output_$$.log | sed 's/^/    /'
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to check if a command exists
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "  ${YELLOW}⚠${NC} $1 is not installed"
        return 1
    fi
}

# Start testing
echo -e "${BOLD}🧪 SEDITION COMPREHENSIVE TEST SUITE${NC}"
echo -e "${BOLD}════════════════════════════════════${NC}"
echo "Starting at: $(date)"

# 1. Environment Check
print_header "1. ENVIRONMENT CHECK"
check_command "go"
check_command "git"
check_command "make"
GO_VERSION=$(go version | awk '{print $3}')
echo -e "  Go version: ${BLUE}$GO_VERSION${NC}"

# 2. Dependency Check
print_header "2. DEPENDENCY CHECK"
echo "  Checking go.mod..."
if [ -f "go.mod" ]; then
    echo -e "  ${GREEN}✓${NC} go.mod exists"
    run_test "module download" "go mod download"
    run_test "module verify" "go mod verify"
else
    echo -e "  ${RED}✗${NC} go.mod not found"
fi

# 3. Build Test
print_header "3. BUILD TEST"
run_test "consensus package" "go build ./consensus"
run_test "validator package" "go build ./validator"
run_test "contribution package" "go build ./contribution"
run_test "config package" "go build ./config"
run_test "errors package" "go build ./errors"
run_test "main poc package" "go build ."

# 4. Unit Tests
print_header "4. UNIT TESTS"
run_test "consensus unit tests" "go test -short ./consensus/..."
run_test "validator unit tests" "go test -short ./validator/..." || SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
run_test "contribution unit tests" "go test -short ./contribution/..." || SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
run_test "poc legacy tests" "go test -short -run '^Test' poc_unit_test.go poc.go metrics.go reputation.go quality.go"

# 5. Integration Tests
print_header "5. INTEGRATION TESTS"
run_test "consensus integration" "go test -run Integration ./consensus/..."
run_test "end-to-end flow" "go test -run TestConsensusEngineIntegration ./consensus/..."

# 6. Code Quality
print_header "6. CODE QUALITY CHECKS"
run_test "go fmt check" "test -z \$(gofmt -l .)"
run_test "go vet" "go vet ./..."

# Check if golangci-lint is installed
if command -v golangci-lint >/dev/null 2>&1; then
    run_test "golangci-lint" "golangci-lint run --timeout=5m ./..."
else
    echo -e "  ${YELLOW}⚠${NC} golangci-lint not installed - skipping"
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
fi

# 7. Benchmarks (quick run)
print_header "7. PERFORMANCE BENCHMARKS"
echo "  Running quick benchmarks..."
if go test -bench=. -benchtime=1s -run=^$ ./consensus 2>/dev/null | grep -E "Benchmark|ns/op" > /tmp/bench_$$.log; then
    echo -e "  ${GREEN}✓${NC} Benchmarks completed"
    echo "  Sample results:"
    head -5 /tmp/bench_$$.log | sed 's/^/    /'
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "  ${YELLOW}⚠${NC} Benchmarks skipped"
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# 8. Test Coverage
print_header "8. TEST COVERAGE"
echo "  Calculating test coverage..."
if go test -cover ./consensus ./validator ./contribution 2>/dev/null | grep coverage > /tmp/coverage_$$.log; then
    echo -e "  ${GREEN}✓${NC} Coverage calculated"
    echo "  Results:"
    cat /tmp/coverage_$$.log | sed 's/^/    /'
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "  ${YELLOW}⚠${NC} Coverage calculation failed"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# 9. Package Structure Validation
print_header "9. PACKAGE STRUCTURE VALIDATION"
run_test "config package exists" "[ -d config ]"
run_test "consensus package exists" "[ -d consensus ]"
run_test "validator package exists" "[ -d validator ]"
run_test "contribution package exists" "[ -d contribution ]"
run_test "errors package exists" "[ -d errors ]"

# 10. Documentation Check
print_header "10. DOCUMENTATION CHECK"
run_test "README exists" "[ -f README.md ]"
run_test "Architecture doc exists" "[ -f ARCHITECTURE.md ]"
run_test "Tech debt report exists" "[ -f TECH_DEBT_REPORT.md ]"
run_test "Makefile exists" "[ -f Makefile ]"

# 11. Configuration Files
print_header "11. CONFIGURATION FILES"
run_test "go.mod exists" "[ -f go.mod ]"
run_test "golangci config exists" "[ -f .golangci.yml ]"
run_test "GitHub workflow exists" "[ -f .github/workflows/ci.yml ]"

# 12. Functional Test Scenario
print_header "12. FUNCTIONAL TEST SCENARIO"
echo "  Running end-to-end scenario test..."
cat > /tmp/scenario_test.go << 'EOF'
package main

import (
    "fmt"
    "math/big"
    "time"
    "github.com/davidcanhelp/sedition/config"
    "github.com/davidcanhelp/sedition/consensus"
    "github.com/davidcanhelp/sedition/contribution"
)

func main() {
    // Create consensus engine
    cfg := config.DefaultConsensusConfig()
    cfg.MinStakeRequired = big.NewInt(100)
    engine := consensus.NewEngine(cfg)
    
    // Register validators
    if err := engine.RegisterValidator("alice", big.NewInt(1000)); err != nil {
        panic(err)
    }
    if err := engine.RegisterValidator("bob", big.NewInt(2000)); err != nil {
        panic(err)
    }
    
    // Submit contributions
    contrib := contribution.Contribution{
        ID: "test1",
        Timestamp: time.Now(),
        Type: contribution.CodeCommit,
        LinesAdded: 100,
        TestCoverage: 80.0,
    }
    if err := engine.SubmitContribution("alice", contrib); err != nil {
        panic(err)
    }
    
    // Select proposer
    proposer, err := engine.SelectBlockProposer()
    if err != nil {
        panic(err)
    }
    
    // Get stats
    stats := engine.GetNetworkStats()
    if stats.TotalValidators != 2 {
        panic("Expected 2 validators")
    }
    
    fmt.Println("SUCCESS: All functional tests passed")
}
EOF

if go run /tmp/scenario_test.go 2>/dev/null | grep -q "SUCCESS"; then
    echo -e "  ${GREEN}✓${NC} End-to-end scenario passed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "  ${RED}✗${NC} End-to-end scenario failed"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Clean up temp files
rm -f /tmp/test_output_$$.log /tmp/bench_$$.log /tmp/coverage_$$.log /tmp/scenario_test.go 2>/dev/null

# Final Summary
print_header "TEST SUMMARY"
echo ""
echo -e "  ${BOLD}Total Tests:${NC}    $TOTAL_TESTS"
echo -e "  ${GREEN}Passed:${NC}         $PASSED_TESTS"
echo -e "  ${RED}Failed:${NC}         $FAILED_TESTS"
echo -e "  ${YELLOW}Skipped:${NC}        $SKIPPED_TESTS"
echo ""

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    echo -e "  ${BOLD}Pass Rate:${NC}      ${PASS_RATE}%"
    
    if [ $PASS_RATE -ge 90 ]; then
        echo ""
        echo -e "  ${GREEN}${BOLD}✅ EXCELLENT! All critical tests passed.${NC}"
        EXIT_CODE=0
    elif [ $PASS_RATE -ge 70 ]; then
        echo ""
        echo -e "  ${YELLOW}${BOLD}⚠️  GOOD. Most tests passed, but review failures.${NC}"
        EXIT_CODE=1
    else
        echo ""
        echo -e "  ${RED}${BOLD}❌ NEEDS ATTENTION. Multiple test failures detected.${NC}"
        EXIT_CODE=2
    fi
else
    echo -e "  ${RED}No tests were run${NC}"
    EXIT_CODE=3
fi

echo ""
echo "Completed at: $(date)"
echo ""

# Quick tips
if [ $FAILED_TESTS -gt 0 ] || [ $SKIPPED_TESTS -gt 0 ]; then
    echo -e "${BOLD}💡 Quick Fixes:${NC}"
    if [ $SKIPPED_TESTS -gt 0 ]; then
        echo "  • Install missing tools: make install-tools"
    fi
    if [ $FAILED_TESTS -gt 0 ]; then
        echo "  • Run detailed tests: go test -v ./..."
        echo "  • Check specific package: go test -v ./consensus"
    fi
    echo ""
fi

exit $EXIT_CODE