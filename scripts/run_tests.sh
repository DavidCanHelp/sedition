#!/bin/bash

# Comprehensive Test Suite Runner for Sedition Project
# This script runs all tests with coverage analysis and detailed reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COVERAGE_THRESHOLD=70
VERBOSE=${VERBOSE:-false}
RACE_DETECTION=${RACE:-true}
BENCHMARK=${BENCHMARK:-false}

# Print header
echo "================================================"
echo "🧪 Sedition Project - Comprehensive Test Suite"
echo "================================================"
echo ""

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    
    if [ "$status" = "success" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "error" ]; then
        echo -e "${RED}✗${NC} $message"
    elif [ "$status" = "warning" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    elif [ "$status" = "info" ]; then
        echo -e "${BLUE}ℹ${NC} $message"
    fi
}

# Function to run tests for a package
run_package_tests() {
    local package=$1
    local package_name=$(basename $package)
    
    echo -e "\n${BLUE}Testing package: $package_name${NC}"
    echo "----------------------------------------"
    
    # Check if package has test files
    if ! ls ${package}/*_test.go &> /dev/null; then
        print_status "warning" "No test files found in $package_name"
        return 0
    fi
    
    # Build test flags
    local test_flags=""
    if [ "$VERBOSE" = "true" ]; then
        test_flags="$test_flags -v"
    fi
    if [ "$RACE_DETECTION" = "true" ]; then
        test_flags="$test_flags -race"
    fi
    
    # Run tests with coverage
    if go test $test_flags -coverprofile=/tmp/coverage_${package_name}.out -covermode=atomic ./${package}/... 2>/tmp/test_${package_name}.log; then
        # Extract coverage percentage
        coverage=$(go tool cover -func=/tmp/coverage_${package_name}.out | grep total | awk '{print $3}' | sed 's/%//')
        
        if [ ! -z "$coverage" ]; then
            coverage_int=$(echo $coverage | cut -d. -f1)
            if [ $coverage_int -ge $COVERAGE_THRESHOLD ]; then
                print_status "success" "$package_name tests passed (coverage: ${coverage}%)"
            else
                print_status "warning" "$package_name tests passed (coverage: ${coverage}% - below threshold)"
            fi
        else
            print_status "success" "$package_name tests passed"
        fi
        return 0
    else
        print_status "error" "$package_name tests failed"
        if [ "$VERBOSE" = "true" ]; then
            cat /tmp/test_${package_name}.log
        fi
        return 1
    fi
}

# Function to run benchmark tests
run_benchmarks() {
    echo -e "\n${BLUE}Running Benchmark Tests${NC}"
    echo "----------------------------------------"
    
    local bench_packages="config errors consensus"
    
    for pkg in $bench_packages; do
        if ls ${pkg}/*_test.go &> /dev/null 2>&1; then
            echo "Benchmarking $pkg..."
            go test -bench=. -benchmem -run=^$ ./${pkg}/... | grep -E "^(Benchmark|ok|PASS|FAIL)" || true
        fi
    done
}

# Function to generate coverage report
generate_coverage_report() {
    echo -e "\n${BLUE}Generating Coverage Report${NC}"
    echo "----------------------------------------"
    
    # Combine all coverage files
    echo "mode: atomic" > /tmp/coverage_total.out
    
    for file in /tmp/coverage_*.out; do
        if [ -f "$file" ] && [ "$file" != "/tmp/coverage_total.out" ]; then
            tail -n +2 "$file" >> /tmp/coverage_total.out 2>/dev/null || true
        fi
    done
    
    # Generate HTML report if combined file has content
    if [ -s /tmp/coverage_total.out ]; then
        go tool cover -html=/tmp/coverage_total.out -o coverage.html 2>/dev/null || true
        
        # Calculate total coverage
        total_coverage=$(go tool cover -func=/tmp/coverage_total.out | grep total | awk '{print $3}' | sed 's/%//' || echo "0")
        
        if [ ! -z "$total_coverage" ]; then
            print_status "info" "Total coverage: ${total_coverage}%"
            print_status "info" "Coverage report saved to coverage.html"
        fi
    fi
}

# Function to check for common issues
run_static_analysis() {
    echo -e "\n${BLUE}Running Static Analysis${NC}"
    echo "----------------------------------------"
    
    # Check formatting
    if [ -z "$(gofmt -l .)" ]; then
        print_status "success" "Code formatting check passed"
    else
        print_status "warning" "Some files need formatting (run: go fmt ./...)"
        if [ "$VERBOSE" = "true" ]; then
            gofmt -l .
        fi
    fi
    
    # Run go vet
    if go vet ./... 2>/tmp/vet.log; then
        print_status "success" "go vet passed"
    else
        print_status "warning" "go vet found issues"
        if [ "$VERBOSE" = "true" ]; then
            cat /tmp/vet.log
        fi
    fi
    
    # Check for inefficient assignments (if ineffassign is installed)
    if command -v ineffassign &> /dev/null; then
        if ineffassign . 2>/dev/null | grep -q "ineffectual"; then
            print_status "warning" "ineffassign found ineffectual assignments"
        else
            print_status "success" "No ineffectual assignments found"
        fi
    fi
    
    # Check for unused code (if unused is installed)
    if command -v unused &> /dev/null; then
        if unused ./... 2>/dev/null | grep -q "unused"; then
            print_status "warning" "Found unused code"
        else
            print_status "success" "No unused code found"
        fi
    fi
}

# Main execution
main() {
    local start_time=$(date +%s)
    local exit_code=0
    
    # Clean up previous coverage files
    rm -f /tmp/coverage_*.out 2>/dev/null
    rm -f /tmp/test_*.log 2>/dev/null
    
    # Step 1: Check if Go is installed
    if ! command -v go &> /dev/null; then
        print_status "error" "Go is not installed"
        exit 1
    fi
    
    print_status "info" "Go version: $(go version)"
    
    # Step 2: Download dependencies
    print_status "info" "Downloading dependencies..."
    go mod download 2>/dev/null || print_status "warning" "Some dependencies could not be downloaded"
    
    # Step 3: Build the main package
    echo -e "\n${BLUE}Building Main Package${NC}"
    echo "----------------------------------------"
    if go build -o /tmp/sedition_test . 2>/tmp/build.log; then
        print_status "success" "Main package builds successfully"
        rm -f /tmp/sedition_test
    else
        print_status "error" "Main package build failed"
        if [ "$VERBOSE" = "true" ]; then
            cat /tmp/build.log
        fi
        exit_code=1
    fi
    
    # Step 4: Run tests for each package
    echo -e "\n${BLUE}Running Unit Tests${NC}"
    echo "========================================"
    
    # Core packages
    for package in config errors consensus validator contribution; do
        if [ -d "$package" ]; then
            run_package_tests $package || exit_code=1
        fi
    done
    
    # Extended packages (allow failures)
    for package in storage network crypto benchmarks optimization validation research deployment monitoring security zkp github; do
        if [ -d "$package" ]; then
            run_package_tests $package || print_status "warning" "$package tests failed (non-critical)"
        fi
    done
    
    # Step 5: Run integration tests
    echo -e "\n${BLUE}Running Integration Tests${NC}"
    echo "----------------------------------------"
    if go test -v -tags=integration . 2>/tmp/integration.log; then
        print_status "success" "Integration tests passed"
    else
        print_status "warning" "Some integration tests failed"
        if [ "$VERBOSE" = "true" ]; then
            cat /tmp/integration.log | grep -E "(FAIL|Error)" || true
        fi
    fi
    
    # Step 6: Run benchmarks if requested
    if [ "$BENCHMARK" = "true" ]; then
        run_benchmarks
    fi
    
    # Step 7: Run static analysis
    run_static_analysis
    
    # Step 8: Generate coverage report
    generate_coverage_report
    
    # Calculate execution time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Print summary
    echo ""
    echo "================================================"
    echo "📊 Test Suite Summary"
    echo "================================================"
    
    if [ $exit_code -eq 0 ]; then
        print_status "success" "All critical tests passed!"
    else
        print_status "error" "Some critical tests failed"
    fi
    
    print_status "info" "Execution time: ${duration} seconds"
    
    # Provide recommendations
    echo ""
    echo "📝 Recommendations:"
    echo "-------------------"
    if [ "$BENCHMARK" != "true" ]; then
        echo "• Run with BENCHMARK=true to include benchmark tests"
    fi
    if [ "$RACE_DETECTION" != "true" ]; then
        echo "• Run with RACE=true to enable race condition detection"
    fi
    if [ "$VERBOSE" != "true" ]; then
        echo "• Run with VERBOSE=true for detailed output"
    fi
    echo "• Open coverage.html in a browser to view detailed coverage report"
    
    exit $exit_code
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --benchmark|-b)
            BENCHMARK=true
            shift
            ;;
        --no-race)
            RACE_DETECTION=false
            shift
            ;;
        --coverage-threshold)
            COVERAGE_THRESHOLD=$2
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose           Enable verbose output"
            echo "  -b, --benchmark         Run benchmark tests"
            echo "  --no-race              Disable race detection"
            echo "  --coverage-threshold N  Set coverage threshold (default: 70)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  VERBOSE=true          Enable verbose output"
            echo "  BENCHMARK=true        Run benchmark tests"
            echo "  RACE=false           Disable race detection"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main