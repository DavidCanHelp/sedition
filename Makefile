.PHONY: all build test clean lint fmt vet install-tools coverage benchmark help

# Variables
BINARY_NAME=sedition
MAIN_PATH=./demo/simple_poc_demo.go
GO=go
GOFLAGS=-v
COVERAGE_FILE=coverage.txt
BENCHMARK_FILE=benchmark.txt

# Default target
all: clean fmt vet lint test build

# Quick test - fastest way to validate everything works
quick:
	@./quick_test.sh

# Full test - comprehensive test suite
test-all:
	@./test_all.sh

# Help target
help:
	@echo "Available targets:"
	@echo "  make build       - Build the binary"
	@echo "  make test        - Run all tests"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make coverage    - Generate test coverage report"
	@echo "  make benchmark   - Run benchmarks"
	@echo "  make lint        - Run golangci-lint"
	@echo "  make fmt         - Format code with gofmt"
	@echo "  make vet         - Run go vet"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make install-tools - Install required tools"
	@echo "  make check-tech-debt - Run tech debt analysis"
	@echo "  make fix-imports - Fix and organize imports"
	@echo "  make security    - Run security checks"
	@echo "  make all         - Run everything (clean, fmt, vet, lint, test, build)"

# Build the binary
build:
	@echo "Building $(BINARY_NAME)..."
	@$(GO) build $(GOFLAGS) -o $(BINARY_NAME) $(MAIN_PATH)
	@echo "Build complete: $(BINARY_NAME)"

# Run all tests with coverage
test:
	@echo "Running all tests with coverage..."
	@./run_tests.sh

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	@$(GO) test $(GOFLAGS) -race -short ./...

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	@$(GO) test $(GOFLAGS) -race -run Integration ./...

# Generate test coverage
coverage:
	@echo "Generating test coverage..."
	@$(GO) test -coverprofile=$(COVERAGE_FILE) -covermode=atomic ./...
	@$(GO) tool cover -html=$(COVERAGE_FILE) -o coverage.html
	@echo "Coverage report generated: coverage.html"
	@echo "Coverage summary:"
	@$(GO) tool cover -func=$(COVERAGE_FILE) | tail -1

# Run benchmarks
benchmark:
	@echo "Running benchmarks..."
	@$(GO) test -bench=. -benchmem -run=^$$ ./... | tee $(BENCHMARK_FILE)
	@echo "Benchmark results saved to $(BENCHMARK_FILE)"

# Run linter
lint:
	@echo "Running golangci-lint..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not installed. Run 'make install-tools' first."; \
		exit 1; \
	fi

# Format code
fmt:
	@echo "Formatting code..."
	@$(GO) fmt ./...
	@echo "Code formatting complete"

# Run go vet
vet:
	@echo "Running go vet..."
	@$(GO) vet ./...
	@echo "Vet complete"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -f $(BINARY_NAME)
	@rm -f $(COVERAGE_FILE) coverage.html
	@rm -f $(BENCHMARK_FILE)
	@rm -rf dist/
	@echo "Clean complete"

# Install required tools
install-tools:
	@echo "Installing tools..."
	@echo "Installing golangci-lint..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.55.2
	@echo "Installing goimports..."
	@go install golang.org/x/tools/cmd/goimports@latest
	@echo "Installing mockery..."
	@go install github.com/vektra/mockery/v2@latest
	@echo "Installing godoc..."
	@go install golang.org/x/tools/cmd/godoc@latest
	@echo "Tools installation complete"

# Fix imports
fix-imports:
	@echo "Fixing imports..."
	@if command -v goimports >/dev/null 2>&1; then \
		goimports -w -local github.com/davidcanhelp/sedition .; \
	else \
		echo "goimports not installed. Run 'make install-tools' first."; \
		exit 1; \
	fi
	@echo "Import fixing complete"

# Run security checks
security:
	@echo "Running security checks..."
	@echo "Running gosec..."
	@if command -v gosec >/dev/null 2>&1; then \
		gosec -fmt json -out gosec-report.json ./... || true; \
		echo "Security report saved to gosec-report.json"; \
	else \
		go install github.com/securego/gosec/v2/cmd/gosec@latest; \
		gosec -fmt json -out gosec-report.json ./... || true; \
		echo "Security report saved to gosec-report.json"; \
	fi
	@echo "Checking for vulnerabilities with govulncheck..."
	@if command -v govulncheck >/dev/null 2>&1; then \
		govulncheck ./...; \
	else \
		go install golang.org/x/vuln/cmd/govulncheck@latest; \
		govulncheck ./...; \
	fi

# Check technical debt
check-tech-debt:
	@echo "Analyzing technical debt..."
	@./remediate_tech_debt.sh

# Quick check - format, vet, and test
quick: fmt vet test-unit

# CI/CD pipeline simulation
ci: clean install-tools fmt vet lint test coverage

# Development setup
setup: install-tools
	@echo "Setting up development environment..."
	@$(GO) mod download
	@$(GO) mod tidy
	@echo "Development environment ready"

# Generate documentation
docs:
	@echo "Generating documentation..."
	@if command -v godoc >/dev/null 2>&1; then \
		echo "Starting godoc server on http://localhost:6060"; \
		godoc -http=:6060; \
	else \
		echo "godoc not installed. Run 'make install-tools' first."; \
		exit 1; \
	fi

# Run specific test
test-specific:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-specific TEST=TestFunctionName"; \
		exit 1; \
	fi
	@echo "Running test: $(TEST)"
	@$(GO) test -v -run $(TEST) ./...

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	@$(GO) get -u ./...
	@$(GO) mod tidy
	@echo "Dependencies updated"

# Build for multiple platforms
build-all:
	@echo "Building for multiple platforms..."
	@GOOS=linux GOARCH=amd64 $(GO) build -o dist/$(BINARY_NAME)-linux-amd64 $(MAIN_PATH)
	@GOOS=darwin GOARCH=amd64 $(GO) build -o dist/$(BINARY_NAME)-darwin-amd64 $(MAIN_PATH)
	@GOOS=darwin GOARCH=arm64 $(GO) build -o dist/$(BINARY_NAME)-darwin-arm64 $(MAIN_PATH)
	@GOOS=windows GOARCH=amd64 $(GO) build -o dist/$(BINARY_NAME)-windows-amd64.exe $(MAIN_PATH)
	@echo "Multi-platform build complete in dist/"

# Check for outdated dependencies
check-deps:
	@echo "Checking for outdated dependencies..."
	@$(GO) list -u -m all

# Run go mod tidy and verify
tidy:
	@echo "Running go mod tidy..."
	@$(GO) mod tidy
	@echo "Verifying modules..."
	@$(GO) mod verify

# Count lines of code
loc:
	@echo "Lines of code statistics:"
	@find . -name "*.go" -not -path "./vendor/*" | xargs wc -l | sort -n