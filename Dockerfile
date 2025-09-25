# Multi-stage build for PoC Blockchain

# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make gcc musl-dev

# Set working directory
WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -o server demo/server.go

# Runtime stage
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1000 blockchain && \
    adduser -D -u 1000 -G blockchain blockchain

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/server .

# Copy static files if any
COPY --from=builder /build/demo/static ./static

# Create data directory
RUN mkdir -p /data && chown -R blockchain:blockchain /data

# Switch to non-root user
USER blockchain

# Expose ports
EXPOSE 8080 8545

# Volume for persistent data
VOLUME ["/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/api/status || exit 1

# Default command
ENTRYPOINT ["/app/server"]
CMD ["--data", "/data"]