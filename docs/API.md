# PoC Blockchain API Documentation

## Overview

The PoC Blockchain provides a comprehensive REST API for interacting with the blockchain network. All API responses follow a consistent format and include proper error handling.

## Base URL

```
Production: https://api.poc-blockchain.io
Staging: https://staging-api.poc-blockchain.io
Local: http://localhost:8080
```

## Authentication

### API Key Authentication

Include your API key in the request header:

```http
X-API-Key: your-api-key-here
```

### Bearer Token Authentication

For authenticated endpoints, first obtain a token:

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure-password"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refreshToken": "refresh-token-here",
    "expiresIn": 3600
  }
}
```

Include the token in subsequent requests:

```http
Authorization: Bearer your-token-here
```

## Response Format

All API responses follow this structure:

```json
{
  "success": true|false,
  "data": { ... },
  "error": "error message if success is false"
}
```

## Rate Limiting

- Default: 100 requests per minute
- Authenticated: 1000 requests per minute
- Headers returned:
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Endpoints

### Node Information

#### Get Node Status
```http
GET /api/status
```

Response:
```json
{
  "success": true,
  "data": {
    "nodeId": "abc123...",
    "isValidator": true,
    "validatorAddress": "validator1",
    "blockHeight": 12345,
    "peerCount": 8,
    "pendingTxCount": 25,
    "networkStats": {
      "messagesReceived": 10000,
      "messagesSent": 9500,
      "bytesReceived": 5242880,
      "bytesSent": 4194304
    },
    "consensusActive": true,
    "uptime": "72h15m30s"
  }
}
```

#### Get Connected Peers
```http
GET /api/peers
```

Response:
```json
{
  "success": true,
  "data": [
    "peer1:8545",
    "peer2:8545",
    "peer3:8545"
  ]
}
```

### Blockchain Operations

#### Get Recent Blocks
```http
GET /api/blocks?limit=10&offset=0
```

Parameters:
- `limit` (optional): Number of blocks to return (default: 10, max: 100)
- `offset` (optional): Pagination offset (default: 0)

Response:
```json
{
  "success": true,
  "data": [
    {
      "header": {
        "height": 12345,
        "previousHash": "0x...",
        "timestamp": "2024-01-01T00:00:00Z",
        "proposer": "validator1",
        "stateRoot": "0x...",
        "txRoot": "0x..."
      },
      "transactions": [...],
      "hash": "0x...",
      "signatures": [...]
    }
  ]
}
```

#### Get Specific Block
```http
GET /api/block/{height}
```

Response: Single block object

#### Get Latest Block
```http
GET /api/block/latest
```

Response: Single block object

### Transaction Operations

#### Submit Transaction
```http
POST /api/transaction
Content-Type: application/json

{
  "from": "alice",
  "to": "bob",
  "amount": "100",
  "data": {
    "memo": "Payment for services"
  }
}
```

Response:
```json
{
  "success": true,
  "data": {
    "txId": "tx_1234567890"
  }
}
```

#### Get Transaction
```http
GET /api/transaction/{txId}
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "tx_1234567890",
    "from": "alice",
    "to": "bob",
    "amount": "100",
    "timestamp": "2024-01-01T00:00:00Z",
    "data": { ... },
    "signature": "0x...",
    "status": "confirmed",
    "blockHeight": 12345
  }
}
```

#### Get Pending Transactions
```http
GET /api/transactions/pending
```

Response: Array of pending transactions

### Account Operations

#### Get Balance
```http
GET /api/balance/{address}
```

Response:
```json
{
  "success": true,
  "data": {
    "address": "alice",
    "balance": "10000"
  }
}
```

#### Get Account History
```http
GET /api/account/{address}/history?limit=20
```

Response: Array of transactions involving the address

### Validator Operations

#### Get Validators
```http
GET /api/validators
```

Response:
```json
{
  "success": true,
  "data": [
    {
      "address": "validator1",
      "stake": "10000",
      "reputation": 8.5,
      "active": true,
      "lastBlock": 12340,
      "totalBlocks": 500
    }
  ]
}
```

#### Submit Contribution (Validators Only)
```http
POST /api/contribute
Authorization: Bearer validator-token
Content-Type: application/json

{
  "type": "CodeCommit",
  "linesAdded": 500,
  "linesModified": 100,
  "testCoverage": 90.5,
  "complexity": 5.2,
  "documentation": 85.0
}
```

Response:
```json
{
  "success": true,
  "data": "Contribution submitted successfully"
}
```

### Metrics & Monitoring

#### Get Prometheus Metrics
```http
GET /metrics
```

Response: Prometheus-formatted metrics

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "checks": {
    "database": "ok",
    "network": "ok",
    "consensus": "ok"
  }
}
```

## WebSocket API

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('wss://api.poc-blockchain.io/ws');

// Subscribe to events
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['blocks', 'transactions']
}));

// Handle messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch(data.type) {
    case 'block':
      console.log('New block:', data.block);
      break;
    case 'transaction':
      console.log('New transaction:', data.transaction);
      break;
  }
};
```

### WebSocket Events

- `block`: New block added to chain
- `transaction`: New transaction in mempool
- `validator`: Validator status change
- `peer`: Peer connection/disconnection

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## SDK Usage Examples

### JavaScript/TypeScript

```typescript
import { PoCBlockchainSDK } from '@poc-blockchain/sdk';

const client = new PoCBlockchainSDK({
  nodeUrl: 'https://api.poc-blockchain.io',
  apiKey: 'your-api-key'
});

// Get status
const status = await client.getStatus();

// Send transaction
const tx = await client.sendTransaction({
  from: 'alice',
  to: 'bob',
  amount: '100'
});

// Subscribe to blocks
client.subscribeToBlocks((block) => {
  console.log('New block:', block);
});
```

### Python

```python
from poc_blockchain import Client

client = Client(
    node_url='https://api.poc-blockchain.io',
    api_key='your-api-key'
)

# Get status
status = client.get_status()

# Send transaction
tx = client.send_transaction(
    from_addr='alice',
    to_addr='bob',
    amount='100'
)
```

### Go

```go
import "github.com/davidcanhelp/sedition/sdk/go"

client := sdk.NewClient("https://api.poc-blockchain.io", "your-api-key")

// Get status
status, err := client.GetStatus()

// Send transaction
tx, err := client.SendTransaction(&sdk.Transaction{
    From: "alice",
    To: "bob",
    Amount: "100",
})
```

## Postman Collection

Download our Postman collection for easy API testing:
[Download Postman Collection](https://api.poc-blockchain.io/postman-collection.json)

## OpenAPI Specification

Access the OpenAPI 3.0 specification:
[OpenAPI Spec](https://api.poc-blockchain.io/openapi.json)

## Support

- GitHub Issues: https://github.com/davidcanhelp/sedition/issues
- Discord: https://discord.gg/poc-blockchain
- Email: support@poc-blockchain.io