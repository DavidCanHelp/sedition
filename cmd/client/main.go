package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
)

// RPCRequest represents a JSON-RPC request
type RPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
	ID      interface{}     `json:"id"`
}

// RPCResponse represents a JSON-RPC response
type RPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
	ID      interface{}     `json:"id"`
}

// RPCError represents a JSON-RPC error
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

func main() {
	var (
		rpcURL  string
		command string
	)

	flag.StringVar(&rpcURL, "rpc", "http://localhost:8545", "RPC server URL")
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		printUsage()
		os.Exit(1)
	}

	command = args[0]
	cmdArgs := args[1:]

	switch command {
	case "accounts":
		getAccounts(rpcURL)
	case "balance":
		if len(cmdArgs) < 1 {
			fmt.Println("Usage: client balance <address>")
			os.Exit(1)
		}
		getBalance(rpcURL, cmdArgs[0])
	case "send":
		if len(cmdArgs) < 3 {
			fmt.Println("Usage: client send <from> <to> <amount>")
			os.Exit(1)
		}
		sendTransaction(rpcURL, cmdArgs[0], cmdArgs[1], cmdArgs[2])
	case "tx":
		if len(cmdArgs) < 1 {
			fmt.Println("Usage: client tx <hash>")
			os.Exit(1)
		}
		getTransaction(rpcURL, cmdArgs[0])
	case "block":
		if len(cmdArgs) < 1 {
			fmt.Println("Usage: client block <number|latest>")
			os.Exit(1)
		}
		getBlock(rpcURL, cmdArgs[0])
	case "blocknumber":
		getBlockNumber(rpcURL)
	case "mining":
		getMiningStatus(rpcURL)
	case "peers":
		getPeerCount(rpcURL)
	case "version":
		getClientVersion(rpcURL)
	default:
		fmt.Printf("Unknown command: %s\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Sedition Blockchain Client")
	fmt.Println("\nUsage: client [flags] <command> [args]")
	fmt.Println("\nFlags:")
	fmt.Println("  -rpc string    RPC server URL (default: http://localhost:8545)")
	fmt.Println("\nCommands:")
	fmt.Println("  accounts                List all accounts")
	fmt.Println("  balance <address>       Get balance of an address")
	fmt.Println("  send <from> <to> <amt>  Send transaction")
	fmt.Println("  tx <hash>              Get transaction by hash")
	fmt.Println("  block <number>         Get block by number")
	fmt.Println("  blocknumber            Get current block number")
	fmt.Println("  mining                 Get mining status")
	fmt.Println("  peers                  Get peer count")
	fmt.Println("  version                Get client version")
}

func makeRPCCall(url string, method string, params interface{}) (*RPCResponse, error) {
	// Prepare request
	var paramsJSON json.RawMessage
	if params != nil {
		data, err := json.Marshal(params)
		if err != nil {
			return nil, err
		}
		paramsJSON = data
	}

	req := RPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  paramsJSON,
		ID:      1,
	}

	// Marshal request
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	// Make HTTP request
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Parse response
	var rpcResp RPCResponse
	if err := json.Unmarshal(respBody, &rpcResp); err != nil {
		return nil, err
	}

	return &rpcResp, nil
}

func getAccounts(url string) {
	resp, err := makeRPCCall(url, "eth_accounts", nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var accounts []string
	if err := json.Unmarshal(resp.Result, &accounts); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	fmt.Println("Accounts:")
	for i, account := range accounts {
		fmt.Printf("  [%d] %s\n", i, account)
	}
}

func getBalance(url string, address string) {
	params := []string{address, "latest"}
	resp, err := makeRPCCall(url, "eth_getBalance", params)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var balance string
	if err := json.Unmarshal(resp.Result, &balance); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	fmt.Printf("Balance of %s: %s\n", address, balance)
}

func sendTransaction(url string, from, to, amount string) {
	// Convert amount to hex
	// For simplicity, assuming amount is in wei
	params := map[string]string{
		"from":     from,
		"to":       to,
		"value":    fmt.Sprintf("0x%x", parseAmount(amount)),
		"gas":      "0x5208",       // 21000 gas
		"gasPrice": "0x3b9aca00",   // 1 Gwei
	}

	resp, err := makeRPCCall(url, "eth_sendTransaction", params)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var txHash string
	if err := json.Unmarshal(resp.Result, &txHash); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	fmt.Printf("Transaction sent: %s\n", txHash)
}

func getTransaction(url string, hash string) {
	params := []string{hash}
	resp, err := makeRPCCall(url, "eth_getTransactionByHash", params)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var tx map[string]interface{}
	if err := json.Unmarshal(resp.Result, &tx); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	if tx == nil {
		fmt.Println("Transaction not found")
		return
	}

	fmt.Println("Transaction Details:")
	fmt.Printf("  Hash: %s\n", tx["hash"])
	fmt.Printf("  From: %s\n", tx["from"])
	fmt.Printf("  To: %s\n", tx["to"])
	fmt.Printf("  Value: %s\n", tx["value"])
	fmt.Printf("  Gas: %s\n", tx["gas"])
	fmt.Printf("  Gas Price: %s\n", tx["gasPrice"])
	fmt.Printf("  Nonce: %s\n", tx["nonce"])
}

func getBlock(url string, number string) {
	params := []interface{}{number, true} // true = include full transactions
	resp, err := makeRPCCall(url, "eth_getBlockByNumber", params)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var block map[string]interface{}
	if err := json.Unmarshal(resp.Result, &block); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	if block == nil {
		fmt.Println("Block not found")
		return
	}

	fmt.Println("Block Details:")
	fmt.Printf("  Number: %s\n", block["number"])
	fmt.Printf("  Hash: %s\n", block["hash"])
	fmt.Printf("  Parent Hash: %s\n", block["parentHash"])
	fmt.Printf("  Timestamp: %s\n", block["timestamp"])
	fmt.Printf("  Miner: %s\n", block["miner"])

	if txs, ok := block["transactions"].([]interface{}); ok {
		fmt.Printf("  Transactions: %d\n", len(txs))
		for i, tx := range txs {
			if txMap, ok := tx.(map[string]interface{}); ok {
				fmt.Printf("    [%d] %s\n", i, txMap["hash"])
			}
		}
	}
}

func getBlockNumber(url string) {
	resp, err := makeRPCCall(url, "eth_blockNumber", nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var blockNum string
	if err := json.Unmarshal(resp.Result, &blockNum); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	fmt.Printf("Current block number: %s\n", blockNum)
}

func getMiningStatus(url string) {
	resp, err := makeRPCCall(url, "eth_mining", nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var mining bool
	if err := json.Unmarshal(resp.Result, &mining); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	fmt.Printf("Mining: %v\n", mining)
}

func getPeerCount(url string) {
	resp, err := makeRPCCall(url, "net_peerCount", nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var count string
	if err := json.Unmarshal(resp.Result, &count); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	fmt.Printf("Peer count: %s\n", count)
}

func getClientVersion(url string) {
	resp, err := makeRPCCall(url, "web3_clientVersion", nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if resp.Error != nil {
		fmt.Printf("RPC Error: %s\n", resp.Error.Message)
		return
	}

	var version string
	if err := json.Unmarshal(resp.Result, &version); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		return
	}

	fmt.Printf("Client version: %s\n", version)
}

func parseAmount(amount string) int64 {
	// Simple parsing - in production use big.Int
	var value int64
	fmt.Sscanf(strings.TrimSpace(amount), "%d", &value)
	return value
}