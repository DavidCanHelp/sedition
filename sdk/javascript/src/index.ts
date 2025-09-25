/**
 * PoC Blockchain JavaScript SDK
 * Complete client library for interacting with PoC Blockchain nodes
 */

import axios, { AxiosInstance } from 'axios';
import { io, Socket } from 'socket.io-client';
import { EventEmitter } from 'events';

// Types
export interface NodeStatus {
  nodeId: string;
  isValidator: boolean;
  validatorAddress?: string;
  blockHeight: number;
  peerCount: number;
  pendingTxCount: number;
  networkStats: Record<string, any>;
  consensusActive: boolean;
  uptime: string;
}

export interface Block {
  header: BlockHeader;
  transactions: Transaction[];
  hash: string;
  signatures: Signature[];
}

export interface BlockHeader {
  height: number;
  previousHash: string;
  timestamp: string;
  proposer: string;
  stateRoot: string;
  txRoot: string;
}

export interface Transaction {
  id: string;
  from: string;
  to: string;
  amount: string;
  timestamp: string;
  data?: Record<string, any>;
  signature?: string;
}

export interface Signature {
  validatorId: string;
  signature: string;
  timestamp: string;
}

export interface Validator {
  address: string;
  stake: string;
  reputation: number;
  active: boolean;
}

export interface Contribution {
  type: 'CodeCommit' | 'Documentation' | 'BugFix' | 'Review';
  linesAdded?: number;
  linesModified?: number;
  testCoverage?: number;
  complexity?: number;
  documentation?: number;
}

export interface Balance {
  address: string;
  balance: string;
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface SDKConfig {
  nodeUrl: string;
  apiKey?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  enableWebSocket?: boolean;
}

// Main SDK Class
export class PoCBlockchainSDK extends EventEmitter {
  private client: AxiosInstance;
  private socket?: Socket;
  private config: Required<SDKConfig>;
  private authToken?: string;

  constructor(config: SDKConfig) {
    super();

    this.config = {
      nodeUrl: config.nodeUrl,
      apiKey: config.apiKey || '',
      timeout: config.timeout || 10000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 1000,
      enableWebSocket: config.enableWebSocket || false
    };

    // Initialize HTTP client
    this.client = axios.create({
      baseURL: this.config.nodeUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey && { 'X-API-Key': this.config.apiKey })
      }
    });

    // Add request interceptor for auth
    this.client.interceptors.request.use((config) => {
      if (this.authToken) {
        config.headers.Authorization = `Bearer ${this.authToken}`;
      }
      return config;
    });

    // Add response interceptor for retry logic
    this.client.interceptors.response.use(
      response => response,
      async error => {
        const config = error.config;
        if (!config || !config.retry) {
          config.retry = 0;
        }

        if (config.retry < this.config.retryAttempts) {
          config.retry++;
          await this.delay(this.config.retryDelay);
          return this.client(config);
        }

        return Promise.reject(error);
      }
    );

    // Initialize WebSocket if enabled
    if (this.config.enableWebSocket) {
      this.initWebSocket();
    }
  }

  // Authentication
  async authenticate(username: string, password: string): Promise<string> {
    try {
      const response = await this.client.post<APIResponse<{ token: string }>>('/api/auth/login', {
        username,
        password
      });

      if (response.data.success && response.data.data) {
        this.authToken = response.data.data.token;
        return this.authToken;
      }

      throw new Error(response.data.error || 'Authentication failed');
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Node Information
  async getStatus(): Promise<NodeStatus> {
    try {
      const response = await this.client.get<APIResponse<NodeStatus>>('/api/status');
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getPeers(): Promise<string[]> {
    try {
      const response = await this.client.get<APIResponse<string[]>>('/api/peers');
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Blockchain Operations
  async getBlocks(limit?: number): Promise<Block[]> {
    try {
      const response = await this.client.get<APIResponse<Block[]>>('/api/blocks', {
        params: { limit }
      });
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getBlock(height: number): Promise<Block> {
    try {
      const response = await this.client.get<APIResponse<Block>>(`/api/block/${height}`);
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getLatestBlock(): Promise<Block> {
    try {
      const response = await this.client.get<APIResponse<Block>>('/api/block/latest');
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Transaction Operations
  async sendTransaction(transaction: Omit<Transaction, 'id' | 'timestamp'>): Promise<{ txId: string }> {
    try {
      const response = await this.client.post<APIResponse<{ tx_id: string }>>('/api/transaction', transaction);
      const data = this.handleResponse(response);
      return { txId: data.tx_id };
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getTransaction(txId: string): Promise<Transaction> {
    try {
      const response = await this.client.get<APIResponse<Transaction>>(`/api/transaction/${txId}`);
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getPendingTransactions(): Promise<Transaction[]> {
    try {
      const response = await this.client.get<APIResponse<Transaction[]>>('/api/transactions/pending');
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Account Operations
  async getBalance(address: string): Promise<Balance> {
    try {
      const response = await this.client.get<APIResponse<Balance>>(`/api/balance/${address}`);
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Validator Operations
  async getValidators(): Promise<Validator[]> {
    try {
      const response = await this.client.get<APIResponse<Validator[]>>('/api/validators');
      return this.handleResponse(response);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async submitContribution(contribution: Contribution): Promise<void> {
    try {
      await this.client.post('/api/contribute', contribution);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // WebSocket Operations
  private initWebSocket(): void {
    const wsUrl = this.config.nodeUrl.replace(/^http/, 'ws');

    this.socket = io(wsUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    this.socket.on('connect', () => {
      this.emit('connected');
      console.log('WebSocket connected');
    });

    this.socket.on('disconnect', () => {
      this.emit('disconnected');
      console.log('WebSocket disconnected');
    });

    this.socket.on('block', (block: Block) => {
      this.emit('block', block);
    });

    this.socket.on('transaction', (tx: Transaction) => {
      this.emit('transaction', tx);
    });

    this.socket.on('error', (error: Error) => {
      this.emit('error', error);
    });
  }

  subscribeToBlocks(callback: (block: Block) => void): void {
    this.on('block', callback);
    if (this.socket) {
      this.socket.emit('subscribe', { channel: 'blocks' });
    }
  }

  subscribeToTransactions(callback: (tx: Transaction) => void): void {
    this.on('transaction', callback);
    if (this.socket) {
      this.socket.emit('subscribe', { channel: 'transactions' });
    }
  }

  unsubscribe(channel: 'blocks' | 'transactions'): void {
    if (this.socket) {
      this.socket.emit('unsubscribe', { channel });
    }
    this.removeAllListeners(channel === 'blocks' ? 'block' : 'transaction');
  }

  // Utility Methods
  private handleResponse<T>(response: any): T {
    if (response.data.success) {
      return response.data.data;
    }
    throw new Error(response.data.error || 'Request failed');
  }

  private handleError(error: any): Error {
    if (error.response) {
      // Server responded with error
      const message = error.response.data?.error || error.response.statusText;
      return new Error(`API Error: ${message} (${error.response.status})`);
    } else if (error.request) {
      // No response received
      return new Error('Network error: No response from server');
    }
    // Something else happened
    return error;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Cleanup
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
    }
    this.removeAllListeners();
  }
}

// Convenience function
export function createClient(config: SDKConfig): PoCBlockchainSDK {
  return new PoCBlockchainSDK(config);
}

// Export everything
export default PoCBlockchainSDK;