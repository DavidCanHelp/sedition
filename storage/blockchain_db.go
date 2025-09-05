// Package storage implements blockchain persistence and state management
package storage

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/opt"
	"github.com/syndtr/goleveldb/leveldb/util"
)

// Block represents a blockchain block
type Block struct {
	Height       uint64
	PreviousHash []byte
	Hash         []byte
	Timestamp    time.Time
	Data         []byte
	Proposer     string
	Signatures   map[string][]byte
	Commits      []Commit
	StateRoot    []byte
}

// EnhancedValidator represents a consensus validator
type EnhancedValidator struct {
	Address    string
	PublicKey  []byte
	Stake      int64
	Reputation float64
	Active     bool
}

// Commit represents block commitment
type Commit struct {
	BlockHash  []byte
	Height     int64
	Signatures map[string][]byte
	Hash       []byte
}

// Database key prefixes
var (
	// Block storage
	blockHeightKey    = []byte("bh") // block height -> block hash
	blockHashPrefix   = []byte("b")  // block hash -> block data
	blockHeaderPrefix = []byte("h")  // block hash -> block header

	// Chain metadata
	chainTipKey    = []byte("tip")     // current chain tip
	chainHeightKey = []byte("height")  // current chain height
	genesisKey     = []byte("genesis") // genesis block

	// Transaction/Commit storage
	commitPrefix      = []byte("c")  // commit hash -> commit data
	commitIndexPrefix = []byte("ci") // block height -> commit hashes

	// State storage
	statePrefix     = []byte("s") // state key -> state value
	validatorPrefix = []byte("v") // validator address -> validator data

	// Indexes
	timeIndexPrefix     = []byte("t") // timestamp -> block hashes
	proposerIndexPrefix = []byte("p") // proposer -> block hashes

	// Checkpoints
	checkpointPrefix = []byte("cp") // height -> checkpoint data
)

// BlockchainDB implements persistent storage for the blockchain
type BlockchainDB struct {
	mu sync.RWMutex

	db   *leveldb.DB
	path string

	// Caches
	blockCache *LRUCache
	stateCache *LRUCache

	// Write batch
	batch     *leveldb.Batch
	batchSize int

	// Metrics
	reads       int64
	writes      int64
	cacheHits   int64
	cacheMisses int64
}

// NewBlockchainDB creates a new blockchain database
func NewBlockchainDB(path string) (*BlockchainDB, error) {
	// Open LevelDB
	opts := &opt.Options{
		Compression:        opt.SnappyCompression,
		WriteBuffer:        16 * 1024 * 1024,  // 16MB
		BlockCacheCapacity: 128 * 1024 * 1024, // 128MB
	}

	db, err := leveldb.OpenFile(filepath.Join(path, "blocks"), opts)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	bdb := &BlockchainDB{
		db:         db,
		path:       path,
		blockCache: NewLRUCache(100),
		stateCache: NewLRUCache(1000),
		batch:      new(leveldb.Batch),
	}

	return bdb, nil
}

// Close closes the database
func (db *BlockchainDB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Flush any pending batch
	if db.batchSize > 0 {
		db.flushBatch()
	}

	return db.db.Close()
}

// StoreBlock stores a block in the database
func (db *BlockchainDB) StoreBlock(block *Block) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Serialize block
	blockData, err := json.Marshal(block)
	if err != nil {
		return fmt.Errorf("failed to serialize block: %w", err)
	}

	// Store block by hash
	blockKey := append(blockHashPrefix, block.Hash...)
	db.batch.Put(blockKey, blockData)

	// Store block height index
	heightKey := makeHeightKey(int64(block.Height))
	db.batch.Put(heightKey, block.Hash)

	// Store time index
	timeKey := makeTimeKey(block.Timestamp, block.Hash)
	db.batch.Put(timeKey, []byte{1})

	// Store proposer index
	proposerKey := makeProposerKey(block.Proposer, int64(block.Height))
	db.batch.Put(proposerKey, block.Hash)

	// Store commits
	for _, commit := range block.Commits {
		if err := db.storeCommit(&commit, int64(block.Height)); err != nil {
			return err
		}
	}

	// Update chain tip if this is the new highest block
	currentHeight, err := db.getChainHeight()
	if err != nil || block.Height > uint64(currentHeight) {
		db.batch.Put(chainTipKey, block.Hash)
		db.batch.Put(chainHeightKey, encodeUint64(uint64(block.Height)))
	}

	// Add to cache
	db.blockCache.Put(string(block.Hash), block)

	// Flush batch if it gets too large
	db.batchSize++
	if db.batchSize >= 100 {
		return db.flushBatch()
	}

	db.writes++
	return nil
}

// GetBlock retrieves a block by hash
func (db *BlockchainDB) GetBlock(hash []byte) (*Block, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Check cache
	if cached := db.blockCache.Get(string(hash)); cached != nil {
		db.cacheHits++
		return cached.(*Block), nil
	}
	db.cacheMisses++

	// Get from database
	blockKey := append(blockHashPrefix, hash...)
	data, err := db.db.Get(blockKey, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return nil, errors.New("block not found")
		}
		return nil, err
	}

	// Deserialize block
	var block Block
	if err := json.Unmarshal(data, &block); err != nil {
		return nil, fmt.Errorf("failed to deserialize block: %w", err)
	}

	// Add to cache
	db.blockCache.Put(string(hash), &block)

	db.reads++
	return &block, nil
}

// GetBlockByHeight retrieves a block by height
func (db *BlockchainDB) GetBlockByHeight(height int64) (*Block, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Get block hash at height
	heightKey := makeHeightKey(height)
	hash, err := db.db.Get(heightKey, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return nil, errors.New("block not found at height")
		}
		return nil, err
	}

	// Get block by hash
	return db.GetBlock(hash)
}

// GetLatestBlock returns the latest block
func (db *BlockchainDB) GetLatestBlock() (*Block, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Get chain tip
	tipHash, err := db.db.Get(chainTipKey, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return nil, errors.New("no chain tip")
		}
		return nil, err
	}

	return db.GetBlock(tipHash)
}

// GetChainHeight returns the current chain height
func (db *BlockchainDB) GetChainHeight() (int64, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	return db.getChainHeight()
}

// getChainHeight internal version without lock
func (db *BlockchainDB) getChainHeight() (int64, error) {
	data, err := db.db.Get(chainHeightKey, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return -1, nil
		}
		return -1, err
	}

	return int64(decodeUint64(data)), nil
}

// StoreValidator stores validator state
func (db *BlockchainDB) StoreValidator(validator *EnhancedValidator) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	data, err := json.Marshal(validator)
	if err != nil {
		return fmt.Errorf("failed to serialize validator: %w", err)
	}

	key := append(validatorPrefix, []byte(validator.Address)...)
	db.batch.Put(key, data)

	db.batchSize++
	if db.batchSize >= 100 {
		return db.flushBatch()
	}

	return nil
}

// GetValidator retrieves validator state
func (db *BlockchainDB) GetValidator(address string) (*EnhancedValidator, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	key := append(validatorPrefix, []byte(address)...)
	data, err := db.db.Get(key, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return nil, errors.New("validator not found")
		}
		return nil, err
	}

	var validator EnhancedValidator
	if err := json.Unmarshal(data, &validator); err != nil {
		return nil, fmt.Errorf("failed to deserialize validator: %w", err)
	}

	return &validator, nil
}

// GetAllValidators retrieves all validators
func (db *BlockchainDB) GetAllValidators() ([]*EnhancedValidator, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	validators := make([]*EnhancedValidator, 0)

	iter := db.db.NewIterator(util.BytesPrefix(validatorPrefix), nil)
	defer iter.Release()

	for iter.Next() {
		var validator EnhancedValidator
		if err := json.Unmarshal(iter.Value(), &validator); err != nil {
			continue
		}
		validators = append(validators, &validator)
	}

	if err := iter.Error(); err != nil {
		return nil, err
	}

	return validators, nil
}

// StoreState stores arbitrary state data
func (db *BlockchainDB) StoreState(key string, value interface{}) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to serialize state: %w", err)
	}

	stateKey := append(statePrefix, []byte(key)...)
	db.batch.Put(stateKey, data)

	// Update cache
	db.stateCache.Put(key, value)

	db.batchSize++
	if db.batchSize >= 100 {
		return db.flushBatch()
	}

	return nil
}

// GetState retrieves state data
func (db *BlockchainDB) GetState(key string, value interface{}) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Check cache
	if cached := db.stateCache.Get(key); cached != nil {
		// Type assert and copy
		*value.(*interface{}) = cached
		db.cacheHits++
		return nil
	}
	db.cacheMisses++

	stateKey := append(statePrefix, []byte(key)...)
	data, err := db.db.Get(stateKey, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return errors.New("state not found")
		}
		return err
	}

	if err := json.Unmarshal(data, value); err != nil {
		return fmt.Errorf("failed to deserialize state: %w", err)
	}

	// Update cache
	db.stateCache.Put(key, value)

	db.reads++
	return nil
}

// storeCommit stores a commit
func (db *BlockchainDB) storeCommit(commit *Commit, blockHeight int64) error {
	// Serialize commit
	commitData, err := json.Marshal(commit)
	if err != nil {
		return fmt.Errorf("failed to serialize commit: %w", err)
	}

	// Store commit by hash
	commitKey := append(commitPrefix, commit.Hash...)
	db.batch.Put(commitKey, commitData)

	// Add to block index
	indexKey := makeCommitIndexKey(blockHeight, commit.Hash)
	db.batch.Put(indexKey, []byte{1})

	return nil
}

// GetCommit retrieves a commit by hash
func (db *BlockchainDB) GetCommit(hash []byte) (*Commit, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	commitKey := append(commitPrefix, hash...)
	data, err := db.db.Get(commitKey, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return nil, errors.New("commit not found")
		}
		return nil, err
	}

	var commit Commit
	if err := json.Unmarshal(data, &commit); err != nil {
		return nil, fmt.Errorf("failed to deserialize commit: %w", err)
	}

	return &commit, nil
}

// GetBlocksByTimeRange retrieves blocks within a time range
func (db *BlockchainDB) GetBlocksByTimeRange(start, end time.Time) ([]*Block, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	blocks := make([]*Block, 0)

	startKey := makeTimeKey(start, nil)
	endKey := makeTimeKey(end, []byte{0xff})

	iter := db.db.NewIterator(&util.Range{Start: startKey, Limit: endKey}, nil)
	defer iter.Release()

	for iter.Next() {
		// Extract block hash from key
		hash := extractHashFromTimeKey(iter.Key())
		block, err := db.GetBlock(hash)
		if err != nil {
			continue
		}
		blocks = append(blocks, block)
	}

	if err := iter.Error(); err != nil {
		return nil, err
	}

	return blocks, nil
}

// GetBlocksByProposer retrieves blocks by proposer
func (db *BlockchainDB) GetBlocksByProposer(proposer string) ([]*Block, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	blocks := make([]*Block, 0)

	prefix := append(proposerIndexPrefix, []byte(proposer)...)
	iter := db.db.NewIterator(util.BytesPrefix(prefix), nil)
	defer iter.Release()

	for iter.Next() {
		hash := iter.Value()
		block, err := db.GetBlock(hash)
		if err != nil {
			continue
		}
		blocks = append(blocks, block)
	}

	if err := iter.Error(); err != nil {
		return nil, err
	}

	return blocks, nil
}

// CreateCheckpoint creates a checkpoint at the specified height
func (db *BlockchainDB) CreateCheckpoint(height int64) (*Checkpoint, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Get block at height
	block, err := db.GetBlockByHeight(height)
	if err != nil {
		return nil, err
	}

	// Get all validator states
	validators, err := db.GetAllValidators()
	if err != nil {
		return nil, err
	}

	checkpoint := &Checkpoint{
		Height:     height,
		BlockHash:  block.Hash,
		StateRoot:  block.StateRoot,
		Timestamp:  time.Now(),
		Validators: validators,
	}

	// Serialize checkpoint
	data, err := json.Marshal(checkpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize checkpoint: %w", err)
	}

	// Store checkpoint
	key := append(checkpointPrefix, encodeUint64(uint64(height))...)
	if err := db.db.Put(key, data, nil); err != nil {
		return nil, err
	}

	return checkpoint, nil
}

// GetCheckpoint retrieves a checkpoint
func (db *BlockchainDB) GetCheckpoint(height int64) (*Checkpoint, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	key := append(checkpointPrefix, encodeUint64(uint64(height))...)
	data, err := db.db.Get(key, nil)
	if err != nil {
		if err == leveldb.ErrNotFound {
			return nil, errors.New("checkpoint not found")
		}
		return nil, err
	}

	var checkpoint Checkpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, fmt.Errorf("failed to deserialize checkpoint: %w", err)
	}

	return &checkpoint, nil
}

// Prune removes blocks before the specified height
func (db *BlockchainDB) Prune(beforeHeight int64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Get latest checkpoint
	var latestCheckpoint int64 = -1
	iter := db.db.NewIterator(util.BytesPrefix(checkpointPrefix), nil)
	for iter.Next() {
		height := int64(decodeUint64(iter.Key()[len(checkpointPrefix):]))
		if height > latestCheckpoint && height < beforeHeight {
			latestCheckpoint = height
		}
	}
	iter.Release()

	if latestCheckpoint == -1 {
		return errors.New("cannot prune without checkpoint")
	}

	// Delete blocks before checkpoint
	batch := new(leveldb.Batch)
	for height := int64(0); height < latestCheckpoint; height++ {
		// Get block hash
		heightKey := makeHeightKey(height)
		hash, err := db.db.Get(heightKey, nil)
		if err != nil {
			continue
		}

		// Delete block
		blockKey := append(blockHashPrefix, hash...)
		batch.Delete(blockKey)
		batch.Delete(heightKey)

		// Remove from cache
		db.blockCache.Remove(string(hash))
	}

	return db.db.Write(batch, nil)
}

// flushBatch writes the current batch to disk
func (db *BlockchainDB) flushBatch() error {
	if db.batchSize == 0 {
		return nil
	}

	err := db.db.Write(db.batch, nil)
	db.batch = new(leveldb.Batch)
	db.batchSize = 0

	return err
}

// GetMetrics returns database metrics
func (db *BlockchainDB) GetMetrics() *DBMetrics {
	db.mu.RLock()
	defer db.mu.RUnlock()

	stats, _ := db.db.GetProperty("leveldb.stats")

	return &DBMetrics{
		Reads:        db.reads,
		Writes:       db.writes,
		CacheHits:    db.cacheHits,
		CacheMisses:  db.cacheMisses,
		LevelDBStats: stats,
	}
}

// Checkpoint represents a state checkpoint
type Checkpoint struct {
	Height     int64
	BlockHash  []byte
	StateRoot  []byte
	Timestamp  time.Time
	Validators []*EnhancedValidator
}

// DBMetrics contains database metrics
type DBMetrics struct {
	Reads        int64
	Writes       int64
	CacheHits    int64
	CacheMisses  int64
	LevelDBStats string
}

// Helper functions

func makeHeightKey(height int64) []byte {
	key := make([]byte, len(blockHeightKey)+8)
	copy(key, blockHeightKey)
	binary.BigEndian.PutUint64(key[len(blockHeightKey):], uint64(height))
	return key
}

func makeTimeKey(timestamp time.Time, hash []byte) []byte {
	key := make([]byte, len(timeIndexPrefix)+8+len(hash))
	copy(key, timeIndexPrefix)
	binary.BigEndian.PutUint64(key[len(timeIndexPrefix):], uint64(timestamp.Unix()))
	if hash != nil {
		copy(key[len(timeIndexPrefix)+8:], hash)
	}
	return key
}

func makeProposerKey(proposer string, height int64) []byte {
	key := make([]byte, len(proposerIndexPrefix)+len(proposer)+8)
	copy(key, proposerIndexPrefix)
	copy(key[len(proposerIndexPrefix):], []byte(proposer))
	binary.BigEndian.PutUint64(key[len(proposerIndexPrefix)+len(proposer):], uint64(height))
	return key
}

func makeCommitIndexKey(height int64, hash []byte) []byte {
	key := make([]byte, len(commitIndexPrefix)+8+len(hash))
	copy(key, commitIndexPrefix)
	binary.BigEndian.PutUint64(key[len(commitIndexPrefix):], uint64(height))
	copy(key[len(commitIndexPrefix)+8:], hash)
	return key
}

func extractHashFromTimeKey(key []byte) []byte {
	if len(key) <= len(timeIndexPrefix)+8 {
		return nil
	}
	return key[len(timeIndexPrefix)+8:]
}

func encodeUint64(n uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, n)
	return b
}

func decodeUint64(b []byte) uint64 {
	if len(b) < 8 {
		return 0
	}
	return binary.BigEndian.Uint64(b)
}

// LRUCache implements a simple LRU cache
type LRUCache struct {
	mu       sync.RWMutex
	capacity int
	items    map[string]*cacheItem
	head     *cacheItem
	tail     *cacheItem
}

type cacheItem struct {
	key   string
	value interface{}
	prev  *cacheItem
	next  *cacheItem
}

// NewLRUCache creates a new LRU cache
func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		items:    make(map[string]*cacheItem),
	}
}

// Get retrieves an item from cache
func (c *LRUCache) Get(key string) interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()

	item, exists := c.items[key]
	if !exists {
		return nil
	}

	// Move to front
	c.moveToFront(item)

	return item.value
}

// Put adds an item to cache
func (c *LRUCache) Put(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if exists
	if item, exists := c.items[key]; exists {
		item.value = value
		c.moveToFront(item)
		return
	}

	// Create new item
	item := &cacheItem{
		key:   key,
		value: value,
	}

	// Add to front
	if c.head == nil {
		c.head = item
		c.tail = item
	} else {
		item.next = c.head
		c.head.prev = item
		c.head = item
	}

	c.items[key] = item

	// Evict if over capacity
	if len(c.items) > c.capacity {
		c.evictLRU()
	}
}

// Remove removes an item from cache
func (c *LRUCache) Remove(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	item, exists := c.items[key]
	if !exists {
		return
	}

	c.removeItem(item)
}

func (c *LRUCache) moveToFront(item *cacheItem) {
	if item == c.head {
		return
	}

	// Remove from current position
	if item.prev != nil {
		item.prev.next = item.next
	}
	if item.next != nil {
		item.next.prev = item.prev
	}
	if item == c.tail {
		c.tail = item.prev
	}

	// Move to front
	item.prev = nil
	item.next = c.head
	c.head.prev = item
	c.head = item
}

func (c *LRUCache) evictLRU() {
	if c.tail == nil {
		return
	}

	c.removeItem(c.tail)
}

func (c *LRUCache) removeItem(item *cacheItem) {
	delete(c.items, item.key)

	if item.prev != nil {
		item.prev.next = item.next
	} else {
		c.head = item.next
	}

	if item.next != nil {
		item.next.prev = item.prev
	} else {
		c.tail = item.prev
	}
}
