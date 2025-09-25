// Package optimization provides performance optimization tools
package optimization

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// Cache provides a high-performance in-memory cache
type Cache struct {
	mu       sync.RWMutex
	items    map[string]*CacheItem
	maxSize  int64
	size     int64
	ttl      time.Duration
	stats    *CacheStats
	shards   []*CacheShard
	numShards int
}

// CacheItem represents a cached item
type CacheItem struct {
	Key       string
	Value     interface{}
	Size      int64
	ExpiresAt time.Time
	AccessCount uint64
	LastAccess time.Time
}

// CacheShard for sharded caching
type CacheShard struct {
	mu    sync.RWMutex
	items map[string]*CacheItem
}

// CacheStats tracks cache performance
type CacheStats struct {
	Hits       uint64
	Misses     uint64
	Evictions  uint64
	Sets       uint64
	Gets       uint64
	HitRate    float64
}

// ObjectPool provides object pooling for memory efficiency
type ObjectPool struct {
	pool sync.Pool
	new  func() interface{}
	reset func(interface{})
	stats *PoolStats
}

// PoolStats tracks pool performance
type PoolStats struct {
	Gets    uint64
	Puts    uint64
	News    uint64
	Reuses  uint64
}

// BatchProcessor processes items in batches for efficiency
type BatchProcessor struct {
	mu          sync.Mutex
	items       []interface{}
	batchSize   int
	flushInterval time.Duration
	processor   func([]interface{}) error
	ctx         context.Context
	cancel      context.CancelFunc
}

// ParallelExecutor runs tasks in parallel with controlled concurrency
type ParallelExecutor struct {
	workers   int
	taskQueue chan Task
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
}

// Task represents a parallel task
type Task struct {
	ID       string
	Execute  func() error
	Callback func(error)
}

// AdvancedMemoryOptimizer provides memory optimization utilities
type AdvancedMemoryOptimizer struct {
	gcInterval   time.Duration
	memThreshold uint64
	running      atomic.Bool
}

// NewCache creates a high-performance cache
func NewCache(maxSize int64, ttl time.Duration, numShards int) *Cache {
	if numShards == 0 {
		numShards = runtime.NumCPU() * 2
	}

	shards := make([]*CacheShard, numShards)
	for i := 0; i < numShards; i++ {
		shards[i] = &CacheShard{
			items: make(map[string]*CacheItem),
		}
	}

	cache := &Cache{
		items:     make(map[string]*CacheItem),
		maxSize:   maxSize,
		ttl:       ttl,
		stats:     &CacheStats{},
		shards:    shards,
		numShards: numShards,
	}

	// Start cleanup routine
	go cache.cleanupRoutine()

	return cache
}

// Set adds an item to the cache
func (c *Cache) Set(key string, value interface{}, size int64) {
	shard := c.getShard(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	// Check if we need to evict items
	if atomic.LoadInt64(&c.size)+size > c.maxSize {
		c.evictLRU(size)
	}

	item := &CacheItem{
		Key:        key,
		Value:      value,
		Size:       size,
		ExpiresAt:  time.Now().Add(c.ttl),
		AccessCount: 0,
		LastAccess: time.Now(),
	}

	// Update size
	if oldItem, exists := shard.items[key]; exists {
		atomic.AddInt64(&c.size, size-oldItem.Size)
	} else {
		atomic.AddInt64(&c.size, size)
	}

	shard.items[key] = item
	atomic.AddUint64(&c.stats.Sets, 1)
}

// Get retrieves an item from the cache
func (c *Cache) Get(key string) (interface{}, bool) {
	shard := c.getShard(key)
	shard.mu.RLock()
	item, exists := shard.items[key]
	shard.mu.RUnlock()

	atomic.AddUint64(&c.stats.Gets, 1)

	if !exists {
		atomic.AddUint64(&c.stats.Misses, 1)
		return nil, false
	}

	// Check expiration
	if time.Now().After(item.ExpiresAt) {
		c.Delete(key)
		atomic.AddUint64(&c.stats.Misses, 1)
		return nil, false
	}

	// Update access stats
	atomic.AddUint64(&item.AccessCount, 1)
	item.LastAccess = time.Now()
	atomic.AddUint64(&c.stats.Hits, 1)

	// Update hit rate
	hits := atomic.LoadUint64(&c.stats.Hits)
	gets := atomic.LoadUint64(&c.stats.Gets)
	if gets > 0 {
		c.stats.HitRate = float64(hits) / float64(gets)
	}

	return item.Value, true
}

// Delete removes an item from the cache
func (c *Cache) Delete(key string) {
	shard := c.getShard(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	if item, exists := shard.items[key]; exists {
		atomic.AddInt64(&c.size, -item.Size)
		delete(shard.items, key)
	}
}

// getShard returns the shard for a key
func (c *Cache) getShard(key string) *CacheShard {
	hash := fnv32(key)
	return c.shards[hash%uint32(c.numShards)]
}

// evictLRU evicts least recently used items
func (c *Cache) evictLRU(needed int64) {
	// Simple LRU eviction - in production use more sophisticated algorithm
	for _, shard := range c.shards {
		shard.mu.Lock()

		var oldest *CacheItem
		var oldestKey string

		for key, item := range shard.items {
			if oldest == nil || item.LastAccess.Before(oldest.LastAccess) {
				oldest = item
				oldestKey = key
			}
		}

		if oldest != nil {
			atomic.AddInt64(&c.size, -oldest.Size)
			delete(shard.items, oldestKey)
			atomic.AddUint64(&c.stats.Evictions, 1)
		}

		shard.mu.Unlock()

		if atomic.LoadInt64(&c.size)+needed <= c.maxSize {
			break
		}
	}
}

// cleanupRoutine removes expired items
func (c *Cache) cleanupRoutine() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		now := time.Now()
		for _, shard := range c.shards {
			shard.mu.Lock()
			for key, item := range shard.items {
				if now.After(item.ExpiresAt) {
					atomic.AddInt64(&c.size, -item.Size)
					delete(shard.items, key)
				}
			}
			shard.mu.Unlock()
		}
	}
}

// GetStats returns cache statistics
func (c *Cache) GetStats() CacheStats {
	return *c.stats
}

// NewObjectPool creates an object pool
func NewObjectPool(new func() interface{}, reset func(interface{})) *ObjectPool {
	return &ObjectPool{
		pool: sync.Pool{
			New: func() interface{} {
				atomic.AddUint64(&pool.stats.News, 1)
				return new()
			},
		},
		new:   new,
		reset: reset,
		stats: &PoolStats{},
	}
}

var pool *ObjectPool // Package-level variable for the closure

// Get retrieves an object from the pool
func (p *ObjectPool) Get() interface{} {
	atomic.AddUint64(&p.stats.Gets, 1)
	obj := p.pool.Get()
	if obj != nil {
		atomic.AddUint64(&p.stats.Reuses, 1)
	}
	return obj
}

// Put returns an object to the pool
func (p *ObjectPool) Put(obj interface{}) {
	if p.reset != nil {
		p.reset(obj)
	}
	atomic.AddUint64(&p.stats.Puts, 1)
	p.pool.Put(obj)
}

// GetStats returns pool statistics
func (p *ObjectPool) GetStats() PoolStats {
	return *p.stats
}

// NewBatchProcessor creates a batch processor
func NewBatchProcessor(batchSize int, flushInterval time.Duration, processor func([]interface{}) error) *BatchProcessor {
	ctx, cancel := context.WithCancel(context.Background())

	bp := &BatchProcessor{
		items:         make([]interface{}, 0, batchSize),
		batchSize:     batchSize,
		flushInterval: flushInterval,
		processor:     processor,
		ctx:           ctx,
		cancel:        cancel,
	}

	go bp.flushRoutine()
	return bp
}

// Add adds an item to the batch
func (bp *BatchProcessor) Add(item interface{}) error {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	bp.items = append(bp.items, item)

	if len(bp.items) >= bp.batchSize {
		return bp.flush()
	}

	return nil
}

// flush processes the current batch
func (bp *BatchProcessor) flush() error {
	if len(bp.items) == 0 {
		return nil
	}

	batch := make([]interface{}, len(bp.items))
	copy(batch, bp.items)
	bp.items = bp.items[:0]

	return bp.processor(batch)
}

// flushRoutine periodically flushes the batch
func (bp *BatchProcessor) flushRoutine() {
	ticker := time.NewTicker(bp.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			bp.mu.Lock()
			bp.flush()
			bp.mu.Unlock()
		case <-bp.ctx.Done():
			return
		}
	}
}

// Stop stops the batch processor
func (bp *BatchProcessor) Stop() {
	bp.cancel()
	bp.mu.Lock()
	bp.flush()
	bp.mu.Unlock()
}

// NewParallelExecutor creates a parallel task executor
func NewParallelExecutor(workers int) *ParallelExecutor {
	ctx, cancel := context.WithCancel(context.Background())

	pe := &ParallelExecutor{
		workers:   workers,
		taskQueue: make(chan Task, workers*2),
		ctx:       ctx,
		cancel:    cancel,
	}

	// Start workers
	for i := 0; i < workers; i++ {
		pe.wg.Add(1)
		go pe.worker()
	}

	return pe
}

// Execute submits a task for execution
func (pe *ParallelExecutor) Execute(task Task) {
	select {
	case pe.taskQueue <- task:
	case <-pe.ctx.Done():
	}
}

// worker processes tasks
func (pe *ParallelExecutor) worker() {
	defer pe.wg.Done()

	for {
		select {
		case task := <-pe.taskQueue:
			err := task.Execute()
			if task.Callback != nil {
				task.Callback(err)
			}
		case <-pe.ctx.Done():
			return
		}
	}
}

// Stop stops the executor
func (pe *ParallelExecutor) Stop() {
	pe.cancel()
	close(pe.taskQueue)
	pe.wg.Wait()
}

// NewAdvancedMemoryOptimizer creates a memory optimizer
func NewAdvancedMemoryOptimizer(gcInterval time.Duration, memThreshold uint64) *AdvancedMemoryOptimizer {
	mo := &AdvancedMemoryOptimizer{
		gcInterval:   gcInterval,
		memThreshold: memThreshold,
	}
	mo.running.Store(true)
	go mo.optimize()
	return mo
}

// optimize runs memory optimization
func (mo *AdvancedMemoryOptimizer) optimize() {
	ticker := time.NewTicker(mo.gcInterval)
	defer ticker.Stop()

	for mo.running.Load() {
		select {
		case <-ticker.C:
			var m runtime.MemStats
			runtime.ReadMemStats(&m)

			// Force GC if memory usage is high
			if m.Alloc > mo.memThreshold {
				runtime.GC()
				runtime.GC() // Double GC for thorough cleanup
			}

			// Return memory to OS
			if m.HeapIdle > mo.memThreshold/2 {
				runtime.GC()
				runtime.Gosched()
			}
		}
	}
}

// Stop stops the memory optimizer
func (mo *AdvancedMemoryOptimizer) Stop() {
	mo.running.Store(false)
}

// Helper functions

// fnv32 is a fast hash function
func fnv32(key string) uint32 {
	hash := uint32(2166136261)
	for i := 0; i < len(key); i++ {
		hash *= 16777619
		hash ^= uint32(key[i])
	}
	return hash
}

// ZeroCopy converts string to bytes without allocation
func StringToBytes(s string) []byte {
	return unsafe.Slice(unsafe.StringData(s), len(s))
}

// BytesToString converts bytes to string without allocation
func BytesToString(b []byte) string {
	return unsafe.String(&b[0], len(b))
}