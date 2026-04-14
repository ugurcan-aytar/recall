package embed

import (
	"container/list"
	"sync"
)

// QueryCache is a small LRU cache for query embeddings, keyed by the
// formatted prompt text. Embedding the same query twice in a chat or
// follow-up search is the common case in brain, so reusing the vector
// avoids running the model again.
//
// The cache is intentionally tiny (default 32 entries) — vectors are
// ~1.2 KB each and chat sessions rarely exceed that depth.
type QueryCache struct {
	mu       sync.Mutex
	capacity int
	items    map[string]*list.Element
	order    *list.List // oldest at front, newest at back
}

type cacheEntry struct {
	key string
	vec []float32
}

// DefaultQueryCacheSize is the number of query→vector pairs retained.
const DefaultQueryCacheSize = 32

// NewQueryCache returns an LRU cache with the given capacity. Pass 0 for
// the default of 32.
func NewQueryCache(capacity int) *QueryCache {
	if capacity <= 0 {
		capacity = DefaultQueryCacheSize
	}
	return &QueryCache{
		capacity: capacity,
		items:    make(map[string]*list.Element, capacity),
		order:    list.New(),
	}
}

// Get returns the cached vector for key and a found-flag. On hit, the
// entry is promoted to most-recently-used.
func (c *QueryCache) Get(key string) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.order.MoveToBack(el)
	return el.Value.(*cacheEntry).vec, true
}

// Put records a new (key, vec) pair, evicting the oldest entry when the
// cache is at capacity.
func (c *QueryCache) Put(key string, vec []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		el.Value.(*cacheEntry).vec = vec
		c.order.MoveToBack(el)
		return
	}
	el := c.order.PushBack(&cacheEntry{key: key, vec: vec})
	c.items[key] = el
	if c.order.Len() > c.capacity {
		oldest := c.order.Front()
		c.order.Remove(oldest)
		delete(c.items, oldest.Value.(*cacheEntry).key)
	}
}

// Len returns the current number of cached entries.
func (c *QueryCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}
