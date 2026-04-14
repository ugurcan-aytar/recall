package embed

import (
	"sync"
	"testing"
)

func TestQueryCachePutGet(t *testing.T) {
	c := NewQueryCache(4)
	v := []float32{1, 2, 3}
	c.Put("a", v)
	got, ok := c.Get("a")
	if !ok {
		t.Fatal("Get(a) miss")
	}
	if len(got) != 3 || got[0] != 1 {
		t.Errorf("wrong vector returned: %v", got)
	}
}

func TestQueryCacheLRUEviction(t *testing.T) {
	c := NewQueryCache(3)
	c.Put("a", []float32{1})
	c.Put("b", []float32{2})
	c.Put("c", []float32{3})
	c.Put("d", []float32{4}) // evicts "a"

	if _, ok := c.Get("a"); ok {
		t.Error("oldest entry was not evicted")
	}
	for _, k := range []string{"b", "c", "d"} {
		if _, ok := c.Get(k); !ok {
			t.Errorf("%q evicted unexpectedly", k)
		}
	}
}

func TestQueryCacheLRUPromotion(t *testing.T) {
	c := NewQueryCache(3)
	c.Put("a", []float32{1})
	c.Put("b", []float32{2})
	c.Put("c", []float32{3})
	// touch "a" so it becomes most-recently used
	if _, ok := c.Get("a"); !ok {
		t.Fatal("Get(a) miss")
	}
	c.Put("d", []float32{4}) // should now evict "b", not "a"

	if _, ok := c.Get("a"); !ok {
		t.Error("a was evicted despite being most recently used")
	}
	if _, ok := c.Get("b"); ok {
		t.Error("b should have been evicted")
	}
}

func TestQueryCacheUpdateExisting(t *testing.T) {
	c := NewQueryCache(2)
	c.Put("a", []float32{1})
	c.Put("a", []float32{9})
	v, _ := c.Get("a")
	if v[0] != 9 {
		t.Errorf("update did not replace value: %v", v)
	}
	if c.Len() != 1 {
		t.Errorf("Len = %d, want 1", c.Len())
	}
}

func TestQueryCacheConcurrent(t *testing.T) {
	c := NewQueryCache(64)
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := string(rune('a' + (i % 26)))
			c.Put(key, []float32{float32(i)})
			c.Get(key)
		}(i)
	}
	wg.Wait()
}

func TestQueryCacheDefaultSize(t *testing.T) {
	c := NewQueryCache(0)
	if c.capacity != DefaultQueryCacheSize {
		t.Errorf("default capacity = %d, want %d", c.capacity, DefaultQueryCacheSize)
	}
}
