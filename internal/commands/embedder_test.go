package commands

import (
	"testing"

	"github.com/ugurcan-aytar/recall/internal/embed"
)

func TestSetEmbedderOverrideRoundTrip(t *testing.T) {
	t.Cleanup(func() { SetEmbedderOverride(nil) })

	mock := embed.NewMockEmbedder(0)
	SetEmbedderOverride(mock)

	got, err := openEmbedder()
	if err != nil {
		t.Fatalf("openEmbedder: %v", err)
	}
	if got != mock {
		t.Error("openEmbedder returned a different embedder than the override")
	}
}

func TestEmbedQueryCachedHits(t *testing.T) {
	t.Cleanup(func() {
		SetEmbedderOverride(nil)
		resetQueryCacheForTest()
	})
	mock := embed.NewMockEmbedder(0)
	SetEmbedderOverride(mock)
	resetQueryCacheForTest()

	emb, _ := openEmbedder()

	v1, err := embedQueryCached(emb, "hello")
	if err != nil {
		t.Fatal(err)
	}
	if queryEmbedCache.Len() != 1 {
		t.Errorf("cache len after 1st call = %d, want 1", queryEmbedCache.Len())
	}
	v2, err := embedQueryCached(emb, "hello")
	if err != nil {
		t.Fatal(err)
	}
	for i := range v1 {
		if v1[i] != v2[i] {
			t.Fatalf("cached vector differs at %d: %v vs %v", i, v1[i], v2[i])
		}
	}
	if queryEmbedCache.Len() != 1 {
		t.Errorf("cache len after 2nd call = %d, want 1 (no new entry)", queryEmbedCache.Len())
	}
}
