package commands

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Metric unit tests. Covers the three formulas and the aggregator;
// the end-to-end `recall bench` wiring is exercised by the integration
// smoke in the v0.2.6 test plan.

func TestPrecisionAt(t *testing.T) {
	cases := []struct {
		name string
		ret  []string
		rel  []string
		k    int
		want float64
	}{
		{"all hits", []string{"a", "b", "c"}, []string{"a", "b", "c"}, 3, 1.0},
		{"no hits", []string{"a", "b"}, []string{"c", "d"}, 2, 0.0},
		{"half at 2", []string{"a", "c"}, []string{"a", "b"}, 2, 0.5},
		{"k larger than retrieved", []string{"a"}, []string{"a"}, 5, 0.2}, // 1 hit / k=5
		{"k larger than relevant", []string{"a", "b", "c"}, []string{"a"}, 5, 0.2},
		{"k=0 returns 0", []string{"a"}, []string{"a"}, 0, 0.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := precisionAt(tc.ret, tc.rel, tc.k)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestRecallAt(t *testing.T) {
	cases := []struct {
		name string
		ret  []string
		rel  []string
		k    int
		want float64
	}{
		{"all hits", []string{"a", "b"}, []string{"a", "b"}, 2, 1.0},
		{"half at 2", []string{"a", "x"}, []string{"a", "b"}, 2, 0.5},
		{"no hits", []string{"x", "y"}, []string{"a", "b"}, 5, 0.0},
		{"empty relevant returns 0", []string{"a"}, nil, 5, 0.0},
		{"k past retrieved clamps at retrieved length", []string{"a"}, []string{"a", "b"}, 5, 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := recallAt(tc.ret, tc.rel, tc.k)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestReciprocalRank(t *testing.T) {
	cases := []struct {
		name string
		ret  []string
		rel  []string
		want float64
	}{
		{"first hit", []string{"a", "b"}, []string{"a"}, 1.0},
		{"second hit", []string{"x", "a"}, []string{"a"}, 0.5},
		{"third hit", []string{"x", "y", "a"}, []string{"a"}, 1.0 / 3.0},
		{"no hit", []string{"x", "y"}, []string{"a"}, 0.0},
		{"multiple relevant — first wins", []string{"x", "b", "a"}, []string{"a", "b"}, 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := reciprocalRank(tc.ret, tc.rel)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestLoadBenchFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bench.jsonl")
	content := `{"query": "alpha", "relevant": ["a.md"]}
// comment line skipped
# also a comment

{"query": "beta", "relevant": ["b.md", "c.md"]}
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	entries, err := loadBenchFile(path)
	if err != nil {
		t.Fatalf("loadBenchFile: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(entries))
	}
	if entries[0].Query != "alpha" || len(entries[0].Relevant) != 1 {
		t.Errorf("first entry = %+v", entries[0])
	}
	if entries[1].Query != "beta" || len(entries[1].Relevant) != 2 {
		t.Errorf("second entry = %+v", entries[1])
	}
}

func TestLoadBenchFileRejectsMalformed(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name, body, errSub string
	}{
		{"empty query", `{"query": "", "relevant": ["x"]}`, "empty query"},
		{"empty relevant", `{"query": "q", "relevant": []}`, "empty relevant"},
		{"bad json", `{not json}`, "line 1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(dir, tc.name+".jsonl")
			os.WriteFile(p, []byte(tc.body+"\n"), 0o644)
			_, err := loadBenchFile(p)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestAggregateMacroAverage(t *testing.T) {
	results := []benchResult{
		{PrecisionAt: map[string]float64{"@1": 1.0, "@5": 0.4, "@10": 0.2},
			RecallAt: map[string]float64{"@5": 1.0, "@10": 1.0}, MRR: 1.0},
		{PrecisionAt: map[string]float64{"@1": 0.0, "@5": 0.2, "@10": 0.1},
			RecallAt: map[string]float64{"@5": 0.5, "@10": 0.5}, MRR: 0.5},
	}
	sum := aggregate(results, "hybrid", "c", false)
	if sum.NumQueries != 2 {
		t.Errorf("NumQueries = %d", sum.NumQueries)
	}
	if math.Abs(sum.MacroAvg["P@1"]-0.5) > 1e-9 {
		t.Errorf("P@1 avg = %v, want 0.5", sum.MacroAvg["P@1"])
	}
	if math.Abs(sum.MacroAvg["MRR"]-0.75) > 1e-9 {
		t.Errorf("MRR avg = %v, want 0.75", sum.MacroAvg["MRR"])
	}
}

func TestDedupPreservesOrder(t *testing.T) {
	got := dedup([]string{"a", "b", "", "a", "c", "b"})
	want := []string{"a", "b", "c"}
	if len(got) != len(want) {
		t.Fatalf("len %d, want %d — got %v", len(got), len(want), got)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("pos %d = %q, want %q", i, got[i], want[i])
		}
	}
}
