package compiler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xoai/sage-wiki/internal/config"
	"github.com/xoai/sage-wiki/internal/ontology"
	"github.com/xoai/sage-wiki/internal/storage"
	"github.com/xoai/sage-wiki/internal/store"
	"github.com/xoai/sage-wiki/internal/trust"
)

// P3-6 T6b: write-time supersession trigger and trust conflict emission.

func temporalTestStores(t *testing.T) (*ontology.Store, *trust.Store) {
	t.Helper()
	db, err := storage.Open(filepath.Join(t.TempDir(), "temporal.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	ont := ontology.NewStore(db,
		ontology.ValidRelationNames(ontology.MergedRelations(
			[]config.RelationConfig{{Name: "works_at", Functional: true}})),
		ontology.ValidEntityTypeNames(ontology.MergedEntityTypes(nil)))
	return ont, trust.NewStore(db)
}

func temporalCfg() *config.Config {
	c := enabledCfg()
	c.Ontology.Relations = []config.RelationConfig{{Name: "works_at", Functional: true}}
	return c
}

func graphServer(t *testing.T, graph string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": graph}}},
			"model":   "m",
		})
	}))
	t.Cleanup(srv.Close)
	return srv
}

const worksAtAcme = `{"entities":[
	{"name":"alice","type":"person","description":"A person."},
	{"name":"acme","type":"org","description":"A company."}
],"relations":[
	{"source":"alice","predicate":"works_at","target":"acme",
	 "evidence":"Alice works at Acme.","confidence":0.9}
]}`

const worksAtInitech = `{"entities":[
	{"name":"alice","type":"person","description":"A person."},
	{"name":"initech","type":"org","description":"Another company."}
],"relations":[
	{"source":"alice","predicate":"works_at","target":"initech",
	 "evidence":"Alice works at Initech.","confidence":0.9}
]}`

const worksAtLowConf = `{"entities":[
	{"name":"alice","type":"person","description":"A person."},
	{"name":"globex","type":"org","description":"Yet another."}
],"relations":[
	{"source":"alice","predicate":"works_at","target":"globex",
	 "evidence":"Alice might work at Globex.","confidence":0.5}
]}`

const contradictsGraph = `{"entities":[
	{"name":"acme","type":"org","description":"A company."},
	{"name":"initech","type":"org","description":"Another company."}
],"relations":[
	{"source":"acme","predicate":"contradicts","target":"initech",
	 "evidence":"These statements conflict.","confidence":0.95}
]}`

func runOneDoc(t *testing.T, ont *ontology.Store, ts store.TrustStore, cfg *config.Config, graph, doc string) []FunctionalSupersession {
	t.Helper()
	srv := graphServer(t, graph)
	// The summary grounds every evidence sentence used by this file's
	// fixture graphs (SPEC-08 span verification rejects ungrounded edges).
	const groundedSummary = "Alice works at Acme. Alice works at Initech. " +
		"Alice might work at Globex. These statements conflict."
	_, sup, _ := ExtractTriplesPass(context.Background(), ont,
		[]SummaryResult{{SourcePath: doc, Summary: groundedSummary}}, nil,
		cfg, triplesClient(t, srv.URL), false, t.TempDir(), nil, ts, nil, nil, nil)
	return sup
}

func liveTargets(t *testing.T, ont *ontology.Store, entity string) []string {
	t.Helper()
	rels, err := ont.GetRelations(entity, ontology.Outbound, "")
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, r := range rels {
		out = append(out, r.TargetID)
	}
	return out
}

func conflicts(t *testing.T, ts *trust.Store) []*store.PendingOutput {
	t.Helper()
	rows, err := ts.ListByState(store.StateConflict)
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

func TestFunctionalSupersessionAutoApplies(t *testing.T) {
	ont, ts := temporalTestStores(t)
	cfg := temporalCfg()

	runOneDoc(t, ont, ts, cfg, worksAtAcme, "raw/a.md")
	sup := runOneDoc(t, ont, ts, cfg, worksAtInitech, "raw/b.md")

	targets := liveTargets(t, ont, "alice")
	if len(targets) != 1 || targets[0] != "initech" {
		t.Errorf("default read must return only the winner (initech), got %v", targets)
	}
	if len(sup) != 1 || sup[0].KeepTargetID != "initech" || sup[0].Predicate != "works_at" {
		t.Errorf("supersession not collected for sweep: %+v", sup)
	}
	// The loser is invalidated, not deleted: the unfiltered admin read still
	// returns it, stamped with valid_to + invalidated_by.
	all, err := ont.AllRelations()
	if err != nil {
		t.Fatal(err)
	}
	var loser *store.Relation
	for i := range all {
		if all[i].TargetID == "acme" {
			loser = &all[i]
		}
	}
	if loser == nil || loser.ValidTo == "" || loser.InvalidatedBy == "" {
		t.Errorf("loser must be invalidated (not deleted): %+v", loser)
	}
	if n := len(conflicts(t, ts)); n != 0 {
		t.Errorf("auto-applied supersession must not raise a conflict, got %d", n)
	}
}

func TestFunctionalBelowThresholdRaisesConflict(t *testing.T) {
	ont, ts := temporalTestStores(t)
	cfg := temporalCfg()

	runOneDoc(t, ont, ts, cfg, worksAtAcme, "raw/a.md")
	runOneDoc(t, ont, ts, cfg, worksAtLowConf, "raw/b.md")

	targets := liveTargets(t, ont, "alice")
	if len(targets) != 2 {
		t.Errorf("below-threshold must leave both edges live, got %v", targets)
	}
	rows := conflicts(t, ts)
	if len(rows) != 1 {
		t.Fatalf("expected 1 trust conflict, got %d", len(rows))
	}
	if rows[0].CreatedAt.IsZero() {
		t.Error("conflict row must carry CreatedAt")
	}
	if rows[0].QuestionHash == "" || rows[0].AnswerHash == "" {
		t.Error("conflict row must carry hashes")
	}

	// Dedup: same conflict again → still one row.
	runOneDoc(t, ont, ts, cfg, worksAtLowConf, "raw/b.md")
	if n := len(conflicts(t, ts)); n != 1 {
		t.Errorf("re-run must dedup conflicts, got %d", n)
	}
}

func TestBareContradictsRaisesConflictOnly(t *testing.T) {
	ont, ts := temporalTestStores(t)
	cfg := temporalCfg()

	runOneDoc(t, ont, ts, cfg, contradictsGraph, "raw/a.md")

	if n := len(conflicts(t, ts)); n != 1 {
		t.Errorf("bare contradicts must raise exactly one conflict, got %d", n)
	}
	// And nothing is invalidated: the contradicts edge itself is live.
	if targets := liveTargets(t, ont, "acme"); len(targets) != 1 {
		t.Errorf("contradicts edge must stay live, got %v", targets)
	}
}

func TestSupersessionWithNilTrustStore(t *testing.T) {
	ont, _ := temporalTestStores(t)
	cfg := temporalCfg()

	runOneDoc(t, ont, nil, cfg, worksAtAcme, "raw/a.md")
	runOneDoc(t, ont, nil, cfg, worksAtInitech, "raw/b.md") // must not panic

	if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "initech" {
		t.Errorf("nil trust store: supersession must still auto-apply, got %v", targets)
	}
}

func TestTemporalDisabledNoTrigger(t *testing.T) {
	ont, ts := temporalTestStores(t)
	cfg := temporalCfg()
	off := false
	cfg.Ontology.Temporal.Enabled = &off

	runOneDoc(t, ont, ts, cfg, worksAtAcme, "raw/a.md")
	runOneDoc(t, ont, ts, cfg, worksAtInitech, "raw/b.md")

	if targets := liveTargets(t, ont, "alice"); len(targets) != 2 {
		t.Errorf("disabled: both edges must stay, got %v", targets)
	}
	if n := len(conflicts(t, ts)); n != 0 {
		t.Errorf("disabled: no conflicts, got %d", n)
	}
}

// T7: the post-resolution sweep closes the pre-resolution hazard — an alias
// form applied AFTER the write-time trigger must not resurrect the loser.
func TestPostResolutionSweepCoversNewAliasForms(t *testing.T) {
	ont, ts := temporalTestStores(t)
	cfg := temporalCfg()

	// Doc A asserts alice works_at acme; doc B asserts alice_alias works_at
	// initech. The write-time trigger for B cannot see "alice" — no alias
	// exists yet — so acme stays live at this point.
	runOneDoc(t, ont, ts, cfg, worksAtAcme, "raw/a.md")
	sup := runOneDoc(t, ont, ts, cfg,
		`{"entities":[
			{"name":"alice_alias","type":"person","description":"A person."},
			{"name":"initech","type":"org","description":"Another company."}
		],"relations":[
			{"source":"alice_alias","predicate":"works_at","target":"initech",
			 "evidence":"Alice works at Initech.","confidence":0.9}
		]}`, "raw/b.md")

	if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "acme" {
		t.Fatalf("pre-link: acme must still be live under alice, got %v", targets)
	}

	// Resolution applies the alias and derives the winner onto the canonical.
	if _, err := ont.LinkAlias(store.EntityAlias{
		Alias: "alice_alias", CanonicalID: "alice", EntityType: "person",
		Status: store.AliasApplied, Source: "llm", CreatedAt: "2026-07-29T00:00:00Z",
	}); err != nil {
		t.Fatal(err)
	}
	// Without the sweep, the canonical now answers BOTH acme (alias original
	// untouched) and initech (derived copy).
	runSupersessionSweep(ont, sup)

	if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "initech" {
		t.Errorf("post-sweep: only initech must answer under the canonical, got %v", targets)
	}
	if targets := liveTargets(t, ont, "alice_alias"); len(targets) != 1 || targets[0] != "initech" {
		t.Errorf("post-sweep: alias form must answer only the winner, got %v", targets)
	}
}

// runDatedDoc runs the triples pass over doc, a file in projectDir dated by
// its frontmatter, so the edges it writes start on that date, and returns the
// supersessions it recorded for the post-resolution sweep.
func runDatedDoc(t *testing.T, ont store.OntologyStore, ts store.TrustStore, cfg *config.Config, graph, projectDir, doc, date string) []FunctionalSupersession {
	t.Helper()
	path := filepath.Join(projectDir, doc)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("---\ndate: "+date+"\n---\n\nAlice.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	srv := graphServer(t, graph)
	_, sup, _ := ExtractTriplesPass(context.Background(), ont,
		[]SummaryResult{{SourcePath: doc, Summary: "Alice works at Acme. Alice works at Initech. Alice works at Globex."}}, nil,
		cfg, triplesClient(t, srv.URL), false, projectDir, nil, ts, nil, nil, nil)
	return sup
}

// worksAt is a one-edge graph: source works_at target at confidence 0.9.
func worksAt(source, target, evidence string) string {
	return `{"entities":[
		{"name":"` + source + `","type":"person","description":"A person."},
		{"name":"` + target + `","type":"org","description":"A company."}
	],"relations":[
		{"source":"` + source + `","predicate":"works_at","target":"` + target + `",
		 "evidence":"` + evidence + `","confidence":0.9}
	]}`
}

// failingReads is an ontology store whose edge reads fail.
type failingReads struct{ store.OntologyStore }

func (failingReads) GetRelationsAt(string, store.Direction, string, time.Time) ([]store.Relation, error) {
	return nil, errors.New("read failed")
}

// A functional edge from an older source compiled after a newer one must not
// replace the newer edge: the older edge is history, closed where the newer
// starts. With equal dates the later compile still wins, as before.
func TestFunctionalSupersessionFollowsValidFrom(t *testing.T) {
	edge := func(t *testing.T, ont *ontology.Store, target string) store.Relation {
		t.Helper()
		all, err := ont.AllRelations()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range all {
			if r.SourceID == "alice" && r.TargetID == target {
				return r
			}
		}
		t.Fatalf("no alice -> %s edge", target)
		return store.Relation{}
	}
	link := func(t *testing.T, ont *ontology.Store, alias string) {
		t.Helper()
		if _, err := ont.LinkAlias(store.EntityAlias{
			Alias: alias, CanonicalID: "alice", EntityType: "person",
			Status: store.AliasApplied, Source: "llm", CreatedAt: "2026-10-28T00:00:00Z",
		}); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("older source compiled second", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtInitech, dir, "raw/new.md", "2026-10-27")
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtAcme, dir, "raw/old.md", "2026-08-04")

		if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "initech" {
			t.Fatalf("live targets = %v, want [initech] (the newer source's edge)", targets)
		}
		initech, acme := edge(t, ont, "initech"), edge(t, ont, "acme")
		if initech.ValidTo != "" {
			t.Errorf("newer edge was closed: valid_to %q", initech.ValidTo)
		}
		if acme.ValidFrom != "2026-08-04T00:00:00Z" || acme.ValidTo != initech.ValidFrom || acme.InvalidatedBy != initech.ID {
			t.Errorf("older edge = from %q to %q by %q, want from 2026-08-04 to %q by %q",
				acme.ValidFrom, acme.ValidTo, acme.InvalidatedBy, initech.ValidFrom, initech.ID)
		}
		if n := len(conflicts(t, ts)); n != 0 {
			t.Errorf("date-ordered supersession must not raise a conflict, got %d", n)
		}
	})

	t.Run("newer source dated in the future", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtInitech, dir, "raw/next.md", "2999-01-01")
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtAcme, dir, "raw/old.md", "2026-08-04")

		initech, acme := edge(t, ont, "initech"), edge(t, ont, "acme")
		if initech.ValidTo != "" {
			t.Errorf("future-dated newer edge was closed: valid_to %q", initech.ValidTo)
		}
		if acme.ValidTo != initech.ValidFrom || acme.InvalidatedBy != initech.ID {
			t.Errorf("older edge = to %q by %q, want to %q by %q", acme.ValidTo, acme.InvalidatedBy, initech.ValidFrom, initech.ID)
		}
	})

	t.Run("older source names an alias of the newer one's entity", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtInitech, dir, "raw/new.md", "2026-10-27")
		if err := ont.AddEntity(store.Entity{ID: "alice_alias", Type: "concept", Name: "Alice"}); err != nil {
			t.Fatal(err)
		}
		link(t, ont, "alice_alias")
		runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice_alias", "acme", "Alice works at Acme."), dir, "raw/old.md", "2026-08-04")

		if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "initech" {
			t.Errorf("live targets = %v, want [initech] (the newer source's edge)", targets)
		}
		if initech := edge(t, ont, "initech"); initech.ValidTo != "" {
			t.Errorf("newer edge was closed: valid_to %q", initech.ValidTo)
		}
	})

	// Resolution applies the alias only after the older edge was written, so
	// the post-resolution sweep has to find the later winner.
	t.Run("alias applied by resolution after the older edge", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtInitech, dir, "raw/new.md", "2026-10-27")
		sup := runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice_alias", "acme", "Alice works at Acme."), dir, "raw/old.md", "2026-08-04")
		link(t, ont, "alice_alias")
		runSupersessionSweep(ont, sup)

		if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "initech" {
			t.Errorf("live targets = %v, want [initech] (the newer source's edge)", targets)
		}
		initech, acme := edge(t, ont, "initech"), edge(t, ont, "acme")
		if initech.ValidTo != "" {
			t.Errorf("newer edge was closed: valid_to %q", initech.ValidTo)
		}
		if acme.ValidTo != initech.ValidFrom {
			t.Errorf("older alias edge valid_to = %q, want %q", acme.ValidTo, initech.ValidFrom)
		}
	})

	// Three open edges under three forms, merged by resolution: the sweep for
	// the oldest must keep the newest current, not the middle one.
	t.Run("three forms merged by resolution", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		var sup []FunctionalSupersession
		sup = append(sup, runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice_c", "acme", "Alice works at Acme."), dir, "raw/2022.md", "2022-01-01")...)
		sup = append(sup, runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice_b", "globex", "Alice works at Globex."), dir, "raw/2023.md", "2023-01-01")...)
		sup = append(sup, runDatedDoc(t, ont, ts, temporalCfg(), worksAtInitech, dir, "raw/2024.md", "2024-01-01")...)
		link(t, ont, "alice_b")
		link(t, ont, "alice_c")
		runSupersessionSweep(ont, sup)

		if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "initech" {
			t.Errorf("live targets = %v, want [initech] (the newest edge)", targets)
		}
		if initech := edge(t, ont, "initech"); initech.ValidTo != "" {
			t.Errorf("newest edge was closed: valid_to %q", initech.ValidTo)
		}
	})

	// The target comes back after another: under three forms, acme (2022),
	// initech (2023), then acme again (2024), merged by resolution. Swept
	// oldest first, the 2022 edge fires initech's supersession, which closes
	// the 2024 acme edge at its own start and leaves nothing current.
	t.Run("target returns after another, merged by resolution", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		var sup []FunctionalSupersession
		sup = append(sup, runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice_c", "acme", "Alice works at Acme."), dir, "raw/2022.md", "2022-01-01")...)
		sup = append(sup, runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice_b", "initech", "Alice works at Initech."), dir, "raw/2023.md", "2023-01-01")...)
		sup = append(sup, runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice", "acme", "Alice works at Acme."), dir, "raw/2024.md", "2024-01-01")...)
		link(t, ont, "alice_b")
		link(t, ont, "alice_c")
		runSupersessionSweep(ont, sup)

		targets := liveTargets(t, ont, "alice")
		if len(targets) == 0 {
			t.Fatal("no live target, want acme (the newest edge)")
		}
		for _, target := range targets {
			if target != "acme" {
				t.Errorf("live targets = %v, want only acme (the newest edge)", targets)
				break
			}
		}
	})

	t.Run("lookup failure skips the sweep", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtInitech, dir, "raw/new.md", "2026-10-27")
		sup := runDatedDoc(t, ont, ts, temporalCfg(), worksAt("alice_alias", "acme", "Alice works at Acme."), dir, "raw/old.md", "2026-08-04")
		link(t, ont, "alice_alias")
		runSupersessionSweep(failingReads{ont}, sup)

		if initech := edge(t, ont, "initech"); initech.ValidTo != "" {
			t.Errorf("newer edge was closed after a failed lookup: valid_to %q", initech.ValidTo)
		}
	})

	t.Run("equal dates", func(t *testing.T) {
		ont, ts := temporalTestStores(t)
		dir := t.TempDir()
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtInitech, dir, "raw/a.md", "2026-10-27")
		runDatedDoc(t, ont, ts, temporalCfg(), worksAtAcme, dir, "raw/b.md", "2026-10-27")

		if targets := liveTargets(t, ont, "alice"); len(targets) != 1 || targets[0] != "acme" {
			t.Errorf("live targets = %v, want [acme] (equal dates: the later compile wins)", targets)
		}
	})
}
