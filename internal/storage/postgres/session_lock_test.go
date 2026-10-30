package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"testing"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"

	"github.com/xoai/sage-wiki/internal/store"
)

// TestSessionLockPerProcess: serve opens a writer for its worker, web UI and
// MCP server in one process. They must share the vault's session lock, the
// lock must outlive the backend that took it, and another process must still
// be locked out until the last one closes.
func TestSessionLockPerProcess(t *testing.T) {
	vault, dbName := lockTestVault(t)
	opts := store.OpenOptions{Mode: store.ModeWriter, VectorDimension: 3, LockTimeout: 300 * time.Millisecond}

	first, err := Open(vault, opts)
	if err != nil {
		t.Fatalf("first writer: %v", err)
	}
	second, err := Open(vault, opts)
	if err != nil {
		t.Fatalf("second writer in the same process: %v", err)
	}

	// Another process is a separate session trying the same key.
	raw, err := sql.Open("pgx", vault)
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	other, err := raw.Conn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	otherGets := func() bool {
		t.Helper()
		var got bool
		if err := other.QueryRowContext(context.Background(),
			"SELECT pg_try_advisory_lock($1)", advisoryKey("session", dbName)).Scan(&got); err != nil {
			t.Fatal(err)
		}
		return got
	}

	if otherGets() {
		t.Fatal("another session took the lock while two writers hold it")
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close first: %v", err)
	}
	if otherGets() {
		t.Fatal("closing the writer that took the lock released it while another writer is open")
	}
	if err := second.Close(); err != nil {
		t.Fatalf("close second: %v", err)
	}
	if !otherGets() {
		t.Fatal("lock still held after every writer closed")
	}

	// With the lock held elsewhere, a writer open fails as before.
	if _, err := Open(vault, opts); !errors.Is(err, store.ErrWriterActive) {
		t.Fatalf("writer open against a foreign lock: err = %v, want ErrWriterActive", err)
	}
	if _, err := other.ExecContext(context.Background(), "SELECT pg_advisory_unlock_all()"); err != nil {
		t.Fatal(err)
	}

	// The lock's session dies while a writer is open, and another process
	// takes the lock: a new writer in this process must not join the dead lock.
	third, err := Open(vault, opts)
	if err != nil {
		t.Fatalf("third writer: %v", err)
	}
	defer third.Close()
	// With a timeout, pg_terminate_backend returns only once the session has
	// exited (and released its locks), or false if it is still running.
	var terminated bool
	if err := other.QueryRowContext(context.Background(), `SELECT pg_terminate_backend(pid, 5000) FROM pg_locks
		WHERE locktype = 'advisory' AND granted AND pid <> pg_backend_pid()
		AND ((classid::bigint << 32) | objid::bigint) = $1`, advisoryKey("session", dbName)).Scan(&terminated); err != nil {
		t.Fatalf("terminate the lock's session: %v", err)
	}
	if !terminated {
		t.Fatal("the lock's session did not exit within 5s")
	}
	if !otherGets() {
		t.Fatal("lock still held after its session was terminated")
	}
	if _, err := Open(vault, opts); !errors.Is(err, store.ErrWriterActive) {
		t.Fatalf("writer open after the lock's session died and another took it: err = %v, want ErrWriterActive", err)
	}
}

// lockTestVault clones the test database into a fresh vault and returns its
// DSN and database name.
func lockTestVault(t *testing.T) (string, string) {
	t.Helper()
	dsn := migrationTestDSN(t)
	dbName := fmt.Sprintf("seslock_%d", time.Now().UnixNano())
	boot, err := sql.Open("pgx", swapDB(dsn, "postgres"))
	if err != nil {
		t.Fatalf("bootstrap connect: %v", err)
	}
	createClone(t, boot, dbName, dsnDB(dsn))
	boot.Close()
	t.Cleanup(func() {
		c, err := sql.Open("pgx", swapDB(dsn, "postgres"))
		if err == nil {
			c.Exec("DROP DATABASE " + dbName)
			c.Close()
		}
	})
	return swapDB(dsn, dbName), dbName
}

// TestSessionLockWaitIsPerVault: a writer open waiting for a vault whose
// lock another process holds must not hold up a writer open or close on a
// different vault in the same process.
func TestSessionLockWaitIsPerVault(t *testing.T) {
	busy, busyDB := lockTestVault(t)
	free, _ := lockTestVault(t)

	raw, err := sql.Open("pgx", busy)
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	other, err := raw.Conn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if _, err := other.ExecContext(context.Background(),
		"SELECT pg_advisory_lock($1)", advisoryKey("session", busyDB)); err != nil {
		t.Fatal(err)
	}

	waiting := make(chan error, 1)
	go func() {
		_, err := Open(busy, store.OpenOptions{Mode: store.ModeWriter, VectorDimension: 3, LockTimeout: 3 * time.Second})
		waiting <- err
	}()
	time.Sleep(300 * time.Millisecond) // let it start polling

	start := time.Now()
	b, err := Open(free, store.OpenOptions{Mode: store.ModeWriter, VectorDimension: 3, LockTimeout: 300 * time.Millisecond})
	if err != nil {
		t.Fatalf("writer open on the free vault: %v", err)
	}
	if err := b.Close(); err != nil {
		t.Fatalf("close on the free vault: %v", err)
	}
	if d := time.Since(start); d > 1500*time.Millisecond {
		t.Errorf("open and close on the free vault took %v while another vault's open waited; want under 1.5s", d)
	}
	if err := <-waiting; !errors.Is(err, store.ErrWriterActive) {
		t.Fatalf("writer open on the busy vault: err = %v, want ErrWriterActive", err)
	}
}

// TestLockConnOpen: a server error on the lock's connection keeps the
// session, and the lock, so the process's next writer must join it; a
// terminated session must not be joined.
func TestLockConnOpen(t *testing.T) {
	vault, _ := lockTestVault(t)
	db, err := sql.Open("pgx", vault)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	var n int
	if err := conn.QueryRowContext(ctx, "SELECT 1/0").Scan(&n); err == nil {
		t.Fatal("SELECT 1/0 succeeded")
	}
	if !lockConnOpen(conn) {
		t.Error("connection reported closed after a server error; its session is alive")
	}

	var pid int
	if err := conn.QueryRowContext(ctx, "SELECT pg_backend_pid()").Scan(&pid); err != nil {
		t.Fatal(err)
	}
	var terminated bool
	if err := db.QueryRowContext(ctx, "SELECT pg_terminate_backend($1, 5000)", pid).Scan(&terminated); err != nil || !terminated {
		t.Fatalf("terminate the session: %v (terminated %v)", err, terminated)
	}
	if err := conn.QueryRowContext(ctx, "SELECT 1").Scan(&n); err == nil {
		t.Fatal("SELECT 1 succeeded on a terminated session")
	}
	if lockConnOpen(conn) {
		t.Error("connection reported open after its session was terminated")
	}
}
