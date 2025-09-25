package storage

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/syndtr/goleveldb/leveldb"
)

// Migration represents a database migration
type Migration struct {
	ID          string    `json:"id"`
	Version     int       `json:"version"`
	Description string    `json:"description"`
	AppliedAt   time.Time `json:"applied_at"`
	Checksum    string    `json:"checksum"`
}

// MigrationFunc is a function that performs a migration
type MigrationFunc func(db *leveldb.DB) error

// MigrationEntry defines a migration to be executed
type MigrationEntry struct {
	Version     int
	Description string
	Up          MigrationFunc
	Down        MigrationFunc
}

// MigrationManager handles database migrations
type MigrationManager struct {
	db         *leveldb.DB
	migrations []MigrationEntry
}

// NewMigrationManager creates a new migration manager
func NewMigrationManager(db *leveldb.DB) *MigrationManager {
	return &MigrationManager{
		db:         db,
		migrations: getAllMigrations(),
	}
}

// getAllMigrations returns all defined migrations
func getAllMigrations() []MigrationEntry {
	return []MigrationEntry{
		{
			Version:     1,
			Description: "Initialize blockchain schema",
			Up:          migration001InitSchema,
			Down:        migration001RollbackSchema,
		},
		{
			Version:     2,
			Description: "Add indexes for fast lookups",
			Up:          migration002AddIndexes,
			Down:        migration002RemoveIndexes,
		},
		{
			Version:     3,
			Description: "Add validator state tracking",
			Up:          migration003ValidatorState,
			Down:        migration003RollbackValidatorState,
		},
		{
			Version:     4,
			Description: "Add transaction receipts",
			Up:          migration004TransactionReceipts,
			Down:        migration004RollbackTransactionReceipts,
		},
		{
			Version:     5,
			Description: "Add merkle tree roots",
			Up:          migration005MerkleRoots,
			Down:        migration005RollbackMerkleRoots,
		},
	}
}

// Migrate runs all pending migrations
func (m *MigrationManager) Migrate() error {
	currentVersion, err := m.getCurrentVersion()
	if err != nil {
		return fmt.Errorf("failed to get current version: %w", err)
	}

	log.Printf("Current database version: %d", currentVersion)

	for _, migration := range m.migrations {
		if migration.Version <= currentVersion {
			continue
		}

		log.Printf("Applying migration %d: %s", migration.Version, migration.Description)

		// Start transaction
		batch := new(leveldb.Batch)

		// Run migration
		if err := migration.Up(m.db); err != nil {
			return fmt.Errorf("migration %d failed: %w", migration.Version, err)
		}

		// Record migration
		migrationRecord := Migration{
			ID:          fmt.Sprintf("migration_%03d", migration.Version),
			Version:     migration.Version,
			Description: migration.Description,
			AppliedAt:   time.Now(),
			Checksum:    calculateChecksum(migration.Description),
		}

		data, _ := json.Marshal(migrationRecord)
		batch.Put([]byte(fmt.Sprintf("migration:%d", migration.Version)), data)

		// Update version
		batch.Put([]byte("db:version"), []byte(fmt.Sprintf("%d", migration.Version)))

		// Commit transaction
		if err := m.db.Write(batch, nil); err != nil {
			// Attempt rollback
			log.Printf("Migration failed, attempting rollback...")
			if rollbackErr := migration.Down(m.db); rollbackErr != nil {
				log.Printf("Rollback failed: %v", rollbackErr)
			}
			return fmt.Errorf("failed to commit migration %d: %w", migration.Version, err)
		}

		log.Printf("Migration %d completed successfully", migration.Version)
	}

	return nil
}

// Rollback rolls back to a specific version
func (m *MigrationManager) Rollback(targetVersion int) error {
	currentVersion, err := m.getCurrentVersion()
	if err != nil {
		return fmt.Errorf("failed to get current version: %w", err)
	}

	if targetVersion >= currentVersion {
		return fmt.Errorf("target version %d must be less than current version %d", targetVersion, currentVersion)
	}

	// Roll back migrations in reverse order
	for i := len(m.migrations) - 1; i >= 0; i-- {
		migration := m.migrations[i]
		if migration.Version <= targetVersion || migration.Version > currentVersion {
			continue
		}

		log.Printf("Rolling back migration %d: %s", migration.Version, migration.Description)

		if err := migration.Down(m.db); err != nil {
			return fmt.Errorf("rollback of migration %d failed: %w", migration.Version, err)
		}

		// Remove migration record
		m.db.Delete([]byte(fmt.Sprintf("migration:%d", migration.Version)), nil)

		log.Printf("Rollback of migration %d completed", migration.Version)
	}

	// Update version
	m.db.Put([]byte("db:version"), []byte(fmt.Sprintf("%d", targetVersion)), nil)

	return nil
}

// getCurrentVersion gets the current database version
func (m *MigrationManager) getCurrentVersion() (int, error) {
	data, err := m.db.Get([]byte("db:version"), nil)
	if err == leveldb.ErrNotFound {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}

	var version int
	fmt.Sscanf(string(data), "%d", &version)
	return version, nil
}

// GetMigrationHistory returns the history of applied migrations
func (m *MigrationManager) GetMigrationHistory() ([]Migration, error) {
	iter := m.db.NewIterator(nil, nil)
	defer iter.Release()

	var migrations []Migration
	prefix := []byte("migration:")

	for iter.Seek(prefix); iter.Valid() && iter.Key()[0] == prefix[0]; iter.Next() {
		var migration Migration
		if err := json.Unmarshal(iter.Value(), &migration); err != nil {
			continue
		}
		migrations = append(migrations, migration)
	}

	return migrations, iter.Error()
}

// Migration implementations

// Migration 1: Initialize schema
func migration001InitSchema(db *leveldb.DB) error {
	batch := new(leveldb.Batch)

	// Create initial keys
	batch.Put([]byte("chain:height"), []byte("0"))
	batch.Put([]byte("chain:genesis"), []byte("0x0000000000000000"))
	batch.Put([]byte("chain:initialized"), []byte("true"))

	return db.Write(batch, nil)
}

func migration001RollbackSchema(db *leveldb.DB) error {
	batch := new(leveldb.Batch)
	batch.Delete([]byte("chain:height"))
	batch.Delete([]byte("chain:genesis"))
	batch.Delete([]byte("chain:initialized"))
	return db.Write(batch, nil)
}

// Migration 2: Add indexes
func migration002AddIndexes(db *leveldb.DB) error {
	// Create index for transaction lookups
	// In a real implementation, this would scan existing transactions
	// and create index entries
	batch := new(leveldb.Batch)
	batch.Put([]byte("index:tx:initialized"), []byte("true"))
	batch.Put([]byte("index:addr:initialized"), []byte("true"))
	return db.Write(batch, nil)
}

func migration002RemoveIndexes(db *leveldb.DB) error {
	// Remove indexes
	iter := db.NewIterator(nil, nil)
	defer iter.Release()

	batch := new(leveldb.Batch)
	prefix := []byte("index:")

	for iter.Seek(prefix); iter.Valid() && iter.Key()[0] == prefix[0]; iter.Next() {
		batch.Delete(iter.Key())
	}

	return db.Write(batch, nil)
}

// Migration 3: Validator state
func migration003ValidatorState(db *leveldb.DB) error {
	batch := new(leveldb.Batch)
	batch.Put([]byte("validators:active:count"), []byte("0"))
	batch.Put([]byte("validators:total:stake"), []byte("0"))
	return db.Write(batch, nil)
}

func migration003RollbackValidatorState(db *leveldb.DB) error {
	batch := new(leveldb.Batch)
	batch.Delete([]byte("validators:active:count"))
	batch.Delete([]byte("validators:total:stake"))
	return db.Write(batch, nil)
}

// Migration 4: Transaction receipts
func migration004TransactionReceipts(db *leveldb.DB) error {
	batch := new(leveldb.Batch)
	batch.Put([]byte("receipts:initialized"), []byte("true"))
	return db.Write(batch, nil)
}

func migration004RollbackTransactionReceipts(db *leveldb.DB) error {
	// Remove all receipt entries
	iter := db.NewIterator(nil, nil)
	defer iter.Release()

	batch := new(leveldb.Batch)
	prefix := []byte("receipt:")

	for iter.Seek(prefix); iter.Valid() && iter.Key()[0] == prefix[0]; iter.Next() {
		batch.Delete(iter.Key())
	}

	batch.Delete([]byte("receipts:initialized"))
	return db.Write(batch, nil)
}

// Migration 5: Merkle roots
func migration005MerkleRoots(db *leveldb.DB) error {
	batch := new(leveldb.Batch)
	batch.Put([]byte("merkle:enabled"), []byte("true"))
	return db.Write(batch, nil)
}

func migration005RollbackMerkleRoots(db *leveldb.DB) error {
	batch := new(leveldb.Batch)
	batch.Delete([]byte("merkle:enabled"))

	// Remove all merkle entries
	iter := db.NewIterator(nil, nil)
	defer iter.Release()

	prefix := []byte("merkle:")
	for iter.Seek(prefix); iter.Valid() && iter.Key()[0] == prefix[0]; iter.Next() {
		batch.Delete(iter.Key())
	}

	return db.Write(batch, nil)
}

// Helper functions

func calculateChecksum(data string) string {
	// Simple checksum for migration integrity
	sum := 0
	for _, b := range []byte(data) {
		sum += int(b)
	}
	return fmt.Sprintf("%x", sum)
}

// ValidateDatabase checks database integrity
func ValidateDatabase(db *leveldb.DB) error {
	// Check if database is initialized
	initialized, err := db.Get([]byte("chain:initialized"), nil)
	if err != nil || string(initialized) != "true" {
		return fmt.Errorf("database not initialized")
	}

	// Check version
	version, err := db.Get([]byte("db:version"), nil)
	if err == leveldb.ErrNotFound {
		log.Println("Warning: No database version found, assuming version 0")
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read database version: %w", err)
	}

	log.Printf("Database validation passed, version: %s", version)
	return nil
}