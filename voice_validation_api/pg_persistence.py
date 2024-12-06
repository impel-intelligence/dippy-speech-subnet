import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel


class MinerEntry(BaseModel):
    hotkey: str
    hash: str
    block: int


class HashEntry(BaseModel):
    hash: str
    status: str
    repo_name: str
    repo_namespace: str
    safetensors_hash: Optional[str] = None
    total_score: Optional[float] = None
    timestamp: Optional[datetime] = None
    notes: Optional[str] = None


class MinerWithHashes(BaseModel):
    miner_entries: List[MinerEntry]
    hash_entries: List[HashEntry]


class Persistence:
    def __init__(
        self,
        connection_string: str = "postgresql://vapi:vapi@localhost:5432/vapi",
        migrations_path="./voice_validation_api/migrations",
    ):
        self.logger = logging.getLogger(__name__)
        self.pool = ConnectionPool(connection_string, kwargs={"row_factory": dict_row}, min_size=5, max_size=20, timeout=30)
        self.migrations_path = migrations_path

    def ensure_connection(self) -> bool:
        """
        Tests database connectivity by running a simple query.
        Returns True if successful, False if connection fails.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT 1 LIMIT 1")
                    return True
                except psycopg.Error as e:
                    self.logger.error(f"Database connection test failed: {e}")
                    return False

    def insert_miner_entry(self, entry: MinerEntry) -> bool:
        """
        Inserts a new MinerEntry if it doesn't exist.
        Returns True if inserted, False if already existed.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO miner_entries (hotkey, hash, block)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (hotkey, hash, block) DO NOTHING
                        RETURNING hotkey
                    """,
                        (entry.hotkey, entry.hash, entry.block),
                    )

                    result = cur.fetchone()
                    conn.commit()

                    return result is not None  # True if inserted, False if already existed
                except psycopg.Error as e:
                    conn.rollback()
                    raise Exception(f"Failed to insert miner entry: {e}")

    def insert_hash_entry(self, entry: HashEntry) -> bool:
        """
        Inserts a new HashEntry if it doesn't exist.
        Returns True if inserted, False if already existed.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO hash_entries (hash, safetensors_hash, status)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (hash) DO NOTHING
                        RETURNING hash
                    """,
                        (entry.hash, entry.safetensors_hash, entry.status),
                    )

                    result = cur.fetchone()
                    conn.commit()

                    return result is not None  # True if inserted, False if already existed
                except psycopg.Error as e:
                    conn.rollback()
                    raise Exception(f"Failed to insert hash entry: {e}")

    def run_migrations(self):
        """
        Runs SQL migrations from the migrations directory.
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Get list of migration files
                    migration_files = sorted([f for f in os.listdir(self.migrations_path) if f.endswith(".sql")])

                    for migration_file in migration_files:
                        migration_path = os.path.join(self.migrations_path, migration_file)
                        with open(migration_path, "r") as f:
                            migration_sql = f.read()

                        # Execute the migration
                        cur.execute(migration_sql)

                    conn.commit()
                except psycopg.Error as e:
                    conn.rollback()
                    raise Exception(f"Failed to run migrations: {e}")
                except Exception as e:
                    conn.rollback()
                    raise Exception(f"Failed to read/parse migrations: {e}")

    def update_leaderboard_status(self, hash: str, status: str, notes: str = "") -> Optional[Dict]:
        """Updates status and notes for a hash entry"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        UPDATE hash_entries 
                        SET status = %s, notes = %s
                        WHERE hash = %s
                        RETURNING *
                    """,
                        (status, notes, hash),
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return result
                except psycopg.Error as e:
                    conn.rollback()
                    self.logger.error(f"Error updating leaderboard status for {hash}: {e}")
                    return None

    def update_leaderboard_success(
        self,
        hash: str,
        status: str,
        total_score: float,
        notes: str = "",
    ) -> Optional[Dict]:
        """
        Updates the status, notes, and total_score for a hash entry.

        Args:
            hash (str): The hash to update.
            status (str): The new status value.
            notes (str): Additional notes (default: "").
            total_score (float): The new total score.

        Returns:
            Optional[Dict]: The updated row data, or None if the operation fails.
        """
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    # Construct the SQL query using parameterized placeholders
                    query = """
                            UPDATE hash_entries 
                            SET status = %s, notes = %s, total_score = %s
                            WHERE hash = %s
                            RETURNING *
                        """
                    # Execute the query with the provided parameters
                    cur.execute(query, (status, notes, total_score, hash))
                    result = cur.fetchone()
                    conn.commit()
                    return result
                except psycopg.Error as e:
                    conn.rollback()
                    self.logger.error(f"Error updating leaderboard status for {hash}: {e}")
                    return None

    def get_json_result(self, hash: str) -> Optional[Dict]:
        """Gets formatted result for a hash entry"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT * FROM hash_entries WHERE hash = %s
                    """,
                        (hash,),
                        prepare=False,
                    )
                    row = cur.fetchone()

                    if row:
                        return {
                            "score": {
                                "total_score": row["total_score"],
                            },
                            "details": {
                                "safetensors_hash": row["safetensors_hash"],
                            },
                            "hash": row["hash"],
                            "status": row["status"],
                        }
                    return None
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching leaderboard entry: {e}")
                    return None

    def get_from_hash(self, hash: str) -> Optional[Dict]:
        """Gets formatted result for a hash entry"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT * FROM hash_entries WHERE hash = %s
                    """,
                        (hash,),
                        prepare=False,
                    )
                    row = cur.fetchone()

                    if row:
                        return {
                            "score": {
                                "total_score": row["total_score"],
                            },
                            "details": {
                                "safetensors_hash": row["safetensors_hash"],
                            },
                            "status": row["status"],
                        }
                    return None
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching leaderboard entry: {e}")
                    return None

    def get_internal_result(self, hash: str) -> Optional[Dict]:
        """Gets raw database row for a hash entry"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT * FROM hash_entries WHERE hash = %s
                    """,
                        (hash,),
                    )
                    return cur.fetchone()
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching internal result: {e}")
                    return None

    def get_top_completed(self, limit: int = 10) -> List[Dict]:
        """Gets top completed entries sorted by total score"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT * FROM hash_entries 
                        WHERE status = 'COMPLETED'
                        ORDER BY total_score DESC
                        LIMIT %s
                    """,
                        (limit,),
                    )
                    return cur.fetchall()
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching top completed: {e}")
                    return []

    def get_next_model_to_eval(self) -> Optional[HashEntry]:
        """Gets the next queued model for evaluation"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT hash, total_score, alpha_score, beta_score, gamma_score, 
                               notes, repo_namespace, repo_name, timestamp, safetensors_hash, status
                        FROM hash_entries 
                        WHERE status = 'QUEUED'
                        ORDER BY timestamp ASC
                        LIMIT 1
                    """,
                    prepare=False,  # Disables prepared statements
                    )
                    row = cur.fetchone()
                    if row:
                        return HashEntry(**row)
                    return None
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching next model: {e}")
                    return None

    def get_failed_model_to_eval(self) -> Optional[Dict]:
        """Gets the most recent failed model"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT * FROM hash_entries 
                        WHERE status = 'FAILED'
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """
                    )
                    return cur.fetchone()
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching failed model: {e}")
                    return None

    def remove_record(self, hash: str) -> bool:
        """Removes a hash entry record"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        DELETE FROM hash_entries 
                        WHERE hash = %s
                        RETURNING hash
                    """,
                        (hash,),
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return result is not None
                except psycopg.Error as e:
                    conn.rollback()
                    self.logger.error(f"Error removing record: {e}")
                    return False

    def last_uploaded_model(self, miner_hotkey: str) -> Optional[Dict]:
        """Gets the last uploaded model for a miner"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT m.*, h.status 
                        FROM miner_entries m
                        LEFT JOIN hash_entries h ON m.hash = h.hash
                        WHERE m.hotkey = %s
                        ORDER BY m.block DESC
                        LIMIT 1
                    """,
                        (miner_hotkey,),
                    )
                    return cur.fetchone()
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching last uploaded model: {e}")
                    return None

    def update_minerboard_status(
        self,
        hash_entry: str,
        uid: int,
        hotkey: str,
        block: int,
    ) -> Optional[Dict]:
        """Updates or inserts a miner entry record"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    # First check if hash exists in hash_entries
                    cur.execute(
                        """
                        SELECT hash FROM hash_entries 
                        WHERE hash = %s
                    """,
                        (hash_entry,),
                    )

                    if not cur.fetchone():
                        self.logger.error(f"No hash_entry found for hash {hash_entry}")
                        return None

                    # Then handle miner_entries upsert
                    cur.execute(
                        """
                        INSERT INTO miner_entries (
                            hash,
                            uid, 
                            hotkey,
                            block
                        ) VALUES (
                            %s, %s, %s, %s
                        )
                        ON CONFLICT ON CONSTRAINT miner_entries_pkey
                        DO UPDATE SET
                            uid = EXCLUDED.uid,
                            hotkey = EXCLUDED.hotkey, 
                            block = EXCLUDED.block
                        RETURNING *
                    """,
                        (hash_entry, uid, hotkey, block),
                        prepare=False,
                    )

                    result = cur.fetchone()
                    conn.commit()
                    return result
                except psycopg.Error as e:
                    conn.rollback()
                    self.logger.error(f"Error updating miner entry status for {hash_entry}: {e}")
                    return None

    def upsert_and_return(self, entry: Dict, hash: str) -> Optional[Dict]:
        """Upserts a hash entry and returns the updated record"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Convert entry dict to proper format
                    hash_entry = {
                        "hash": hash,
                        "repo_namespace": entry.get("repo_namespace"),
                        "repo_name": entry.get("repo_name"),
                        "total_score": entry.get("total_score", 0),
                        "status": entry.get("status"),
                        "notes": entry.get("notes", ""),
                        "timestamp": entry.get("timestamp"),
                    }

                    # Upsert the record
                    cur.execute(
                        """
                        INSERT INTO hash_entries (
                            hash,
                            repo_namespace,
                            repo_name,
                            total_score,
                            status,
                            notes,
                            timestamp
                        ) VALUES (
                            %(hash)s,
                            %(repo_namespace)s,
                            %(repo_name)s,
                            %(total_score)s,
                            %(status)s,
                            %(notes)s,
                            %(timestamp)s
                        )
                        ON CONFLICT (hash) 
                        DO UPDATE SET
                            repo_namespace = EXCLUDED.repo_namespace,
                            repo_name = EXCLUDED.repo_name,
                            total_score = EXCLUDED.total_score,
                            status = EXCLUDED.status,
                            notes = EXCLUDED.notes,
                            timestamp = EXCLUDED.timestamp
                        RETURNING *
                    """,
                        hash_entry,
                    )

                    result = cur.fetchone()
                    conn.commit()
                    return result
                except psycopg.Error as e:
                    conn.rollback()
                    self.logger.error(f"Error upserting hash entry: {e}")
                    return None
    def insert(self, entry: Dict, hash: str) -> bool:
        """Inserts a hash entry and returns True if successful, False if entry exists or on error"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Check if entry already exists
                    cur.execute("SELECT hash FROM hash_entries WHERE hash = %s", (hash,))
                    if cur.fetchone() is not None:
                        return False

                    # Convert entry dict to proper format
                    hash_entry = {
                        "hash": hash,
                        "repo_namespace": entry.get("repo_namespace"),
                        "repo_name": entry.get("repo_name"), 
                        "total_score": entry.get("total_score", 0),
                        "status": entry.get("status"),
                        "notes": entry.get("notes", ""),
                        "timestamp": entry.get("timestamp"),
                    }

                    # Insert the record
                    cur.execute(
                        """
                        INSERT INTO hash_entries (
                            hash,
                            repo_namespace,
                            repo_name,
                            total_score,
                            status,
                            notes,
                            timestamp
                        ) VALUES (
                            %(hash)s,
                            %(repo_namespace)s,
                            %(repo_name)s,
                            %(total_score)s,
                            %(status)s,
                            %(notes)s,
                            %(timestamp)s
                        )
                        """,
                        hash_entry,
                    )

                    conn.commit()
                    return True
                except psycopg.Error as e:
                    conn.rollback()
                    self.logger.error(f"Error inserting hash entry: {e}")
                    return False

    def fetch_all_miner_entries(self) -> Optional[List[Dict]]:
        """Gets all miner entries with their corresponding hash entry data"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT m.*, h.status, h.safetensors_hash, h.total_score, h.timestamp, h.notes
                        FROM miner_entries m
                        LEFT JOIN hash_entries h ON m.hash = h.hash
                        
                    """
                    )
                    return cur.fetchall()
                except psycopg.Error as e:
                    self.logger.error(f"Error fetching miner entries: {e}")
                    return None

    def fetch_recent_entries(self, limit: int = 256) -> List[HashEntry]:
        """
        Fetch the most recent entries from the hash_entries table

        Args:
            limit (int): Maximum number of entries to return

        Returns:
            List[HashEntry]: List of recent entries
        """
        query = """
            SELECT 
                hash,
                total_score,
                alpha_score,
                beta_score,
                gamma_score,
                notes,
                repo_namespace,
                repo_name,
                timestamp,
                safetensors_hash,
                status
            FROM hash_entries
            ORDER BY timestamp DESC
            LIMIT %s
        """
        try:
            with self.pool.connection() as conn:
                with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute(query, (limit,))
                    results = cur.fetchall()
                    return [HashEntry(**row) for row in results]
        except Exception as e:
            self.logger.error(f"Error fetching recent entries: {e}")
            return None


# Example usage:
def main():
    # Initialize with connection string
    db = Persistence("postgresql://user:pass@localhost:5432/dbname")

    # Create tables if they don't exist
    db.run_migrations()

    # Insert some test data
    hash_entry = HashEntry(hash="hash1", safetensors_hash="safetensors1", status="active")
    db.insert_hash_entry(hash_entry)

    miner_entry = MinerEntry(hotkey="key1", hash="hash1", block=100)
    db.insert_miner_entry(miner_entry)

    # Fetch data
    result = db.get_miner_with_hashes("key1")
    print(f"Found {len(result.miner_entries)} miner entries")
    print(f"Found {len(result.hash_entries)} hash entries")


if __name__ == "__main__":
    main()
