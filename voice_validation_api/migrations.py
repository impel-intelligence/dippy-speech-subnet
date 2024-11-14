import os
from typing import List, Optional
import psycopg
from psycopg.rows import dict_row
from datetime import datetime

class DatabaseMigrations:
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        
    def _ensure_migrations_table(self, conn) -> None:
        """Create migrations table if it doesn't exist"""
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_applied_migrations(self, conn) -> List[str]:
        """Get list of already applied migrations"""
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations ORDER BY version")
            return [row[0] for row in cur.fetchall()]

    def _get_migration_files(self) -> List[str]:
        """Get sorted list of migration files from migrations directory"""
        migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        if not os.path.exists(migrations_dir):
            return []
        
        files = [f for f in os.listdir(migrations_dir) 
                if f.endswith('.sql') and f.split('_')[0].isdigit()]
        return sorted(files)

    def migrate(self, target_version: Optional[str] = None) -> None:
        """Run all pending migrations or up to target_version if specified"""
        with psycopg.connect(self.conn_string) as conn:
            self._ensure_migrations_table(conn)
            applied = self._get_applied_migrations(conn)
            available = self._get_migration_files()

            pending = [f for f in available if f.replace('.sql', '') not in applied]
            if target_version:
                pending = [f for f in pending 
                          if f.replace('.sql', '') <= target_version]

            for migration_file in pending:
                version = migration_file.replace('.sql', '')
                print(f"Applying migration: {migration_file}")

                # Read and execute migration file
                migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
                with open(os.path.join(migrations_dir, migration_file)) as f:
                    sql = f.read()

                try:
                    with conn.cursor() as cur:
                        # Execute migration in a transaction
                        cur.execute(sql)
                        # Record successful migration
                        cur.execute(
                            "INSERT INTO schema_migrations (version) VALUES (%s)",
                            (version,)
                        )
                        conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"Error applying migration {migration_file}: {e}")
                    raise

    def create_migration(self, name: str) -> str:
        """Create a new migration file with timestamp prefix"""
        migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        os.makedirs(migrations_dir, exist_ok=True)
        
        # Generate version using timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{name}.sql"
        filepath = os.path.join(migrations_dir, filename)
        
        # Create empty migration file
        with open(filepath, 'w') as f:
            f.write(f"-- Migration: {name}\n\n")
            f.write("-- Write your migration SQL here\n")
        
        return filepath

# Example usage in application startup:
def initialize_database(app_config):
    migrations = DatabaseMigrations(app_config.db_connection_string)
    
    try:
        migrations.migrate()
        print("Database migrations completed successfully")
    except Exception as e:
        print(f"Failed to apply migrations: {e}")
        raise



if __name__ == "__main__":
    import argparse
    import os
    from typing import Optional

    parser = argparse.ArgumentParser(description='Database migration tool')
    parser.add_argument('command', choices=['create', 'migrate', 'status'],
                       help='Command to execute (create/migrate/status)')
    parser.add_argument('--name', help='Name for the new migration (required for create)')
    parser.add_argument('--db-url', help='Database connection URL', 
                       default=os.getenv('DATABASE_URL', 
                       'postgresql://vapi_user:vapi_password@postgres:5432/vapi_db'))

    args = parser.parse_args()

    migrations = DatabaseMigrations(args.db_url)

    if args.command == 'create':
        if not args.name:
            parser.error("--name is required for create command")
        filepath = migrations.create_migration(args.name)
        print(f"Created new migration at: {filepath}")

    elif args.command == 'migrate':
        try:
            migrations.migrate()
            print("Successfully applied all pending migrations")
        except Exception as e:
            print(f"Migration failed: {e}")
            exit(1)

    elif args.command == 'status':
        try:
            with migrations.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1")
                    result = cur.fetchone()
                    if result:
                        print(f"Current migration version: {result[0]}")
                    else:
                        print("No migrations have been applied yet")
                    
                    # Get list of available migrations
                    migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
                    available = sorted([f for f in os.listdir(migrations_dir) if f.endswith('.sql')])
                    print("\nAvailable migrations:")
                    for migration in available:
                        version = migration.split('_')[0]
                        name = '_'.join(migration.split('_')[1:]).replace('.sql', '')
                        status = "APPLIED" if result and version <= result[0] else "PENDING"
                        print(f"{version} - {name} ({status})")
        except Exception as e:
            print(f"Failed to get migration status: {e}")
            exit(1)
