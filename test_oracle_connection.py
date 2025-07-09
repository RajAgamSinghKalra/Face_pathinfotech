#!/usr/bin/env python3
"""
Test Oracle 23 AI Connection
============================
Simple script to test database connectivity and basic operations.
"""

import oracledb
import os
import sys

# Configuration
ORACLE_DSN = "localhost:1521/FREE"
ORACLE_USER = "system"
ORACLE_PASSWORD = "1123"

def test_connection():
    """Test basic Oracle connection"""
    print("🔍 Testing Oracle 23 AI connection...")
    
    try:
        # Try thick mode first
        try:
            oracledb.init_oracle_client()
            print("✅ Oracle thick mode initialized")
            use_thick = True
        except Exception as e:
            print(f"⚠️ Oracle thick mode unavailable: {e}")
            use_thick = False
        
        # Connect to database
        if use_thick:
            conn = oracledb.connect(
                user=ORACLE_USER,
                password=ORACLE_PASSWORD,
                dsn=ORACLE_DSN
            )
        else:
            oracledb.defaults.fetch_lobs = False
            conn = oracledb.connect(
                user=ORACLE_USER,
                password=ORACLE_PASSWORD,
                dsn=ORACLE_DSN
            )
        
        print(f"✅ Connected to Oracle: {ORACLE_DSN}")
        
        # Test basic query
        cur = conn.cursor()
        cur.execute("SELECT version FROM v$instance")
        version = cur.fetchone()[0]
        print(f"✅ Oracle version: {version}")
        
        # Test vector operations
        print("🔍 Testing vector operations...")
        cur.execute("""
            SELECT COUNT(*) FROM user_tables 
            WHERE table_name = 'FACES'
        """)
        table_exists = cur.fetchone()[0] > 0
        print(f"✅ Faces table exists: {table_exists}")
        
        if table_exists:
            cur.execute("SELECT COUNT(*) FROM faces")
            count = cur.fetchone()[0]
            print(f"✅ Faces table has {count} records")
        
        cur.close()
        conn.close()
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
        # Provide helpful error messages
        if "ORA-12541" in str(e):
            print("\n💡 Solution: Start Oracle listener with 'lsnrctl start'")
        elif "ORA-12514" in str(e):
            print(f"\n💡 Solution: Check service name in DSN '{ORACLE_DSN}'")
            print("   Run 'lsnrctl status' to see available services")
        elif "ORA-01017" in str(e):
            print(f"\n💡 Solution: Check username/password for '{ORACLE_USER}'")
        elif "ORA-12154" in str(e):
            print(f"\n💡 Solution: Check DSN format: '{ORACLE_DSN}'")
            print("   Format should be: host:port/service_name")
        
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1) 