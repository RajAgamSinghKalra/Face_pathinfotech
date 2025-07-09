#!/usr/bin/env python3
"""
test_backend.py - Test backend connectivity and database setup
"""

import oracledb
import sys
from pathlib import Path

# Configuration
ORACLE_DSN = "localhost:1521/FREEPDB1"
ORACLE_USER = "system"
ORACLE_PWD = "1123"

def test_database_connection():
    """Test if we can connect to the Oracle database"""
    try:
        print("🔍 Testing Oracle database connection...")
        con = oracledb.connect(user=ORACLE_USER, password=ORACLE_PWD, dsn=ORACLE_DSN)
        cur = con.cursor()
        
        # Test basic query
        cur.execute("SELECT 1 FROM DUAL")
        result = cur.fetchone()
        print(f"✅ Database connection successful: {result[0]}")
        
        # Check if faces table exists
        cur.execute("""
            SELECT table_name 
            FROM user_tables 
            WHERE table_name = 'FACES'
        """)
        faces_table = cur.fetchone()
        
        if faces_table:
            print("✅ FACES table exists")
            
            # Check table structure
            cur.execute("""
                SELECT column_name, data_type 
                FROM user_tab_columns 
                WHERE table_name = 'FACES'
                ORDER BY column_id
            """)
            columns = cur.fetchall()
            print("📋 FACES table columns:")
            for col in columns:
                print(f"   - {col[0]}: {col[1]}")
            
            # Check if table has data
            cur.execute("SELECT COUNT(*) FROM faces")
            count = cur.fetchone()[0]
            print(f"📊 FACES table has {count} records")
            
        else:
            print("❌ FACES table does not exist")
            print("   You need to run the indexing script first:")
            print("   python 03_indexing.py")
        
        # Check if query_face table exists
        cur.execute("""
            SELECT table_name 
            FROM user_tables 
            WHERE table_name = 'QUERY_FACE'
        """)
        query_table = cur.fetchone()
        
        if query_table:
            print("✅ QUERY_FACE table exists")
        else:
            print("ℹ️  QUERY_FACE table will be created automatically")
        
        cur.close()
        con.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure Oracle database is running")
        print("2. Check if the connection parameters are correct:")
        print(f"   DSN: {ORACLE_DSN}")
        print(f"   User: {ORACLE_USER}")
        print("3. Try connecting with SQL*Plus to verify credentials")
        return False

def test_insightface_import():
    """Test if InsightFace can be imported"""
    try:
        print("\n🔍 Testing InsightFace import...")
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        print("✅ InsightFace imported successfully")
        return True
    except Exception as e:
        print(f"❌ InsightFace import failed: {e}")
        print("   Install with: pip install insightface")
        return False

def test_opencv_import():
    """Test if OpenCV can be imported"""
    try:
        print("\n🔍 Testing OpenCV import...")
        import cv2
        print(f"✅ OpenCV imported successfully (version {cv2.__version__})")
        return True
    except Exception as e:
        print(f"❌ OpenCV import failed: {e}")
        print("   Install with: pip install opencv-python")
        return False

def main():
    print("🧪 Backend System Test")
    print("=" * 50)
    
    # Test imports
    opencv_ok = test_opencv_import()
    insightface_ok = test_insightface_import()
    
    # Test database
    db_ok = test_database_connection()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   OpenCV: {'✅' if opencv_ok else '❌'}")
    print(f"   InsightFace: {'✅' if insightface_ok else '❌'}")
    print(f"   Database: {'✅' if db_ok else '❌'}")
    
    if all([opencv_ok, insightface_ok, db_ok]):
        print("\n🎉 All tests passed! Backend should work correctly.")
        print("   Start the backend with: python start_backend.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 