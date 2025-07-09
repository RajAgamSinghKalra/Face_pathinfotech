#!/usr/bin/env python3
"""
Setup Script for Face Recognition Pipeline
==========================================

Automated setup and configuration for the face recognition pipeline.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🔍 Face Recognition Pipeline Setup")
    print("=" * 60)
    print("Oracle 23 AI + DirectML + InsightFace")
    print("=" * 60)

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Current: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_os():
    """Check operating system compatibility"""
    print("💻 Checking operating system...")
    system = platform.system()
    if system == "Windows":
        print(f"✅ Windows {platform.release()} - OK")
        return True
    elif system == "Linux":
        print(f"✅ Linux - OK (WSL supported)")
        return True
    else:
        print(f"⚠️  {system} - May have compatibility issues")
        return True

def check_directories():
    """Check and create required directories"""
    print("📁 Checking directories...")
    
    directories = [
        "logs",
        "cropped_faces",
        "models",
        "training_data",
        "uploads"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        else:
            print(f"✅ Exists: {directory}")
    
    return True

def check_oracle_path():
    """Check Oracle 23 AI installation path"""
    print("🗄️  Checking Oracle 23 AI path...")
    
    oracle_path = Path(r"C:\Users\Agam\Downloads\intern\pathinfotech\face\oracle23ai")
    if oracle_path.exists():
        print(f"✅ Oracle 23 AI found: {oracle_path}")
        return True
    else:
        print(f"❌ Oracle 23 AI not found at: {oracle_path}")
        print("   Please ensure Oracle 23 AI is installed correctly")
        return False

def check_dataset_path():
    """Check dataset path"""
    print("📊 Checking dataset path...")
    
    dataset_path = Path(r"C:\Users\Agam\Downloads\intern\pathinfotech\face\dataset")
    if dataset_path.exists():
        print(f"✅ Dataset found: {dataset_path}")
        
        # Check for LFW dataset
        lfw_path = dataset_path / "lfw-deepfunneled"
        if lfw_path.exists():
            print(f"✅ LFW dataset found: {lfw_path}")
            return True
        else:
            print(f"⚠️  LFW dataset not found in: {dataset_path}")
            return False
    else:
        print(f"❌ Dataset path not found: {dataset_path}")
        return False

def install_requirements():
    """Install Python requirements with DirectML fallback"""
    print("📦 Installing Python requirements...")
    
    try:
        # First try the main requirements
        if Path("requirements.txt").exists():
            print("   Trying main requirements (with DirectML)...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Main requirements installed successfully")
                return True
            else:
                print(f"⚠️  Main requirements failed: {result.stderr}")
                print("   Trying stable requirements (CPU fallback)...")
        
        # Fallback to stable requirements
        if Path("requirements_stable.txt").exists():
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements_stable.txt"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Stable requirements installed successfully (CPU mode)")
                return True
            else:
                print(f"❌ Stable requirements also failed: {result.stderr}")
                return False
        else:
            print("❌ No requirements files found")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def test_directml():
    """Test DirectML installation"""
    print("🎮 Testing DirectML...")
    
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"✅ DirectML initialized: {device}")
        return True
    except ImportError:
        print("❌ torch-directml not available")
        print("   The pipeline will use CPU fallback")
        print("   See DIRECTML_SETUP.md for installation help")
        return False
    except Exception as e:
        print(f"❌ DirectML error: {e}")
        print("   The pipeline will use CPU fallback")
        return False

def test_pytorch():
    """Test PyTorch installation (CPU fallback)"""
    print("🔥 Testing PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - OK")
        
        # Test basic operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print(f"✅ PyTorch operations: {z.shape}")
        return True
    except ImportError:
        print("❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_insightface():
    """Test InsightFace installation"""
    print("🤖 Testing InsightFace...")
    
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        print("✅ InsightFace imported successfully")
        return True
    except ImportError as e:
        print(f"❌ InsightFace import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ InsightFace error: {e}")
        return False

def test_oracle_connection():
    """Test Oracle database connection"""
    print("🔌 Testing Oracle connection...")
    
    try:
        import oracledb
        
        # Test connection
        connection = oracledb.connect(
            user="face_app",
            password="face_pass",
            dsn="localhost:1521/FREEPDB1",
            thick=False
        )
        connection.close()
        print("✅ Oracle connection successful")
        return True
        
    except ImportError:
        print("❌ oracledb not available")
        print("   Install with: pip install oracledb")
        return False
    except Exception as e:
        print(f"❌ Oracle connection failed: {e}")
        print("   Please ensure Oracle 23 AI is running and credentials are correct")
        return False

def create_config_file():
    """Create configuration file"""
    print("⚙️  Creating configuration file...")
    
    config = {
        "oracle": {
            "dsn": "localhost:1521/FREEPDB1",
            "user": "face_app",
            "password": "face_pass"
        },
        "paths": {
            "dataset": r"C:\Users\Agam\Downloads\intern\pathinfotech\face\dataset",
            "oracle": r"C:\Users\Agam\Downloads\intern\pathinfotech\face\oracle23ai",
            "cropped_faces": r"C:\Users\Agam\Downloads\intern\pathinfotech\face\cropped_faces",
            "models": r"C:\Users\Agam\Downloads\intern\pathinfotech\face\models",
            "uploads": r"C:\Users\Agam\Downloads\intern\pathinfotech\face\uploads"
        },
        "model": {
            "detection_confidence": 0.8,
            "similarity_threshold": 0.6,
            "top_k_results": 50,
            "face_size": [112, 112],
            "embedding_dim": 512
        },
        "processing": {
            "batch_size": 32,
            "max_workers": 4
        },
        "directml": {
            "enabled": False,
            "device": "cpu"
        }
    }
    
    # Update DirectML status
    try:
        import torch_directml
        config["directml"]["enabled"] = True
        config["directml"]["device"] = str(torch_directml.device())
    except ImportError:
        pass
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created: config.json")
    return True

def generate_sql_scripts():
    """Generate SQL setup scripts"""
    print("🗃️  Generating SQL scripts...")
    
    # Database setup SQL
    db_setup_sql = """
-- Oracle 23 AI Database Setup
-- ===========================

-- 1. Create user (run as SYSTEM/SYS)
CREATE USER face_app IDENTIFIED BY face_pass;
GRANT CONNECT, RESOURCE, CREATE VIEW, CREATE SEQUENCE TO face_app;
GRANT EXECUTE ON DBMS_HYBRID_VECTOR TO face_app;
GRANT UNLIMITED TABLESPACE TO face_app;

-- 2. Create faces table (run as face_app)
CREATE TABLE faces (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    original_image VARCHAR2(500),
    cropped_face_path VARCHAR2(500),
    face_id NUMBER,
    bbox_x1 NUMBER,
    bbox_y1 NUMBER,
    bbox_x2 NUMBER,
    bbox_y2 NUMBER,
    confidence NUMBER(5,4),
    bbox_width NUMBER,
    bbox_height NUMBER,
    embedding VECTOR(512, FLOAT32),
    image_blob BLOB,
    processing_timestamp TIMESTAMP DEFAULT SYSTIMESTAMP,
    metadata CLOB
);

-- 3. Create vector index
CREATE INDEX faces_embedding_idx ON faces (embedding) 
INDEXTYPE IS VECTOR_INDEX;

-- 4. Create sequence
CREATE SEQUENCE faces_seq START WITH 1 INCREMENT BY 1;

-- 5. Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON faces TO face_app;
"""
    
    with open("database_setup.sql", "w") as f:
        f.write(db_setup_sql)
    
    print("✅ SQL scripts generated:")
    print("   - database_setup.sql")
    return True

def main():
    """Main setup function"""
    print_banner()
    
    checks = []
    
    # System checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Operating System", check_os()))
    checks.append(("Directories", check_directories()))
    checks.append(("Oracle Path", check_oracle_path()))
    checks.append(("Dataset Path", check_dataset_path()))
    
    # Installation checks
    checks.append(("Requirements", install_requirements()))
    
    # Component tests
    checks.append(("PyTorch", test_pytorch()))
    checks.append(("DirectML", test_directml()))
    checks.append(("InsightFace", test_insightface()))
    checks.append(("Oracle Connection", test_oracle_connection()))
    
    # Configuration
    checks.append(("Configuration File", create_config_file()))
    checks.append(("SQL Scripts", generate_sql_scripts()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(checks)
    
    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run database setup: sqlplus system/password @database_setup.sql")
        print("2. Start pipeline: python 01_face_detection_cropping.py")
        print("3. View README.md for detailed instructions")
    elif passed >= total - 2:  # Allow some failures (like DirectML)
        print("✅ Setup completed with minor issues.")
        print("The pipeline will work with CPU fallback.")
        print("\nNext steps:")
        print("1. Run database setup: sqlplus system/password @database_setup.sql")
        print("2. Start pipeline: python 01_face_detection_cropping.py")
        print("3. For DirectML issues, see DIRECTML_SETUP.md")
    else:
        print("⚠️  Setup completed with significant issues.")
        print("Please resolve the failed checks before proceeding.")
    
    return passed >= total - 2  # Allow some failures

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 