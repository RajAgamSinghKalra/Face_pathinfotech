#!/usr/bin/env python3
"""
Test Script for Face Recognition Pipeline
=========================================

Quick tests to verify all components are working correctly.
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    imports = [
        ("torch_directml", "DirectML"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("insightface", "InsightFace"),
        ("oracledb", "Oracle DB"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn")
    ]
    
    results = []
    for module, name in imports:
        try:
            __import__(module)
            print(f"✅ {name}")
            results.append(True)
        except ImportError:
            print(f"❌ {name}")
            results.append(False)
    
    return all(results)

def test_directml():
    """Test DirectML functionality"""
    print("\n🎮 Testing DirectML...")
    
    try:
        import torch_directml
        import torch
        
        device = torch_directml.device()
        print(f"✅ DirectML device: {device}")
        
        # Test tensor operations
        x = torch.randn(2, 3).to(device)
        y = torch.randn(3, 2).to(device)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations: {z.shape}")
        
        return True
    except Exception as e:
        print(f"❌ DirectML test failed: {e}")
        return False

def test_insightface():
    """Test InsightFace models"""
    print("\n🤖 Testing InsightFace...")
    
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        
        # Test face detection
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ Face detection model loaded")
        
        # Test embedding model
        model = get_model('glint360k_r100_fp16_0.1')
        model.eval()
        print("✅ Embedding model loaded")
        
        return True
    except Exception as e:
        print(f"❌ InsightFace test failed: {e}")
        return False

def test_oracle_connection():
    """Test Oracle database connection"""
    print("\n🗄️  Testing Oracle connection...")
    
    try:
        import oracledb
        
        # Test connection
        connection = oracledb.connect(
            user="face_app",
            password="face_pass",
            dsn="localhost:1521/FREEPDB1",
            thick=False
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM DUAL")
        result = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        print("✅ Oracle connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Oracle connection failed: {e}")
        return False

def test_face_detection():
    """Test face detection on sample image"""
    print("\n👤 Testing face detection...")
    
    try:
        from insightface.app import FaceAnalysis
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Initialize face detector
        app = FaceAnalysis(
            allowed_modules=['detection'],
            providers=['DmlExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Detect faces
        faces = app.get(test_image)
        print(f"✅ Face detection: {len(faces)} faces detected")
        
        return True
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation"""
    print("\n🔢 Testing embedding generation...")
    
    try:
        from insightface.model_zoo import get_model
        import torch
        
        # Load model
        model = get_model('glint360k_r100_fp16_0.1')
        model.eval()
        
        # Create test input
        test_input = torch.randn(1, 3, 112, 112)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(test_input)
        
        print(f"✅ Embedding generated: {embedding.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation test failed: {e}")
        return False

def test_file_structure():
    """Test file structure and permissions"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "01_face_detection_cropping.py",
        "02_embedding_vectorization.py",
        "03_custom_training_indexing.py",
        "04_query_similarity_search.py",
        "05_apex_rest_api.py"
    ]
    
    required_dirs = [
        "logs",
        "cropped_faces",
        "models",
        "training_data",
        "uploads"
    ]
    
    all_good = True
    
    # Check files
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            all_good = False
    
    # Check directories
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/")
            all_good = False
    
    return all_good

def test_config():
    """Test configuration file"""
    print("\n⚙️  Testing configuration...")
    
    try:
        if Path("config.json").exists():
            with open("config.json", "r") as f:
                config = json.load(f)
            
            required_keys = ["oracle", "paths", "model", "processing"]
            for key in required_keys:
                if key in config:
                    print(f"✅ {key}")
                else:
                    print(f"❌ {key}")
                    return False
            
            return True
        else:
            print("❌ config.json not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_performance_test():
    """Run a quick performance test"""
    print("\n⚡ Performance test...")
    
    try:
        import time
        from insightface.app import FaceAnalysis
        
        # Initialize detector
        app = FaceAnalysis(
            allowed_modules=['detection'],
            providers=['DmlExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Time detection
        start_time = time.time()
        for _ in range(5):
            faces = app.get(test_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5
        print(f"✅ Average detection time: {avg_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Face Recognition Pipeline Test Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("DirectML", test_directml),
        ("InsightFace", test_insightface),
        ("Oracle Connection", test_oracle_connection),
        ("Face Detection", test_face_detection),
        ("Embedding Generation", test_embedding_generation),
        ("File Structure", test_file_structure),
        ("Configuration", test_config),
        ("Performance", run_performance_test)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run: python 01_face_detection_cropping.py")
        print("2. Run: python 02_embedding_vectorization.py")
        print("3. Run: python 04_query_similarity_search.py --image <test_image>")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nTroubleshooting:")
        print("1. Run: python setup.py")
        print("2. Check README.md for detailed instructions")
        print("3. Verify Oracle 23 AI is running")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 