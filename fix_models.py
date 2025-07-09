#!/usr/bin/env python3
"""
fix_models.py - Clear corrupted InsightFace models and re-download them
"""

import shutil
import os
from pathlib import Path

def clear_insightface_cache():
    """Clear the InsightFace model cache to force re-download"""
    cache_dir = Path.home() / ".insightface"
    
    if cache_dir.exists():
        print(f"🗑️  Clearing InsightFace cache: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return False
    else:
        print("ℹ️  No InsightFace cache found")
    
    return True

def test_model_download():
    """Test if models can be downloaded correctly"""
    try:
        print("\n🔍 Testing InsightFace model download...")
        from insightface.app import FaceAnalysis
        
        # This will trigger model download
        print("📥 Downloading buffalo_l model...")
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        print("✅ Model downloaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def main():
    print("🔧 Fixing InsightFace Models")
    print("=" * 40)
    
    # Clear cache
    if not clear_insightface_cache():
        print("❌ Failed to clear cache")
        return
    
    # Test download
    if test_model_download():
        print("\n🎉 Models fixed successfully!")
        print("   You can now start the backend server.")
    else:
        print("\n❌ Failed to fix models")
        print("   Try running this script again or check your internet connection.")

if __name__ == "__main__":
    main() 