# DirectML GPU Optimization Guide

## Overview

This document explains the DirectML optimizations made to enable AMD GPU acceleration for the face recognition pipeline.

## Changes Made

### 1. Updated FaceAnalysis Provider Configuration

All scripts now use DirectML as the primary provider with CPU fallback:

```python
# Before (CPU only)
self.app = FaceAnalysis(providers=['CPUExecutionProvider'])

# After (DirectML first, CPU fallback)
self.app = FaceAnalysis(
    allowed_modules=['detection'],  # Only load needed modules for speed
    providers=['DmlExecutionProvider', 'CPUExecutionProvider']
)
```

### 2. Files Updated

- `01_face_detection_cropping.py` - Face detection and cropping
- `04_query_similarity_search.py` - Face recognition and similarity search  
- `test_pipeline.py` - All test functions

### 3. Performance Optimizations

- **Detection-only mode**: When only face detection is needed, we load only the detection model
- **Recognition mode**: When face recognition is needed, we load both detection and recognition models
- **Provider fallback**: If DirectML fails, automatically falls back to CPU

## How to Verify DirectML is Working

### 1. Check Log Output

When you run the scripts, look for these log messages:

**✅ DirectML Working:**
```
Applied providers: ['DmlExecutionProvider', 'CPUExecutionProvider'], with options: {...}
```

**❌ CPU Only (DirectML not working):**
```
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
```

### 2. Performance Indicators

**With DirectML (GPU):**
- 20-50+ frames per second
- Processing hundreds of images in seconds
- GPU utilization visible in Task Manager

**Without DirectML (CPU only):**
- 3-4 frames per second  
- Processing hundreds of images takes minutes
- High CPU utilization

### 3. Test Commands

```bash
# Test face detection with DirectML
python 01_face_detection_cropping.py

# Test similarity search with DirectML
python 04_query_similarity_search.py --image test_image.jpg

# Run full test suite
python test_pipeline.py
```

## Troubleshooting

### DirectML Not Working

1. **Check torch-directml installation:**
   ```bash
   pip list | grep torch-directml
   ```

2. **Verify AMD GPU drivers:**
   - Ensure latest AMD drivers are installed
   - Check Device Manager shows your RX 6550M

3. **Test DirectML manually:**
   ```python
   import torch_directml
   device = torch_directml.device()
   print(f"DirectML device: {device}")
   ```

### Fallback to CPU

If DirectML fails, the pipeline automatically falls back to CPU execution. You'll see:
- Slower performance but still functional
- Log message: "Using providers: ['CPUExecutionProvider']"

## Performance Tuning

### Detection Speed vs Accuracy

For faster processing, you can reduce detection size:

```python
# Faster (less accurate)
self.app.prepare(ctx_id=0, det_size=(384, 384))

# Balanced (default)
self.app.prepare(ctx_id=0, det_size=(640, 640))

# More accurate (slower)
self.app.prepare(ctx_id=0, det_size=(1024, 1024))
```

### Memory Optimization

For large datasets, consider:
- Processing images in batches
- Using detection-only mode when possible
- Monitoring GPU memory usage

## Expected Performance

### AMD Radeon RX 6550M

**Face Detection:**
- 640x640: ~30-50 FPS
- 384x384: ~50-80 FPS

**Face Recognition:**
- Embedding generation: ~20-40 FPS
- Similarity search: ~100-200 comparisons/second

**Processing 1000 images:**
- With DirectML: 20-60 seconds
- Without DirectML: 5-15 minutes

## Monitoring

### Windows Task Manager
- Check GPU utilization in Performance tab
- Look for "GPU Engine" usage on your AMD GPU

### Log Files
Check the logs directory for detailed performance metrics:
```
logs/face_detection_YYYYMMDD_HHMMSS.log
logs/embedding_vectorization_YYYYMMDD_HHMMSS.log
```

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Test DirectML: `python test_pipeline.py`
3. Run face detection: `python 01_face_detection_cropping.py`
4. Monitor performance and GPU utilization

If you see "Applied providers: ['DmlExecutionProvider']" in the logs, congratulations! Your AMD GPU is now accelerating the face recognition pipeline. 