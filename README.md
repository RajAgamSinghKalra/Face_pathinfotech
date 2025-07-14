# Face Recognition Pipeline with Oracle 23 AI

A comprehensive Python-based face recognition pipeline using DirectML for AMD GPUs and Oracle 23 AI for vector storage and similarity search.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dataset       │    │   Face Detection│    │   Embedding     │
│   Images        │───▶│   & Cropping    │───▶│   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Oracle APEX   │    │   Oracle 23 AI  │    │   Similarity    │
│   Frontend      │◀───│   Vector DB     │◀───│   Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Features

- **AMD GPU Support**: DirectML acceleration for AMD Radeon RX 6550M
- **Oracle 23 AI Integration**: Native vector operations and similarity search
- **InsightFace Models**: State-of-the-art face detection and recognition
- **APEX Frontend**: Web-based interface for face search
- **REST API**: Programmatic access to face recognition services
- **Modular Design**: Self-contained scripts for each major task

## 📋 Requirements

### Hardware & OS
- **OS**: Windows 10/11 (or WSL)
- **GPU**: AMD Radeon RX 6550M (DirectML support)
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for dataset and models

### Software
- **Python**: 3.10+
- **Oracle 23 AI**: Installed at `C:\Users\Agam\Downloads\intern\pathinfotech\face\oracle23ai`
- **Dataset**: LFW dataset at `C:\Users\Agam\Downloads\intern\pathinfotech\face\dataset`

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd C:\Users\Agam\Downloads\intern\pathinfotech\face

# Install dependencies
pip install -r requirements.txt

# Verify DirectML installation
python -c "import torch_directml; print('DirectML OK')"
```

### 2. Database Setup

```bash
# Test Oracle connection first
python test_oracle_connection.py
```

If the test passes, you're ready to proceed. If it fails, check:
- Oracle listener is running: `lsnrctl status`
- Database is started: `sqlplus / as sysdba` → `STARTUP;`
- Service name is correct: `lsnrctl status` shows "FREE" service

```sql
-- Optional: Create dedicated user (using system for now)
-- CREATE USER face_app IDENTIFIED BY face_pass;
-- GRANT CONNECT, RESOURCE, CREATE VIEW, CREATE SEQUENCE TO face_app;
-- GRANT EXECUTE ON DBMS_HYBRID_VECTOR TO face_app;
```

### 3. Pipeline Execution

```bash
# Step 1: Face Detection & Cropping
python 01_face_detection_cropping.py

# Step 2: Embedding & Vectorization
python 02_embedding_vectorization.py

# Step 3: Custom Training (Optional)
python 03_custom_training_indexing.py --train

# Step 4: Query & Similarity Search
python 04_query_similarity_search.py --image path/to/query.jpg

# Step 5: Start REST API for APEX
python 05_apex_rest_api.py
```

## 📁 Script Descriptions

### Script 1: `01_face_detection_cropping.py`
- **Purpose**: Detect and crop faces from dataset images
- **Technology**: InsightFace RetinaFace
- **Input**: All images in dataset directory
- **Output**: Cropped faces + metadata CSV
- **DirectML**: GPU-accelerated face detection

### Script 2: `02_embedding_vectorization.py`
- **Purpose**: Generate face embeddings and store in Oracle 23 AI
- **Technology**: ArcFace (512-D embeddings)
- **Process**: Load crops → compute embeddings → insert into Oracle
- **Database**: VECTOR(512, FLOAT32) + BLOB storage

### Script 3: `03_custom_training_indexing.py`
- **Purpose**: Optional custom model training
- **Technology**: PyTorch + DirectML
- **Features**: Fine-tuning, ONNX export
- **Usage**: `--train` flag to enable training

### Script 4: `04_query_similarity_search.py`
- **Purpose**: Query faces and find similar matches
- **Features**: Upload image → detect → search → group results
- **Search**: Oracle 23 AI vector similarity search
- **Output**: Grouped identity clusters

### Script 5: `05_apex_rest_api.py`
- **Purpose**: REST API for APEX integration
- **Framework**: FastAPI
- **Endpoints**: Face search, health check
- **APEX**: SQL snippets for frontend integration

## 🔧 Configuration

### DirectML Setup
```python
# Automatic detection in all scripts
try:
    import torch_directml
    device = torch_directml.device()
    print(f"✅ DirectML: {device}")
except ImportError:
    device = torch.device("cpu")
    print("❌ DirectML not available")
```

### Database Configuration
```python
# Oracle 23 AI connection (Updated for your setup)
ORACLE_DSN = "localhost:1521/FREE"
ORACLE_USER = "system"
ORACLE_PASSWORD = "1123"
```

### Model Parameters
```python
# Face detection
DETECTION_CONFIDENCE = 0.8
FACE_SIZE = (112, 112)

# Similarity search
SIMILARITY_THRESHOLD = 0.6
TOP_K_RESULTS = 50
```

## 🗄️ Database Schema

```sql
-- Faces table with vector support
CREATE TABLE faces (
    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    original_image VARCHAR2(500),
    cropped_face_path VARCHAR2(500),
    face_id NUMBER,
    bbox_x1 NUMBER, bbox_y1 NUMBER, bbox_x2 NUMBER, bbox_y2 NUMBER,
    confidence NUMBER(5,4),
    bbox_width NUMBER, bbox_height NUMBER,
    embedding VECTOR(512, FLOAT32),
    image_blob BLOB,
    processing_timestamp TIMESTAMP DEFAULT SYSTIMESTAMP,
    metadata CLOB
);

-- Vector index for similarity search
CREATE INDEX faces_embedding_idx ON faces (embedding) 
INDEXTYPE IS VECTOR_INDEX;
```

## 🌐 APEX Integration

### 1. REST API Setup
```bash
# Start the REST API
python 05_apex_rest_api.py
```

### 2. APEX Application
- **File Upload**: Drag-and-drop image upload
- **Search Processing**: Automatic face detection and search
- **Results Display**: Grouped identity galleries
- **Similarity Scores**: Visual confidence indicators

### 3. APEX SQL Snippets
```sql
-- File upload process
DECLARE
    l_response CLOB;
    l_url VARCHAR2(500) := 'http://localhost:8000/api/face-search';
BEGIN
    l_response := apex_web_service.make_request(
        p_url => l_url,
        p_http_method => 'POST',
        p_body => '{"file": "' || :P1_UPLOAD_FILE || '"}',
        p_content_type => 'application/json'
    );
    :P1_SEARCH_RESULTS := l_response;
END;
```

## 📊 Performance Optimization

### GPU Acceleration
- **DirectML**: AMD GPU acceleration for all ML operations
- **Batch Processing**: Optimized batch sizes for your hardware
- **Memory Management**: Efficient tensor operations

### Database Optimization
- **Vector Indexing**: Oracle 23 AI vector indexes for fast search
- **Batch Inserts**: Bulk operations for large datasets
- **Connection Pooling**: Efficient database connections

### Caching
- **Model Caching**: Pre-loaded models for faster inference
- **Result Caching**: Cached search results for repeated queries

## 🐛 Troubleshooting

### Common Issues

1. **DirectML Not Available**
   ```bash
   pip install torch-directml
   # Verify AMD GPU drivers are up to date
   ```

2. **Oracle Connection Failed**
   ```bash
   # Check Oracle service is running
   # Verify connection parameters
   # Test with SQL*Plus
   ```

3. **Out of Memory**
   ```python
   # Reduce batch size
   BATCH_SIZE = 16  # Instead of 32
   ```

4. **Face Detection Issues**
   ```python
   # Adjust confidence threshold
   DETECTION_CONFIDENCE = 0.5  # Lower for more faces
   ```

5. **Thumbnails return 403**
   ```bash
   # Allow your cropped_faces directory
   set CROPPED_FACES_DIR=C:\path\to\face\cropped_faces
   # or
   set EXTRA_STATIC_ROOTS=C:\path\to\face\cropped_faces
   ```

### Logging
All scripts generate detailed logs in the `logs/` directory:
- `face_detection_YYYYMMDD_HHMMSS.log`
- `embedding_vectorization_YYYYMMDD_HHMMSS.log`
- `similarity_search_YYYYMMDD_HHMMSS.log`

## 📈 Monitoring & Analytics

### Performance Metrics
- **Processing Speed**: Images per second
- **Detection Accuracy**: Face detection confidence
- **Search Quality**: Similarity score distributions
- **Database Performance**: Query response times

### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Database connectivity
python -c "import oracledb; print('DB OK')"
```

## 🔒 Security Considerations

- **Input Validation**: All uploaded images are validated
- **SQL Injection**: Parameterized queries prevent injection
- **File Upload**: Secure file handling and validation
- **API Security**: CORS configuration for web access

## 📚 API Documentation

### REST Endpoints

#### POST `/api/face-search`
Upload an image and search for similar faces.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/face-search" \
  -F "file=@query_image.jpg" \
  -F "similarity_threshold=0.6" \
  -F "top_k=50"
```

**Response:**
```json
{
  "success": true,
  "message": "Face search completed",
  "data": {
    "query_image": "query_image.jpg",
    "num_faces_detected": 1,
    "search_results": [...]
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### GET `/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "database_connected": true
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Create an issue with detailed error information

## 🎉 Acknowledgments

- **InsightFace**: State-of-the-art face recognition models
- **Oracle 23 AI**: Vector database capabilities
- **DirectML**: AMD GPU acceleration
- **FastAPI**: Modern Python web framework "# Face_pathinfotech" 
