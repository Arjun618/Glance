# Intelligent Fashion Search Engine

A high-performance image retrieval system designed for fashion, capable of handling complex natural language queries like "red tie and white shirt".

## Features
- **Domain-Specific Logic**: Uses **FashionCLIP** for enhanced understanding of fashion attributes (colors, textures, styles).
- **Compositional Search**: Implements a split-and-merge strategy to handle complex queries (e.g., "A and B") better than standard dense retrieval.
- **Efficient Indexing**: Uses **FAISS** for fast similarity search over high-dimensional vectors.
- **Scalable Design**: Modular architecture separating indexing and retrieval.

## Setup

1.  **Environment**:
    It is recommended to use a virtual environment.
    ```bash
    # Create and activate a python env (e.g. using conda)
    conda create -n fashion_search python=3.10
    conda activate fashion_search
    
    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Data**:
    Download the test dataset from [[Here](https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip)] and extract the images into the `test/` directory.
    Ensure the directory structure looks like:
    ```
    project_root/
      ├── test/
      │   ├── image1.jpg
      │   ├── image2.jpg
      │   └── ...
    ```

## Implementation Details

### Part A: The Indexer (`src/indexer.py`)
- Loads **FashionCLIP**.
- Scans `test/` folder for images.
- Generates embeddings using `mps` (Metal Performance Shaders) acceleration on macOS.
- Saves FAISS index to `data/index.faiss`.

### Part B: The Retriever (`src/retriever.py`)
- Loads the FAISS index and metadata.
- **Standard Search**: Encodes query and finds nearest neighbors.
- **Compositional Search**: 
    - Detects conjunctions (e.g., "red tie *and* white shirt").
    - splits query into sub-parts.
    - Retrieves candidates for each part.
    - Fuses results based on combined relevance scores.

## Usage

### 1. Build Index
```bash
python src/indexer.py
```

### 2. Run Evaluation
```bash
python evaluate.py
```

### 3. Programmatic Usage
```python
from src.retriever import Retriever

retriever = Retriever()
results = retriever.search("Casual weekend outfit", k=3)
for res in results:
    print(res['filename'], res['score'])
```

## Performance & Tradeoffs
- **FashionCLIP vs vanilla CLIP**: FashionCLIP provides better zero-shot performance on fashion terminology, but vanilla CLIP is more robust for general "in the wild" scenes. We support falling back to vanilla CLIP if needed.
- **Compositionality**: Our "late fusion" approach (summing scores of parts) addresses the "bag of words" issue in CLIP, where it forgets attribute binding.
