pip install flask-cors flask bm25s sentence_transformers datasets faiss-cpu torch transformers nltk accelerate
pip install --upgrade torch transformers sentence-transformers



# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with correct CUDA version if needed (check your system)
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or for CUDA 11.8 (check your system's CUDA version with `nvidia-smi` if GPU is available):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install specific versions of transformers and sentence-transformers known to work together
# Check the sentence-transformers docs for recommended transformers version
pip install transformers>=4.36.0
pip install sentence-transformers>=2.2.0 # Or latest stable
pip install bm25s datasets faiss-cpu # Your other requirements