import os
import sys
import tempfile
from pathlib import Path

# isolate test artifacts and never call the real API from tests
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="genai_tests_")
os.environ["LLM_MODE"] = "mock"
os.environ["EMBEDDINGS_BACKEND"] = "local"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
