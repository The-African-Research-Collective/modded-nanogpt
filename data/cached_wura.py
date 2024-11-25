import os
import sys
from huggingface_hub import snapshot_download

# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get():
    local_dir = os.path.join(os.path.dirname(__file__), 'wura_train')
    snapshot_download(repo_id="taresco/cached_wura",  allow_patterns="*.bin",
                    repo_type="dataset", local_dir=local_dir)
