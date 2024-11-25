"""
translated_fineweb Dataset (for srs pretraining)

{
    "headline":"Abiọla Ajimọbi ti pada ku o!\n",
    "content":"Gomina ana ....",
    "category":null,
    "url":"https:\/\/www.asejere.net\/abiola-ajimobi-ti-pada-ku-o\/"}
}
"""
import os
import re
import json
import argparse
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np
from transformers import AutoTokenizer
from typing import Union

TOKENIZER_MODEL = "castorini/afriteva_large"
LANGUAGES = {
    "Afrikaans": "afr",
    "Hausa": "hau",
    "Igbo": "ibo",
    "Shona": "sna",
    "Somali": "som",
    "Swahili": "swa",
    "Yoruba": "yor",
    "Zulu": "zul"
}

def write_datafile(filename: str, toks: Union[np.ndarray]):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens

    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks

    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
# ------------------------------------------
        
parser = argparse.ArgumentParser(description="translated_fineweb dataset preprocessing")
parser.add_argument("--split", type=str, default="train", help="Split to use")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
args = parser.parse_args()

assert args.split in ["train", "validation"], "split must be one of train, validation"
if args.split == "train":
    local_dir = "translated_fineweb_train"
elif args.split == "validation":
    local_dir = "translated_fineweb_eval"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
eot = tokenizer.eos_token_id

def clean_document(content:str):

    if content:
        # remove excess whitespace
        content = re.sub(r"\s+", " ", content)

        if content.endswith("\n"):
            content = content[:-1]

        if content.startswith("\r\n"):
            content = content[2:]
    
        return content

    return ""

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special </s> token delimits all documents

    headline = clean_document(doc["headline"], )
    content = clean_document(doc["content"], )
    tokens.extend(tokenizer(headline + "\n" +content)['input_ids'])
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

if __name__ == '__main__':
    language_tokens = {}
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
            
    fw = load_dataset("taresco/fineweb_translated")

    with mp.Pool(nprocs) as pool:

        shard_index = 0

        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < args.shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = args.split
                filename = os.path.join(DATA_CACHE_DIR, f"translated_fineweb_{split}_{shard_index:06d}.bin")

                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = args.split
            filename = os.path.join(DATA_CACHE_DIR, f"translated_fineweb_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])

# ------------------------------------------
# write token counts to a json file
with open(os.path.join(DATA_CACHE_DIR, "token_counts.json"), "w") as f:
    json.dump(language_tokens, f)