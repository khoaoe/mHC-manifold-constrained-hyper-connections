"""
Downloads and tokenizes FineWeb-Edu 10B sample using Qwen2.5-0.5B tokenizer.
Saves pre-tokenized shards as np.uint32 bin files.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Download and pre-tokenize FineWeb-Edu 10B sample")
    parser.add_argument("--output-dir", type=str, default="data/fineweb10B", help="Output directory for shards")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="Number of tokens per shard")
    parser.add_argument("--subset", type=str, default="sample-10BT", help="Dataset subset (e.g., sample-10BT)")
    parser.add_argument("--max-shards", type=int, default=None, help="Maximum number of shards to download")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("📥 Loading Qwen2.5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    print(f"📥 Loading dataset '{args.subset}' from Hugging Face...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True)

    shard_idx = 0
    token_buffer = []
    total_tokens = 0

    print("🚀 Starting tokenization and sharding (writing to np.uint32 to prevent overflow)...")
    for entry in tqdm(dataset, desc="Processing documents"):
        text = entry["text"]
        # Encode text
        tokens = tokenizer.encode(text)
        token_buffer.extend(tokens)
        
        # Write shards
        while len(token_buffer) >= args.shard_size:
            shard_tokens = np.array(token_buffer[:args.shard_size], dtype=np.uint32)
            shard_path = os.path.join(args.output_dir, f"fineweb_{shard_idx:05d}.bin")
            shard_tokens.tofile(shard_path)
            print(f"\n💾 Saved shard {shard_idx} with {args.shard_size} tokens to {shard_path}")
            
            token_buffer = token_buffer[args.shard_size:]
            shard_idx += 1
            total_tokens += args.shard_size
            
            if args.max_shards is not None and shard_idx >= args.max_shards:
                print(f"\n✅ Reached maximum requested shards ({args.max_shards}). Stopping early.")
                return

    # Write remaining tokens
    if len(token_buffer) > 0:
        shard_tokens = np.array(token_buffer, dtype=np.uint32)
        shard_path = os.path.join(args.output_dir, f"fineweb_{shard_idx:05d}.bin")
        shard_tokens.tofile(shard_path)
        total_tokens += len(token_buffer)
        print(f"\n💾 Saved final shard {shard_idx} with {len(token_buffer)} tokens to {shard_path}")

    print(f"\n✅ Pre-tokenization complete! Total tokens processed: {total_tokens}")

if __name__ == "__main__":
    main()
