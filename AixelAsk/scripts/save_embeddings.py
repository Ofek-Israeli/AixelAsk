import json
import hashlib
from tqdm import tqdm
import sys
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from processing_format import get_row_flattened, get_col_description
from utils.processing import clean_table
from utils.request_gpt import request_gpt_embedding


def get_embeddings(descriptions, request_fn):
    """Get the embedding for each description."""
    # return [request_fn(desc) for desc in descriptions]
    embeddings = [request_fn(desc) for desc in tqdm(descriptions, desc="Generating Embeddings")]
    return embeddings


def get_table_id_from_text(table):
    """Generate a unique ID from table_text content (via hash)."""
    table_str = json.dumps(table, sort_keys=True)
    return hashlib.sha1(table_str.encode('utf-8')).hexdigest()


def load_existing_table_ids(output_path):
    """Read the set of table_id values from an existing embeddings file."""
    existing_ids = set()
    if not os.path.exists(output_path):
        return existing_ids

    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                table_id = item.get("table_id")
                if table_id:
                    existing_ids.add(table_id)
            except:
                continue
    return existing_ids


def save_embeddings(index, line, col_prompt, seen_table_ids, lock,
                    col_stats_list=None, col_stats_lock=None):
    """Process a single table: generate descriptions (LLM) + compute embeddings.

    With dual-GPU (LLM on GPU 0, embeddings on GPU 1) these run on
    separate devices and can safely overlap across threads.

    If col_stats_list and col_stats_lock are provided, appends
    {"used_fallback": bool, "num_attempts": int} for col template generation.
    """
    try:
        item = json.loads(line)
        table = item["table_text"]
        statement = item["statement"]

        table_id = get_table_id_from_text(table)

        with lock:
            if table_id in seen_table_ids:
                return None
            seen_table_ids.add(table_id)

        cleaned_table = clean_table(table)

        row_descriptions = get_row_flattened(cleaned_table)

        def col_stats_callback(used_fallback, num_attempts):
            if col_stats_list is not None and col_stats_lock is not None:
                with col_stats_lock:
                    col_stats_list.append({
                        "used_fallback": used_fallback,
                        "num_attempts": num_attempts,
                    })

        col_descriptions = get_col_description(
            cleaned_table, col_prompt, stats_callback=col_stats_callback
        )

        row_embeddings = get_embeddings(row_descriptions, request_gpt_embedding)
        col_embeddings = get_embeddings(col_descriptions, request_gpt_embedding)

        return {
            "table_id": table_id,
            "statement": statement,
            "row_descriptions": row_descriptions,
            "row_embeddings": row_embeddings,
            "col_descriptions": col_descriptions,
            "col_embeddings": col_embeddings,
            "table_text": table,
        }
    except Exception as exc:
        print(f"save_embeddings: skipping index {index}: {exc}")
        return None


def _write_col_template_summary(col_stats_list, stats_output_path):
    """Write a JSON summary of col template stats (fallback count, attempt distribution)."""
    total = len(col_stats_list)
    if total == 0:
        summary = {
            "total_tables": 0,
            "col_template": {
                "used_fallback_count": 0,
                "used_fallback_fraction": 0.0,
                "attempts_distribution": {},
            },
        }
    else:
        fallback_count = sum(1 for s in col_stats_list if s["used_fallback"])
        attempts_dist = {}
        for s in col_stats_list:
            n = s["num_attempts"]
            attempts_dist[n] = attempts_dist.get(n, 0) + 1
        summary = {
            "total_tables": total,
            "col_template": {
                "used_fallback_count": fallback_count,
                "used_fallback_fraction": round(fallback_count / total, 4),
                "attempts_distribution": dict(sorted(attempts_dist.items())),
            },
        }
    os.makedirs(os.path.dirname(stats_output_path) or ".", exist_ok=True)
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Col template stats summary written to {stats_output_path}")


def process_table_embeddings(input_path, output_path, col_prompt_path, stats_output_path=None):
    """Parallel embedding precomputation using a thread pool.

    Each worker generates descriptions (LLM on GPU 0) and computes
    vector embeddings (embedding model on GPU 1) for one table.
    With dual-GPU isolation there is no contention.

    If stats_output_path is set, writes a JSON summary of col template
    statistics (how many tables used fallback, attempt distribution) after
    processing.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    with open(col_prompt_path, 'r', encoding='utf-8') as f:
        col_prompt = f.read()

    seen_table_ids = load_existing_table_ids(output_path)
    print(f"Loaded {len(seen_table_ids)} existing table_ids from {output_path}.")

    lock = threading.Lock()
    col_stats_list = [] if stats_output_path else None
    col_stats_lock = threading.Lock() if stats_output_path else None

    with open(output_path, 'a', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(
                    save_embeddings, idx, line, col_prompt, seen_table_ids, lock,
                    col_stats_list, col_stats_lock,
                )
                for idx, line in enumerate(data)
            ]

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Saving embeddings to {output_path}"):
                result = future.result()
                if result and result.get("table_id"):
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

    if stats_output_path and col_stats_list is not None:
        _write_col_template_summary(col_stats_list, stats_output_path)

    print(f"All new table embeddings appended to {output_path}")


def main():
    col_prompt_path = "prompt/get_col_template.md"

    configs = [
        ("dataset/wikitq/valid/1024-2048/1024-2048_sample.jsonl", "cache/table_embeddings_2k.jsonl"),
        ("dataset/wikitq/valid/2048-3072/2048-3072_sample.jsonl", "cache/table_embeddings_3k.jsonl"),
        ("dataset/wikitq/valid/3072-4096/3072-4096.jsonl", "cache/table_embeddings_4k.jsonl"),
    ]

    for input_path, output_path in configs:
        print(f"🚀 Processing {input_path}...")
        process_table_embeddings(input_path, output_path, col_prompt_path)


if __name__ == "__main__":
    main()
