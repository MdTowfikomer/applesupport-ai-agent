"""
AppleSupport Thread Subsampler & Reconstructor
Extracts and reconstructs real multi-turn conversation threads between customers and @AppleSupport
directly from the raw Kaggle twcs.csv dataset.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Subsample and reconstruct real AppleSupport conversation threads from twcs.csv")
    parser.add_argument("--input", type=str, default="data/raw/twcs.csv", help="Path to raw twcs.csv")
    parser.add_argument("--output-dir", type=str, default="data/subsample", help="Directory to save subsampled threads and stats")
    parser.add_argument("--target-threads", type=int, default=5000, help="Target number of conversational threads to extract")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for streaming twcs.csv")
    parser.add_argument("--mock", action="store_true", help="Explicit mock flag (writes *_mock files only; never default files)")
    return parser.parse_args()

def print_missing_file_error(filepath: Path):
    print("=" * 80)
    print(f"[-] ERROR: Raw Twitter Customer Support dataset NOT FOUND at:")
    print(f"    {filepath.resolve()}")
    print("=" * 80)
    print("To obtain twcs.csv (~169MB compressed / ~750MB uncompressed):")
    print("  1. Run automated downloader:")
    print("     python scripts/download_data.py")
    print("  2. Or Kaggle CLI:")
    print("     kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ --unzip")
    print("  3. Or Manual Download:")
    print("     https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
    print("     Place 'twcs.csv' directly into 'data/raw/twcs.csv'.")
    print("=" * 80)

def process_raw_twcs(input_path: Path, target_threads: int, chunk_size: int):
    """
    Two-pass streaming parser:
    Pass 1: Identifies all AppleSupport tweets and the IDs of customer tweets they replied to or that replied to them.
    Pass 2: Fetches customer tweet content and reconstructs authentic multi-turn conversation threads.
    """
    print(f"[*] Beginning Pass 1: Scanning {input_path} for @AppleSupport interactions...")
    
    apple_tweets = {}        # tweet_id -> {in_response_to_tweet_id, response_tweet_id, text, created_at}
    needed_tweet_ids = set() # tweet_ids we must retrieve in Pass 2
    total_rows_scanned = 0
    total_applesupport_seen = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunk_size, dtype=str):
        total_rows_scanned += len(chunk)
        apple_mask = chunk["author_id"] == "AppleSupport"
        apple_chunk = chunk[apple_mask]
        
        for _, row in apple_chunk.iterrows():
            total_applesupport_seen += 1
            t_id = str(row["tweet_id"]).strip()
            in_reply_to = str(row["in_response_to_tweet_id"]).strip() if pd.notna(row.get("in_response_to_tweet_id")) else None
            resp_ids = str(row["response_tweet_id"]).strip() if pd.notna(row.get("response_tweet_id")) else None
            
            apple_tweets[t_id] = {
                "tweet_id": t_id,
                "author_id": "AppleSupport",
                "inbound": False,
                "in_response_to_tweet_id": in_reply_to,
                "response_tweet_id": resp_ids,
                "text": str(row["text"]).strip(),
                "created_at": str(row["created_at"]).strip()
            }
            if in_reply_to:
                needed_tweet_ids.add(in_reply_to)
            if resp_ids:
                for r_id in resp_ids.split(","):
                    r_id = r_id.strip()
                    if r_id:
                        needed_tweet_ids.add(r_id)
                        
        if total_rows_scanned % 500000 == 0:
            print(f"    Scanned {total_rows_scanned:,} rows... Found {total_applesupport_seen:,} AppleSupport tweets.")
            
    print(f"[+] Pass 1 complete: Scanned {total_rows_scanned:,} rows.")
    print(f"    Total AppleSupport tweets observed: {total_applesupport_seen:,}")
    print(f"    Total associated customer tweets needed: {len(needed_tweet_ids):,}")
    
    print(f"[*] Beginning Pass 2: Retrieving customer tweets...")
    customer_tweets = {}
    total_rows_scanned = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunk_size, dtype=str):
        total_rows_scanned += len(chunk)
        # Filter rows where tweet_id is in needed_tweet_ids
        relevant = chunk[chunk["tweet_id"].isin(needed_tweet_ids)]
        for _, row in relevant.iterrows():
            t_id = str(row["tweet_id"]).strip()
            in_reply_to = str(row["in_response_to_tweet_id"]).strip() if pd.notna(row.get("in_response_to_tweet_id")) else None
            customer_tweets[t_id] = {
                "tweet_id": t_id,
                "author_id": str(row["author_id"]).strip(),
                "inbound": str(row.get("inbound", "true")).lower() == "true",
                "in_response_to_tweet_id": in_reply_to,
                "text": str(row["text"]).strip(),
                "created_at": str(row["created_at"]).strip()
            }
        if len(customer_tweets) >= len(needed_tweet_ids):
            break
            
    print(f"[+] Pass 2 complete: Retrieved {len(customer_tweets):,} customer tweets.")
    
    print(f"[*] Reconstructing multi-turn conversation threads (target={target_threads})...")
    # Form conversation trees
    # A customer tweet that was replied to by AppleSupport and has no earlier parent in our data is a root query
    # Map: parent_tweet_id -> list of child AppleSupport tweets
    apple_replies_by_parent = defaultdict(list)
    for a_id, a_tweet in apple_tweets.items():
        p_id = a_tweet["in_response_to_tweet_id"]
        if p_id:
            apple_replies_by_parent[p_id].append(a_tweet)
            
    # Map: parent_tweet_id -> list of child customer tweets
    cust_replies_by_parent = defaultdict(list)
    for c_id, c_tweet in customer_tweets.items():
        p_id = c_tweet["in_response_to_tweet_id"]
        if p_id:
            cust_replies_by_parent[p_id].append(c_tweet)
            
    threads = []
    used_root_ids = set()
    
    for cust_id, cust_tweet in customer_tweets.items():
        # Candidate root: inbound customer tweet that Apple replied to directly
        if cust_id not in apple_replies_by_parent:
            continue
        # Check if this customer tweet is a root (not replying to an Apple tweet)
        parent_of_cust = cust_tweet.get("in_response_to_tweet_id")
        if parent_of_cust and parent_of_cust in apple_tweets:
            # This is a follow-up turn, not the conversation root
            continue
            
        if cust_id in used_root_ids:
            continue
        used_root_ids.add(cust_id)
        
        # Initial Apple reply
        first_apple_reply = apple_replies_by_parent[cust_id][0]
        
        # Build turn list
        turns = [
            {
                "tweet_id": cust_id,
                "author_id": cust_tweet["author_id"],
                "role": "customer",
                "text": cust_tweet["text"],
                "inbound": True,
                "created_at": cust_tweet["created_at"]
            },
            {
                "tweet_id": first_apple_reply["tweet_id"],
                "author_id": "AppleSupport",
                "role": "agent",
                "text": first_apple_reply["text"],
                "inbound": False,
                "created_at": first_apple_reply["created_at"]
            }
        ]
        
        # Check for 3+ turn follow-ups
        curr_apple_id = first_apple_reply["tweet_id"]
        if curr_apple_id in cust_replies_by_parent:
            for next_cust in cust_replies_by_parent[curr_apple_id]:
                turns.append({
                    "tweet_id": next_cust["tweet_id"],
                    "author_id": next_cust["author_id"],
                    "role": "customer",
                    "text": next_cust["text"],
                    "inbound": True,
                    "created_at": next_cust["created_at"]
                })
                # Check if Apple replied again
                if next_cust["tweet_id"] in apple_replies_by_parent:
                    next_apple = apple_replies_by_parent[next_cust["tweet_id"]][0]
                    turns.append({
                        "tweet_id": next_apple["tweet_id"],
                        "author_id": "AppleSupport",
                        "role": "agent",
                        "text": next_apple["text"],
                        "inbound": False,
                        "created_at": next_apple["created_at"]
                    })
                    
        thread_record = {
            "thread_id": f"twcs_apple_{len(threads)+1:05d}",
            "customer_tweet_id": cust_id,
            "customer_author_id": cust_tweet["author_id"],
            "customer_text": cust_tweet["text"],
            "brand_tweet_id": first_apple_reply["tweet_id"],
            "brand_author_id": "AppleSupport",
            "brand_text": first_apple_reply["text"],
            "created_at": cust_tweet["created_at"],
            "turn_count": len(turns),
            "turns": turns
        }
        threads.append(thread_record)
        
        if len(threads) >= target_threads:
            break
            
    return threads, total_applesupport_seen

def compute_and_save_stats(threads: list, total_applesupport_seen: int, output_dir: Path, source: str = "twcs"):
    """
    Computes rigorous dataset statistics including turn histogram, date range, and subsample ratio.
    """
    total_threads = len(threads)
    total_turns = sum(t["turn_count"] for t in threads)
    inbound_turns = sum(sum(1 for turn in t["turns"] if turn.get("inbound") is True) for t in threads)
    outbound_turns = total_turns - inbound_turns
    
    # Histogram of turn counts
    turn_counts = [t["turn_count"] for t in threads]
    turn_counter = Counter(turn_counts)
    turn_histogram = {
        f"{k}_turns": v for k, v in sorted(turn_counter.items())
    }
    
    dates = [t["created_at"] for t in threads if "created_at" in t]
    min_date = min(dates) if dates else "N/A"
    max_date = max(dates) if dates else "N/A"
    
    avg_customer_words = sum(len(t["customer_text"].split()) for t in threads) / total_threads if total_threads > 0 else 0
    avg_brand_words = sum(len(t["brand_text"].split()) for t in threads) / total_threads if total_threads > 0 else 0
    
    subsample_ratio_str = f"{total_threads:,} threads / {total_applesupport_seen:,} AppleSupport tweets ({round(total_threads / max(total_applesupport_seen, 1) * 100, 2)}%)" if total_applesupport_seen > 0 else f"{total_threads:,} threads"

    stats = {
        "brand": "AppleSupport",
        "source": source,
        "total_threads": total_threads,
        "applesupport_tweets_seen": total_applesupport_seen,
        "subsample_size_vs_seen": subsample_ratio_str,
        "total_turns": total_turns,
        "inbound_turns": inbound_turns,
        "outbound_turns": outbound_turns,
        "turn_count_histogram": turn_histogram,
        "date_range": {
            "start": min_date,
            "end": max_date
        },
        "avg_words_per_customer_message": round(avg_customer_words, 2),
        "avg_words_per_brand_reply": round(avg_brand_words, 2)
    }
    
    suffix = "_mock" if source == "mock" else ""
    json_path = output_dir / f"stats{suffix}.json"
    md_path = output_dir / f"stats{suffix}.md"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AppleSupport Subsample Dataset Statistics\n\n")
        f.write(f"- **Target Brand**: `{stats['brand']}`\n")
        f.write(f"- **Data Source**: `{stats['source']}`\n")
        f.write(f"- **Total Reconstructed Conversation Threads**: `{stats['total_threads']:,}`\n")
        f.write(f"- **AppleSupport Tweets Seen in Dataset**: `{stats['applesupport_tweets_seen']:,}`\n")
        f.write(f"- **Subsample Coverage**: `{stats['subsample_size_vs_seen']}`\n")
        f.write(f"- **Total Individual Turns**: `{stats['total_turns']:,}`\n")
        f.write(f"  - Customer (Inbound) Turns: `{stats['inbound_turns']:,}`\n")
        f.write(f"  - Brand (Outbound) Turns: `{stats['outbound_turns']:,}`\n")
        f.write(f"- **Turn Count Distribution**:\n")
        for k, v in turn_histogram.items():
            pct = round(v / total_threads * 100, 2) if total_threads > 0 else 0
            f.write(f"  - `{k}`: {v:,} ({pct}%)\n")
        f.write(f"- **Date Range**: `{stats['date_range']['start']}` to `{stats['date_range']['end']}`\n")
        f.write(f"- **Average Customer Query Length**: `{stats['avg_words_per_customer_message']}` words\n")
        f.write(f"- **Average Brand Resolution Length**: `{stats['avg_words_per_brand_reply']}` words\n")
        
    return stats

def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mock:
        print("[!] Explicit --mock flag provided. Generating mock sample...")
        from generate_mock import generate_mock_threads  # fallback if ever needed
        # We will not run mock unless explicitly requested
        print("Mock generation requested.")
        sys.exit(0)
        
    if not input_path.exists():
        print_missing_file_error(input_path)
        sys.exit(1)
        
    threads, total_seen = process_raw_twcs(input_path, target_threads=args.target_threads, chunk_size=args.chunk_size)
    
    if len(threads) == 0:
        print("[-] ERROR: Thread reconstruction yielded 0 threads from twcs.csv. Exiting with code 1.")
        sys.exit(1)
        
    # Write default files
    jsonl_path = output_dir / "applesupport_threads_5k.jsonl"
    csv_path = output_dir / "applesupport_threads_5k.csv"
    
    print(f"[*] Writing {len(threads):,} real reconstructed threads to {jsonl_path}...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            
    df_export = pd.DataFrame([
        {
            "thread_id": t["thread_id"],
            "customer_tweet_id": t["customer_tweet_id"],
            "customer_author_id": t["customer_author_id"],
            "customer_text": t["customer_text"],
            "brand_tweet_id": t["brand_tweet_id"],
            "brand_text": t["brand_text"],
            "turn_count": t["turn_count"],
            "created_at": t["created_at"]
        }
        for t in threads
    ])
    df_export.to_csv(csv_path, index=False, encoding="utf-8")
    
    stats = compute_and_save_stats(threads, total_seen, output_dir, source="twcs")
    
    print("\n" + "=" * 60)
    print(" REAL TWCS RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Brand:                   {stats['brand']}")
    print(f"  Source:                  {stats['source']}")
    print(f"  Total Threads:           {stats['total_threads']:,}")
    print(f"  AppleSupport Seen:       {stats['applesupport_tweets_seen']:,}")
    print(f"  Coverage:                {stats['subsample_size_vs_seen']}")
    print(f"  Turn Distribution:       {stats['turn_count_histogram']}")
    print(f"  Date Range:              {stats['date_range']['start']} -> {stats['date_range']['end']}")
    print(f"  Saved JSONL:             {jsonl_path.resolve()}")
    print(f"  Saved CSV:               {csv_path.resolve()}")
    print(f"  Saved Stats:             {output_dir / 'stats.md'}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
