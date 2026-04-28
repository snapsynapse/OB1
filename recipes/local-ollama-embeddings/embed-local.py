#!/usr/bin/env python3
"""
Open Brain — Local Ollama Embeddings

Generates embeddings locally via Ollama and inserts thoughts into Supabase.
No OpenRouter or cloud API key required for the embedding step.

Usage:
    echo "My thought" | python embed-local.py
    python embed-local.py "My thought as an argument"
    python embed-local.py --file thoughts.txt
    python embed-local.py --file thoughts.jsonl

Input formats:
    Plain text:    One thought per line (blank lines skipped)
    JSONL:         One JSON object per line with "content" key
                   Optional keys: "source", "metadata" (merged into metadata)

Options:
    --file FILE            Read thoughts from a file (.txt or .jsonl)
    --model NAME           Ollama embedding model (default: nomic-embed-text)
    --ollama-url URL       Ollama base URL (default: http://localhost:11434)
    --source LABEL         Source label for metadata (default: ollama-local)
    --dry-run              Generate embeddings but don't insert into Supabase
    --verbose              Print each thought and embedding dimension
    --batch-size N         Thoughts per Ollama embed request (default: 1)
    --upsert               Upsert on content_fingerprint conflict instead of plain INSERT
                           Requires a UNIQUE constraint on content_fingerprint column.
                           Only applies when content_fingerprint is present in JSONL input.
    --reembed              Fetch all existing rows and re-embed them (update embedding only).
                           Does not change content or metadata. Useful after model change.
    --reembed-limit N      Cap rows to re-embed (default: all)

Environment variables:
    SUPABASE_URL               Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY  Supabase service role key
    OLLAMA_BASE_URL            Ollama base URL (overridden by --ollama-url)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("Error: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_MODEL = "nomic-embed-text"

# Expected embedding dimensions per model (for validation)
KNOWN_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16": 1536,
}


# ─── HTTP Helpers ────────────────────────────────────────────────────────────


def http_post(url, headers, body, timeout=60):
    """POST JSON with basic retry (2 attempts)."""
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            return resp
        except requests.RequestException as e:
            if attempt == 0:
                time.sleep(1)
                continue
            print(f"   Warning: Request failed after 2 attempts: {e}")
            return None
    return None


# ─── Ollama Embedding ────────────────────────────────────────────────────────


def generate_embedding(text, model, ollama_url):
    """Generate an embedding via Ollama's /api/embed endpoint."""
    truncated = text[:8000]

    resp = http_post(
        f"{ollama_url}/api/embed",
        headers={"Content-Type": "application/json"},
        body={
            "model": model,
            "input": truncated,
        },
        timeout=120,
    )

    if not resp or resp.status_code != 200:
        status = resp.status_code if resp else "no response"
        print(f"   Warning: Ollama embedding failed ({status})")
        if resp:
            try:
                print(f"   Detail: {resp.json()}")
            except ValueError:
                print(f"   Detail: {resp.text[:200]}")
        return None

    try:
        data = resp.json()
        embedding = data["embeddings"][0]
        return embedding
    except (KeyError, IndexError) as e:
        print(f"   Warning: Failed to parse Ollama response: {e}")
        return None


# ─── Supabase Ingestion ──────────────────────────────────────────────────────


def _supa_headers(prefer="return=minimal"):
    return {
        "Content-Type": "application/json",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Prefer": prefer,
    }


def ingest_thought(content, embedding, metadata_dict, content_fingerprint=None, upsert=False):
    """Insert (or upsert) a thought into Supabase with the provided embedding.

    If content_fingerprint is provided and upsert=True, uses ON CONFLICT resolution.
    Requires a UNIQUE constraint on content_fingerprint: see schema setup docs.
    """
    body = {
        "content": content,
        "embedding": embedding,
        "metadata": metadata_dict,
    }
    if content_fingerprint:
        body["content_fingerprint"] = content_fingerprint

    if upsert and content_fingerprint:
        url = f"{SUPABASE_URL}/rest/v1/thoughts?on_conflict=content_fingerprint"
        prefer = "resolution=merge-duplicates,return=minimal"
    else:
        url = f"{SUPABASE_URL}/rest/v1/thoughts"
        prefer = "return=minimal"

    resp = http_post(url, headers=_supa_headers(prefer), body=body)

    if not resp:
        return {"ok": False, "error": "No response from Supabase"}

    if resp.status_code not in (200, 201, 204):
        try:
            error_detail = resp.json()
        except ValueError:
            error_detail = resp.text
        return {"ok": False, "error": f"HTTP {resp.status_code}: {error_detail}"}

    return {"ok": True}


def update_embedding(row_id, embedding):
    """PATCH embedding column for one existing row."""
    resp = http_post(
        f"{SUPABASE_URL}/rest/v1/thoughts?id=eq.{row_id}",
        headers=_supa_headers(),
        body={"embedding": embedding},
    )
    if not resp:
        return {"ok": False, "error": "No response from Supabase"}
    if resp.status_code not in (200, 204):
        try:
            error_detail = resp.json()
        except ValueError:
            error_detail = resp.text
        return {"ok": False, "error": f"HTTP {resp.status_code}: {error_detail}"}
    return {"ok": True}


def fetch_all_rows():
    """GET all thoughts (id + content). Used by --reembed."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/thoughts?select=id,content&order=created_at.asc&limit=10000",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Error fetching rows: {e}")
        return []


# ─── Input Parsing ────────────────────────────────────────────────────────────


def read_thoughts_from_file(filepath):
    """Read thoughts from a .txt or .jsonl file. Returns list of dicts."""
    thoughts = []

    if filepath.endswith(".jsonl"):
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    content = obj.get("content", "").strip()
                    if not content:
                        print(f"   Warning: Line {line_num} missing 'content' key, skipped")
                        continue
                    thoughts.append({
                        "content": content,
                        "source": obj.get("source"),
                        "metadata": obj.get("metadata"),
                        "content_fingerprint": obj.get("content_fingerprint"),
                    })
                except json.JSONDecodeError as e:
                    print(f"   Warning: Line {line_num} invalid JSON ({e}), skipped")
    else:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    thoughts.append({"content": line})

    return thoughts


def read_thoughts_from_stdin():
    """Read thoughts from stdin, one per line."""
    thoughts = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            thoughts.append({"content": line})
    return thoughts


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate local embeddings via Ollama and insert into Open Brain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  echo "My important thought" | python embed-local.py
  python embed-local.py "A thought passed as an argument"
  python embed-local.py --file notes.txt
  python embed-local.py --file thoughts.jsonl --model nomic-embed-text --dry-run
  python embed-local.py --file thoughts.jsonl --upsert
  python embed-local.py --reembed --model mxbai-embed-large
  python embed-local.py --reembed --reembed-limit 100 --dry-run
  cat notes.txt | python embed-local.py --source journal --verbose""",
    )
    parser.add_argument("text", nargs="*", help="Thought(s) to embed (positional arguments)")
    parser.add_argument("--file", type=str, help="Read thoughts from a file (.txt or .jsonl)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", type=str, default=None, help=f"Ollama base URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--source", type=str, default="ollama-local", help="Source label for metadata (default: ollama-local)")
    parser.add_argument("--dry-run", action="store_true", help="Generate embeddings but don't insert into Supabase")
    parser.add_argument("--verbose", action="store_true", help="Print each thought and embedding info")
    parser.add_argument("--batch-size", type=int, default=1, help="Thoughts per request (default: 1)")
    parser.add_argument("--upsert", action="store_true",
                        help="Upsert on content_fingerprint conflict (requires UNIQUE constraint on column)")
    parser.add_argument("--reembed", action="store_true",
                        help="Re-embed all existing rows and update their embedding column (no content change)")
    parser.add_argument("--reembed-limit", type=int, default=None,
                        help="Cap number of rows to re-embed (default: all)")
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    ollama_url = args.ollama_url or OLLAMA_BASE_URL

    # Validate environment
    if not args.dry_run:
        if not SUPABASE_URL:
            print("Error: SUPABASE_URL environment variable required.")
            print("Set it to your Supabase project URL (e.g., https://xxxxx.supabase.co)")
            sys.exit(1)
        if not SUPABASE_SERVICE_ROLE_KEY:
            print("Error: SUPABASE_SERVICE_ROLE_KEY environment variable required.")
            sys.exit(1)

    # Collect thoughts from all input sources (skipped for --reembed)
    thoughts = []

    if not args.reembed:
        if args.file:
            if not os.path.isfile(args.file):
                print(f"Error: File not found: {args.file}")
                sys.exit(1)
            thoughts = read_thoughts_from_file(args.file)
        elif args.text:
            thoughts = [{"content": t} for t in args.text]
        elif not sys.stdin.isatty():
            thoughts = read_thoughts_from_stdin()
        else:
            print("Error: No input provided.")
            print("Usage: echo 'text' | python embed-local.py")
            print("       python embed-local.py 'text'")
            print("       python embed-local.py --file notes.txt")
            sys.exit(1)

        if not thoughts:
            print("No thoughts to process.")
            sys.exit(0)

    # Check Ollama connectivity
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"Warning: Ollama returned {resp.status_code} — is it running at {ollama_url}?")
    except requests.RequestException:
        print(f"Error: Cannot reach Ollama at {ollama_url}")
        print("Start Ollama with: ollama serve")
        print(f"Then pull the model: ollama pull {args.model}")
        sys.exit(1)

    # Display run configuration
    mode = "DRY RUN" if args.dry_run else "LIVE"
    dim_info = f" ({KNOWN_DIMENSIONS[args.model]}-dim)" if args.model in KNOWN_DIMENSIONS else ""
    print(f"\nOpen Brain — Local Ollama Embeddings")
    print(f"  Mode:     {mode}")
    print(f"  Model:    {args.model}{dim_info}")
    print(f"  Ollama:   {ollama_url}")
    if args.reembed:
        limit_str = str(args.reembed_limit) if args.reembed_limit else "all"
        print(f"  Action:   re-embed existing rows (limit: {limit_str})")
    else:
        print(f"  Source:   {args.source}")
        print(f"  Thoughts: {len(thoughts)}")
    if args.model in KNOWN_DIMENSIONS and KNOWN_DIMENSIONS[args.model] != 1536:
        print(f"\n  NOTE: {args.model} produces {KNOWN_DIMENSIONS[args.model]}-dim embeddings.")
        print(f"  The default Open Brain schema uses vector(1536).")
        print(f"  Adjust your schema to match: ALTER TABLE thoughts ALTER COLUMN embedding TYPE vector({KNOWN_DIMENSIONS[args.model]});")
    print()

    # --reembed: fetch existing rows, re-embed, patch embedding column
    if args.reembed:
        print(f"\nFetching existing rows...")
        rows = fetch_all_rows()
        if args.reembed_limit:
            rows = rows[: args.reembed_limit]
        print(f"  Re-embedding {len(rows)} rows\n")

        embedded = 0
        errors = 0

        for i, row in enumerate(rows, 1):
            row_id = row["id"]
            content = row["content"]
            preview = content if len(content) <= 80 else content[:77] + "..."
            print(f"{i}/{len(rows)}: {preview}")

            embedding = generate_embedding(content, args.model, ollama_url)
            if not embedding:
                errors += 1
                print(f"   -> FAILED to generate embedding")
                continue

            embedded += 1
            if args.verbose:
                print(f"   -> Embedding: {len(embedding)}-dim")

            if args.dry_run:
                print(f"   -> OK (dry run)")
                continue

            result = update_embedding(row_id, embedding)
            if result.get("ok"):
                print(f"   -> Updated")
            else:
                errors += 1
                print(f"   -> ERROR: {result.get('error', 'unknown')}")

            time.sleep(0.1)

        print()
        print("-" * 50)
        print("Summary (re-embed):")
        print(f"  Rows:      {len(rows)}")
        print(f"  Embedded:  {embedded}")
        if not args.dry_run:
            print(f"  Updated:   {embedded - errors}")
        print(f"  Errors:    {errors}")
        print(f"  API cost:  $0.00 (local)")
        print("-" * 50)
        return

    # Process
    embedded = 0
    ingested = 0
    errors = 0

    for i, thought in enumerate(thoughts, 1):
        content = thought["content"]
        preview = content if len(content) <= 80 else content[:77] + "..."
        print(f"{i}/{len(thoughts)}: {preview}")

        # Generate embedding
        embedding = generate_embedding(content, args.model, ollama_url)
        if not embedding:
            errors += 1
            print(f"   -> FAILED to generate embedding")
            continue

        embedded += 1
        if args.verbose:
            print(f"   -> Embedding: {len(embedding)}-dim")

        if args.dry_run:
            print(f"   -> OK (dry run)")
            continue

        # Build metadata
        metadata = {
            "source": thought.get("source") or args.source,
            "embedding_model": args.model,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
        }
        if thought.get("metadata") and isinstance(thought["metadata"], dict):
            metadata.update(thought["metadata"])

        # Ingest
        result = ingest_thought(
            content,
            embedding,
            metadata,
            content_fingerprint=thought.get("content_fingerprint"),
            upsert=args.upsert,
        )
        if result.get("ok"):
            ingested += 1
            print(f"   -> Ingested")
        else:
            errors += 1
            print(f"   -> ERROR: {result.get('error', 'unknown')}")

        time.sleep(0.1)  # Gentle rate limit

    # Summary
    print()
    print("-" * 50)
    print("Summary:")
    print(f"  Input:     {len(thoughts)} thoughts")
    print(f"  Embedded:  {embedded}")
    if not args.dry_run:
        print(f"  Ingested:  {ingested}")
    print(f"  Errors:    {errors}")
    print(f"  API cost:  $0.00 (local)")
    print("-" * 50)


if __name__ == "__main__":
    main()
