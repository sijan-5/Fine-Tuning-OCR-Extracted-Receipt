"""Diagnose a mysterious CSV/TSV file."""
import sys
from pathlib import Path
from collections import Counter

def inspect(path: str):
    p = Path(path)
    print(f"=== File: {p} ===")
    print(f"Size: {p.stat().st_size:,} bytes")
    print()

    # --- Raw bytes ---
    with open(p, "rb") as f:
        raw = f.read()
    print(f"Total bytes read: {len(raw):,}")

    # --- BOM check ---
    boms = {
        "UTF-8 BOM":     b"\xef\xbb\xbf",
        "UTF-16 LE BOM": b"\xff\xfe",
        "UTF-16 BE BOM": b"\xfe\xff",
    }
    print("\n--- BOM at start of file ---")
    found_bom = False
    for name, sig in boms.items():
        if raw.startswith(sig):
            print(f"  FOUND: {name}")
            found_bom = True
    if not found_bom:
        print("  No BOM at start.")
        print(f"  First 8 bytes: {raw[:8]!r}")

    # --- Try encodings ---
    print("\n--- Encoding detection ---")
    candidates = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"]
    for enc in candidates:
        try:
            raw.decode(enc)
            print(f"  {enc:12s}: decodes cleanly")
        except UnicodeDecodeError as e:
            print(f"  {enc:12s}: FAILS at byte {e.start} ({raw[e.start:e.start+3]!r})")

    # --- Locate problem bytes if utf-8 fails ---
    print("\n--- Non-UTF-8 byte locations (first 10) ---")
    bad_positions = []
    i = 0
    while i < len(raw) and len(bad_positions) < 10:
        try:
            raw[i:i+1024].decode("utf-8")
            i += 1024
        except UnicodeDecodeError as e:
            abs_pos = i + e.start
            bad_positions.append(abs_pos)
            i = abs_pos + 1
    if bad_positions:
        for pos in bad_positions:
            byte = raw[pos]
            context_start = max(0, pos - 30)
            context_end = min(len(raw), pos + 30)
            context = raw[context_start:context_end]
            print(f"  byte {pos}: 0x{byte:02x} ({byte})  context: {context!r}")
    else:
        print("  None.")

    # --- First line (likely header) ---
    print("\n--- First line (decoded as latin-1 for safety) ---")
    first_line = raw.split(b"\n", 1)[0].decode("latin-1", errors="replace")
    print(f"  Length: {len(first_line)} chars")
    preview = first_line[:300] + ("..." if len(first_line) > 300 else "")
    print(f"  Content: {preview}")

    # --- Separator detection on header line ---
    print("\n--- Separator counts in header ---")
    for sep_name, sep_char in [("tab", "\t"), ("comma", ","),
                                ("semicolon", ";"), ("pipe", "|")]:
        count = first_line.count(sep_char)
        print(f"  {sep_name}: {count}")

    # --- Line endings (precomputed to avoid backslash-in-fstring issue) ---
    print("\n--- Line endings ---")
    crlf = raw.count(b"\r\n")
    all_lf = raw.count(b"\n")
    all_cr = raw.count(b"\r")
    lf_only = all_lf - crlf
    cr_only = all_cr - crlf
    print(f"  CRLF (Windows): {crlf}")
    print(f"  LF only (Unix): {lf_only}")
    print(f"  CR only (old Mac): {cr_only}")

    # --- Byte frequency of non-ASCII bytes ---
    print("\n--- Top 10 non-ASCII bytes ---")
    non_ascii = Counter(b for b in raw if b >= 0x80)
    for byte, count in non_ascii.most_common(10):
        print(f"  0x{byte:02x} ({byte}): {count} occurrences")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/Receipt_Dataset.csv"
    inspect(path)
