"""
Groups R_ph_clean.csv drug names whose pairwise (case-insensitive) Levenshtein
distance is at most MAX_DIST, to surface likely misspelling clusters.

Which channel(s) are actually used is controlled by MODE:
  "lev2"  - Levenshtein distance <= MAX_DIST (+ MAX_RATIO) only.
  "key2"  - strict keyboard distance <= MAX_DIST only.
  "both"  - either channel qualifies (the default; a pair need only pass one).

Usage: python scripts/find_typos.py [lev2|key2|both]
"""

import csv
import re
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

MAX_DIST = 2
MAX_RATIO = 0.3
EXCLUDE_SPACES = True
MODE = "both"  # "lev2" | "key2" | "both" - overridable via argv[1]
VALID_MODES = {"lev2", "key2", "both"}
INPUT = Path("data/R_ph_clean.csv")
LARGE_COMPONENT_WARN = 60
_DIGITS = re.compile(r"\d")

_QWERTY_ROWS = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
_ROW_OFFSETS = [0.0, 0.5, 0.75, 1.0]  # approximate QWERTY row stagger
_ADJACENT_THRESHOLD = 1.3


def _keyboard_positions() -> dict[str, tuple[float, int]]:
    pos = {}
    for r, row in enumerate(_QWERTY_ROWS):
        for c, ch in enumerate(row):
            pos[ch] = (c + _ROW_OFFSETS[r], r)
    return pos


_KB_POS = _keyboard_positions()


def _kb_adjacent(a: str, b: str) -> bool:
    pa, pb = _KB_POS.get(a), _KB_POS.get(b)
    if pa is None or pb is None:
        return False
    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
    return dx * dx + dy * dy <= _ADJACENT_THRESHOLD**2


def keyboard_distance(a: str, b: str, max_dist: int) -> int:
    """
    Strict keyboard distance: only defined for equal-length names (no
    insertions/deletions - this channel is specifically for fat-finger
    substitution typos). Counts differing positions; each one must be a
    keyboard-adjacent key or the pair is disqualified outright, returned
    as max_dist + 1 same as "too far".
    """
    if len(a) != len(b):
        return max_dist + 1
    cost = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        if not _kb_adjacent(ca, cb):
            return max_dist + 1
        cost += 1
        if cost > max_dist:
            return max_dist + 1
    return cost


def load_names(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        seen = set()
        names = []
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if not name or name in seen:
                continue
            if EXCLUDE_SPACES and " " in name:
                continue
            seen.add(name)
            names.append(name)
    return names


def build_edges(names: list[str], mode: str = MODE) -> list[tuple[int, int]]:
    """
    Edge (i, j) if, depending on `mode`:
      "lev2": case-insensitive Levenshtein distance <= MAX_DIST AND
              distance / min(len(i), len(j)) <= MAX_RATIO.
      "key2": strict keyboard distance <= MAX_DIST (see module docstring).
      "both": either of the above qualifies.
    Rejected regardless of mode if the two names are identical once
    digits are stripped (a strength/version suffix, not a typo).

    Names are processed in ascending-length order so each name only needs
    to look forward to names at most MAX_DIST characters longer - that
    covers every unordered pair exactly once without an O(n^2) length
    pre-filter.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)} (got {mode!r})")
    use_lev = mode in ("lev2", "both")
    use_key = mode in ("key2", "both")

    lowered = [n.lower() for n in names]
    stripped = [_DIGITS.sub("", s) for s in lowered]
    n = len(names)
    order = sorted(range(n), key=lambda i: len(lowered[i]))
    lengths = [len(lowered[i]) for i in order]

    edges = []
    for a_pos in range(n):
        i = order[a_pos]
        li = lengths[a_pos]
        b_pos = a_pos + 1
        while b_pos < n and lengths[b_pos] - li <= MAX_DIST:
            j = order[b_pos]
            lj = lengths[b_pos]
            if lowered[i] != lowered[j] and stripped[i] == stripped[j]:
                b_pos += 1
                continue
            d = Levenshtein.distance(lowered[i], lowered[j], score_cutoff=MAX_DIST)
            if d <= MAX_DIST:
                if use_lev and d <= MAX_RATIO * min(li, lj):
                    edges.append((i, j))
                elif use_key and keyboard_distance(lowered[i], lowered[j], MAX_DIST) <= MAX_DIST:
                    edges.append((i, j))
            b_pos += 1
        if a_pos % 2000 == 0:
            print(f"  ...{a_pos:,}/{n:,} names scanned", file=sys.stderr)
    return edges


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    groups: dict[int, set[int]] = {}
    touched = {v for edge in edges for v in edge}
    for v in touched:
        groups.setdefault(find(v), set()).add(v)
    return list(groups.values())


def bron_kerbosch(
    adj: dict[int, set[int]],
) -> list[set[int]]:
    """Standard Bron-Kerbosch with pivoting; returns all maximal cliques."""
    cliques: list[set[int]] = []

    def extend(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            cliques.append(r)
            return
        pivot = max(p | x, key=lambda u: len(adj[u] & p), default=None)
        candidates = p - adj.get(pivot, set()) if pivot is not None else set(p)
        for v in list(candidates):
            extend(r | {v}, p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    extend(set(), set(adj.keys()), set())
    return cliques


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    if mode not in VALID_MODES:
        sys.exit(f"mode must be one of {sorted(VALID_MODES)} (got {mode!r})")
    output = Path(f"results/R_ph_{mode}_groups.csv")

    names = load_names(INPUT)
    print(f"[1/4] Loaded {len(names):,} unique drug names (mode={mode})")

    print("[2/4] Computing pairwise distances...")
    edges = build_edges(names, mode)
    print(f"      {len(edges):,} edges with distance <= {MAX_DIST}")

    print("[3/4] Finding connected components and maximal cliques...")
    components = connected_components(len(names), edges)
    print(f"      {len(components):,} connected components")

    edge_set = set()
    for a, b in edges:
        edge_set.add((a, b))
        edge_set.add((b, a))

    all_cliques: list[set[int]] = []
    for comp in components:
        if len(comp) == 2:
            all_cliques.append(comp)
            continue
        if len(comp) > LARGE_COMPONENT_WARN:
            print(
                f"      NOTE: large component of {len(comp)} names "
                f"(clique search may take a moment): "
                f"{sorted(names[i] for i in comp)[:5]}...",
                file=sys.stderr,
            )
        adj = {v: {u for u in comp if u != v and (v, u) in edge_set} for v in comp}
        all_cliques.extend(bron_kerbosch(adj))

    # De-duplicate, sort rows by descending size then alphabetically, sort names within a row
    seen_rows = set()
    rows = []
    for clique in all_cliques:
        row = tuple(sorted(names[i] for i in clique))
        if row in seen_rows:
            continue
        seen_rows.add(row)
        rows.append(row)
    rows.sort(key=lambda r: (-len(r), r))

    print(f"[4/4] Writing {len(rows):,} groups to {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    involved = {n for row in rows for n in row}
    print(f"\nDone. {len(rows):,} groups covering {len(involved):,} distinct names "
          f"(of {len(names):,} total).")


if __name__ == "__main__":
    main()
