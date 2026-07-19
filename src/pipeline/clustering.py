"""
Union-find over drug-name pairs.

Downstream (the LASA classifier repo) splits train/test by connected
component of the name graph, so walter must never emit an edge that
bridges two components together unless that's intentional. This module
is the single place that answers "are these two names already linked,
directly or transitively?".
"""


class UnionFind:
    """Union-find over arbitrary hashable keys (drug names)."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def __contains__(self, x: str) -> bool:
        return x in self._parent

    def add(self, x: str) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def connected(self, a: str, b: str) -> bool:
        """False if either name is unseen, not just if they differ."""
        if a not in self._parent or b not in self._parent:
            return False
        return self.find(a) == self.find(b)


def build_components(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """
    Group names into connected components given a list of (a, b) edges.
    Returns {component_root: {member names}}.
    """
    uf = UnionFind()
    for a, b in pairs:
        uf.union(a, b)

    components: dict[str, set[str]] = {}
    for name in uf._parent:
        components.setdefault(uf.find(name), set()).add(name)
    return components
