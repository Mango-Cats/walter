"""Groups connected drug names into clusters.

Why this is needed:
If Drug A is confusable with Drug B, and Drug B is confusable with Drug C,
all three belong to the same 'confusion cluster'.

Keeping these clusters separate ensures that machine learning models can split
data fairly into training and testing sets, without the same cluster leaking
into both sets.
"""


class UnionFind:
    """Disjoint-set data structure for grouping connected drug names into clusters."""

    def __init__(self) -> None:
        """Initialize an empty disjoint-set tracker."""
        self._parent: dict[str, str] = {}

    def __contains__(self, x: str) -> bool:
        """Check whether an item is already tracked."""
        return x in self._parent

    def add(self, x: str) -> None:
        """Add an item as its own independent root if not already present.

        Args:
            x: Item key to track.

        """
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        """Find the canonical root representative for an item, applying path compression.

        Args:
            x: Item key to look up.

        Returns:
            The canonical root key for the item's cluster.

        """
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        """Merge the clusters containing two items.

        Args:
            a: First item key.
            b: Second item key.

        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def connected(self, a: str, b: str) -> bool:
        """Check whether two items belong to the same cluster.

        Args:
            a: First item key.
            b: Second item key.

        Returns:
            True if both items have been seen and share the same cluster root, False otherwise.

        """
        if a not in self._parent or b not in self._parent:
            return False
        return self.find(a) == self.find(b)


def build_components(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Group pairs of drug names into connected clusters.

    Args:
        pairs: List of drug name pairs (a, b) representing connections.

    Returns:
        Dictionary mapping each cluster root name to the set of member drug names.

    """
    uf = UnionFind()
    for a, b in pairs:
        uf.union(a, b)

    components: dict[str, set[str]] = {}
    for name in uf._parent:
        components.setdefault(uf.find(name), set()).add(name)
    return components
