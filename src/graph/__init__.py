"""Knowledge graph building pipeline."""

from .graph_utils import normalize, normalize_relation_label, save_graph, load_graph, graph_stats
from .graph_core import build_graph
from .graph_merge import merge_graphs
from .entity_linking import link_similar_entities, link_entities_fast

__all__ = [
    # Utils
    "normalize",
    "normalize_relation_label",
    "save_graph",
    "load_graph",
    "graph_stats",
    # Core
    "build_graph",
    # Merge
    "merge_graphs",
    # Entity linking
    "link_similar_entities",
    "link_entities_fast",
]
