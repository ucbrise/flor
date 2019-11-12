from .load_store_detector import LoadStoreDetector
from .method_change_detector import MethodChangeDetector
from flor.transformer.utils import set_union


def get_change_and_read_set(node):
    lsd = LoadStoreDetector()
    mcd = MethodChangeDetector()
    lsd.visit(node)
    mcd.visit(node)
    return lsd.writes, mcd.mutated_objects, lsd.unmatched_reads


__all__ = ['get_change_and_read_set', 'LoadStoreDetector', 'MethodChangeDetector']
