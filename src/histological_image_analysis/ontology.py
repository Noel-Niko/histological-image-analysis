"""Structure ontology loader and mapper for Allen Brain Institute data.

Loads the structure_graph JSON, flattens the hierarchy, and provides
mappings from raw structure IDs to class IDs at various granularities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Coarse grouping: ancestor IDs → class IDs
_COARSE_ANCESTORS: dict[int, int] = {
    567: 1,   # Cerebrum
    343: 2,   # Brain stem
    512: 3,   # Cerebellum
    1009: 4,  # fiber tracts
    73: 5,    # ventricular systems
}

# Structures whose descendants should map to background (class 0)
_EXCLUDED_ANCESTORS: set[int] = {1024, 304325711}  # grooves, retina

_COARSE_CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Cerebrum",
    2: "Brain stem",
    3: "Cerebellum",
    4: "fiber tracts",
    5: "ventricular systems",
}


class OntologyMapper:
    """Load Allen Brain structure ontology and map structure IDs to class IDs.

    Parameters
    ----------
    ontology_path : str or Path
        Path to structure_graph_1.json (Allen Brain API format).
    """

    def __init__(self, ontology_path: str | Path) -> None:
        data = self._load_json(ontology_path)
        root_node = data["msg"][0]
        self._root_id = root_node["id"]

        # Flatten the tree into a list of nodes
        self._nodes: list[dict[str, Any]] = []
        self._flatten(root_node, depth=0)

        # Build lookups
        self._node_by_id: dict[int, dict[str, Any]] = {
            n["id"]: n for n in self._nodes
        }
        self._parent_lookup: dict[int, int | None] = {
            n["id"]: n.get("parent_structure_id") for n in self._nodes
        }

    @staticmethod
    def _load_json(path: str | Path) -> dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    def _flatten(self, node: dict[str, Any], depth: int) -> None:
        """Recursively flatten the ontology tree."""
        node_copy = {k: v for k, v in node.items() if k != "children"}
        node_copy["depth"] = depth
        self._nodes.append(node_copy)
        for child in node.get("children", []):
            self._flatten(child, depth + 1)

    @property
    def root_id(self) -> int:
        return self._root_id

    @property
    def all_structure_ids(self) -> set[int]:
        return set(self._node_by_id.keys())

    def get_parent_id(self, structure_id: int) -> int | None:
        return self._parent_lookup.get(structure_id)

    def get_structure_name(self, structure_id: int) -> str | None:
        node = self._node_by_id.get(structure_id)
        if node is None:
            return None
        return node["name"]

    def _find_coarse_class(self, structure_id: int) -> int:
        """Walk ancestor chain to find the coarse class for a structure."""
        current: int | None = structure_id
        while current is not None:
            if current in _COARSE_ANCESTORS:
                return _COARSE_ANCESTORS[current]
            if current in _EXCLUDED_ANCESTORS:
                return 0
            current = self._parent_lookup.get(current)
        return 0  # default: background

    def build_coarse_mapping(self) -> dict[int, int]:
        """Map every structure ID to one of 6 coarse classes (0-5).

        Uses ancestor-chain walking: for each structure, walks up the
        parent chain until it hits a known coarse ancestor (Cerebrum,
        Brain stem, etc.) or an excluded ancestor (grooves, retina).
        """
        return {
            sid: self._find_coarse_class(sid)
            for sid in self.all_structure_ids
        }

    def build_depth_mapping(self, target_depth: int) -> dict[int, int]:
        """Map every structure ID to its ancestor at the given depth.

        Assigns contiguous class IDs (sorted by structure ID).
        Structures whose ancestors don't reach target_depth map to 0.
        """
        # Find the ancestor at target_depth for each structure
        ancestor_at_depth: dict[int, int] = {}
        for sid in self.all_structure_ids:
            node = self._node_by_id[sid]
            if node["depth"] == target_depth:
                ancestor_at_depth[sid] = sid
            elif node["depth"] > target_depth:
                # Walk up to find ancestor at target_depth
                current = sid
                while current is not None:
                    cnode = self._node_by_id.get(current)
                    if cnode is not None and cnode["depth"] == target_depth:
                        ancestor_at_depth[sid] = current
                        break
                    current = self._parent_lookup.get(current)
            # depth < target_depth: maps to 0 (background)

        # Assign contiguous class IDs to unique ancestors (sorted)
        unique_ancestors = sorted(set(ancestor_at_depth.values()))
        ancestor_to_class = {a: i + 1 for i, a in enumerate(unique_ancestors)}

        return {
            sid: ancestor_to_class.get(ancestor_at_depth.get(sid, -1), 0)
            for sid in self.all_structure_ids
        }

    def build_full_mapping(self) -> dict[int, int]:
        """Map every structure ID to a contiguous class ID.

        Structure IDs are sorted before assignment to ensure determinism.
        Class 0 is reserved for background (structure ID 0 / unmapped).
        The root node (997) and intermediate grouping nodes (e.g., 8 for
        "grey") get their own class IDs like any other structure.
        """
        sorted_ids = sorted(self.all_structure_ids)
        return {sid: i + 1 for i, sid in enumerate(sorted_ids)}

    def build_present_mapping(
        self,
        full_mapping: dict[int, int],
        present_class_ids: set[int],
    ) -> dict[int, int]:
        """Build a filtered mapping containing only classes with training pixels.

        Takes the full mapping and a set of class IDs that have >0 training
        pixels. Structures whose full-mapping class is NOT in the present set
        are remapped to 0 (background). Present classes get contiguous new
        IDs starting from 1.

        Parameters
        ----------
        full_mapping : dict[int, int]
            Complete structure ID → class ID mapping from ``build_full_mapping()``.
        present_class_ids : set[int]
            Set of class IDs (from the full mapping) that have training pixels.
            Background (0) is always included implicitly.

        Returns
        -------
        dict[int, int]
            New mapping with contiguous class IDs for present classes only.
        """
        # Sort present class IDs (excluding background) for deterministic re-indexing
        sorted_present = sorted(present_class_ids - {0})
        old_to_new = {old_cid: i + 1 for i, old_cid in enumerate(sorted_present)}
        old_to_new[0] = 0  # background stays at 0

        return {
            sid: old_to_new.get(full_mapping[sid], 0)
            for sid in full_mapping
        }

    def get_class_names(self, mapping: dict[int, int]) -> list[str]:
        """Return a list of class names indexed by class ID.

        For coarse mappings, uses predefined names.
        For other mappings, uses structure names from the ontology.
        """
        max_class = max(mapping.values()) if mapping else 0
        names = ["Background"] * (max_class + 1)

        # Check if this is the coarse mapping by verifying anchor points
        is_coarse = all(
            mapping.get(sid) == cid
            for sid, cid in _COARSE_ANCESTORS.items()
        )
        if is_coarse:
            for class_id, name in _COARSE_CLASS_NAMES.items():
                if class_id <= max_class:
                    names[class_id] = name
        else:
            # Build reverse mapping: class_id → structure_id
            # Prefer shallowest-depth structure for naming (e.g., "Cerebrum"
            # at depth 2, not "Cortical plate" at depth 4).
            class_to_sid: dict[int, int] = {}
            for sid, cid in mapping.items():
                if cid == 0:
                    continue  # background already named above
                current = class_to_sid.get(cid)
                if current is None:
                    class_to_sid[cid] = sid
                else:
                    if self._node_by_id[sid]["depth"] < self._node_by_id[current]["depth"]:
                        class_to_sid[cid] = sid

            for cid, sid in class_to_sid.items():
                node = self._node_by_id.get(sid)
                if node is not None:
                    names[cid] = node["name"]

        return names

    @staticmethod
    def get_num_labels(mapping: dict[int, int]) -> int:
        """Return the number of labels (including background class 0).

        Use this for ``UperNetConfig(num_labels=...)`` in Step 9.
        For coarse mapping (classes 0-5), returns 6.
        For full mapping (classes 0-1327), returns 1328.
        """
        if not mapping:
            return 0
        return max(mapping.values()) + 1

    def remap_mask(
        self, mask: np.ndarray, mapping: dict[int, int]
    ) -> np.ndarray:
        """Remap an annotation mask using the given mapping.

        Structure IDs not in the mapping are mapped to 0 (background).
        Uses a lookup array for vectorized operation on large arrays.
        """
        # Build a lookup array covering all values in the mask
        max_val = int(mask.max())
        if max_val > 10_000_000:
            # For very large structure IDs, use dictionary-based approach
            flat = mask.ravel()
            result = np.zeros(flat.shape, dtype=np.int64)
            unique_vals = np.unique(flat)
            for val in unique_vals:
                class_id = mapping.get(int(val), 0)
                result[flat == val] = class_id
            return result.reshape(mask.shape)

        # For reasonable ID ranges, use a lookup array (much faster)
        lut = np.zeros(max_val + 1, dtype=np.int64)
        for sid, cid in mapping.items():
            if sid <= max_val:
                lut[sid] = cid
        return lut[mask]
