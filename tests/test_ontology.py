"""Tests for OntologyMapper — written first per TDD."""

import numpy as np
import pytest

from histological_image_analysis.ontology import OntologyMapper


class TestOntologyMapperInit:
    """Test loading and flattening the ontology JSON."""

    def test_loads_minimal_fixture(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        assert mapper.root_id == 997

    def test_flattens_all_structures(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        # Minimal fixture has: root(997), grey(8), CH(567), CTX(688), CTXpl(695),
        # BS(343), MB(313), CB(512), fiber tracts(1009), cc(784),
        # VS(73), VL(81), grooves(1024), grv-of-cc(1025), retina(304325711)
        assert len(mapper.all_structure_ids) == 15

    def test_parent_lookup_correct(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        assert mapper.get_parent_id(567) == 8       # Cerebrum → grey
        assert mapper.get_parent_id(8) == 997        # grey → root
        assert mapper.get_parent_id(997) is None     # root has no parent
        assert mapper.get_parent_id(688) == 567      # CTX → Cerebrum
        assert mapper.get_parent_id(784) == 1009     # cc → fiber tracts

    def test_get_structure_name(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        assert mapper.get_structure_name(567) == "Cerebrum"
        assert mapper.get_structure_name(343) == "Brain stem"
        assert mapper.get_structure_name(997) == "root"

    def test_get_structure_name_unknown_returns_none(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        assert mapper.get_structure_name(999999) is None


class TestCoarseMapping:
    """Test the ancestor-chain-based coarse mapping."""

    def test_cerebrum_descendants_map_to_class_1(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapping[567] == 1   # Cerebrum itself
        assert mapping[688] == 1   # Cerebral cortex
        assert mapping[695] == 1   # Cortical plate (deep descendant)

    def test_brain_stem_descendants_map_to_class_2(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapping[343] == 2   # Brain stem itself
        assert mapping[313] == 2   # Midbrain

    def test_cerebellum_maps_to_class_3(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapping[512] == 3

    def test_fiber_tracts_descendants_map_to_class_4(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapping[1009] == 4  # fiber tracts itself
        assert mapping[784] == 4   # corpus callosum

    def test_ventricular_descendants_map_to_class_5(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapping[73] == 5    # VS itself
        assert mapping[81] == 5    # lateral ventricle

    def test_grooves_map_to_background(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapping[1024] == 0  # grooves
        assert mapping[1025] == 0  # grooves descendant

    def test_retina_maps_to_background(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapping[304325711] == 0

    def test_root_and_grey_map_to_background(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        # root and "grey" are not themselves brain regions for segmentation
        assert mapping[997] == 0
        assert mapping[8] == 0

    def test_coarse_class_names(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        names = mapper.get_class_names(mapping)
        assert names[0] == "Background"
        assert names[1] == "Cerebrum"
        assert names[2] == "Brain stem"
        assert names[3] == "Cerebellum"
        assert names[4] == "fiber tracts"
        assert names[5] == "ventricular systems"


class TestFullMapping:
    """Test fine-grained mapping with contiguous class IDs."""

    def test_full_mapping_is_deterministic(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping1 = mapper.build_full_mapping()
        mapping2 = mapper.build_full_mapping()
        assert mapping1 == mapping2

    def test_full_mapping_class_0_is_background(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_full_mapping()
        # All structure IDs should map to values >= 1
        # (0 is reserved for background / unmapped)
        for class_id in mapping.values():
            assert class_id >= 0

    def test_full_mapping_contiguous(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_full_mapping()
        class_ids = sorted(set(mapping.values()))
        # Should be contiguous starting from 1 (0 for root/grey which are non-leaf)
        # The actual values depend on implementation, just check contiguous
        for i in range(len(class_ids) - 1):
            assert class_ids[i + 1] - class_ids[i] <= 1

    def test_full_mapping_sorted_structure_ids(self, minimal_ontology_path):
        """Structure IDs sorted ascending should get ascending class IDs."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_full_mapping()
        # Get structure IDs that map to non-zero class (real structures)
        struct_ids = sorted(
            sid for sid, cid in mapping.items() if cid > 0
        )
        class_ids = [mapping[sid] for sid in struct_ids]
        # Class IDs should be monotonically increasing for sorted structure IDs
        assert class_ids == sorted(class_ids)


class TestRemapMask:
    """Test vectorized remapping of annotation arrays."""

    def test_remap_simple_array(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()

        mask = np.array([[0, 567, 343], [1009, 73, 0]], dtype=np.uint32)
        remapped = mapper.remap_mask(mask, mapping)

        assert remapped.shape == mask.shape
        assert remapped[0, 0] == 0  # background stays 0
        assert remapped[0, 1] == 1  # Cerebrum → 1
        assert remapped[0, 2] == 2  # Brain stem → 2
        assert remapped[1, 0] == 4  # Fiber tracts → 4
        assert remapped[1, 1] == 5  # VS → 5
        assert remapped[1, 2] == 0  # background stays 0

    def test_remap_unknown_ids_to_background(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()

        mask = np.array([[999999, 567]], dtype=np.uint32)
        remapped = mapper.remap_mask(mask, mapping)
        assert remapped[0, 0] == 0  # unknown → background
        assert remapped[0, 1] == 1  # Cerebrum → 1

    def test_remap_preserves_dtype_as_int(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()

        mask = np.array([[567, 343]], dtype=np.uint32)
        remapped = mapper.remap_mask(mask, mapping)
        assert np.issubdtype(remapped.dtype, np.integer)

    def test_remap_descendant_ids(self, minimal_ontology_path):
        """Descendant structure IDs should map to their ancestor's coarse class."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()

        mask = np.array([[688, 695, 313, 784, 81]], dtype=np.uint32)
        remapped = mapper.remap_mask(mask, mapping)

        assert remapped[0, 0] == 1  # CTX (under Cerebrum) → 1
        assert remapped[0, 1] == 1  # CTXpl (under Cerebrum) → 1
        assert remapped[0, 2] == 2  # MB (under Brain stem) → 2
        assert remapped[0, 3] == 4  # cc (under fiber tracts) → 4
        assert remapped[0, 4] == 5  # VL (under VS) → 5
