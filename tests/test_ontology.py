"""Tests for OntologyMapper — written first per TDD."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from histological_image_analysis.ontology import OntologyMapper

REAL_ONTOLOGY_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "allen_brain_data"
    / "ontology"
    / "structure_graph_1.json"
)


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


    def test_depth_mapping_names_not_coarse(self, minimal_ontology_path):
        """Depth-1 mapping should NOT return coarse class names."""
        mapper = OntologyMapper(minimal_ontology_path)
        depth_mapping = mapper.build_depth_mapping(target_depth=1)
        names = mapper.get_class_names(depth_mapping)
        # Depth-1 ancestors include "Basic cell groups and regions" (id=8),
        # not "Cerebrum" — should not be misidentified as coarse
        assert "Cerebrum" not in names


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


class TestGetNumLabels:
    """Test num_labels helper for UperNet config."""

    def test_coarse_num_labels(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_coarse_mapping()
        assert mapper.get_num_labels(mapping) == 6  # classes 0-5

    def test_full_num_labels(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_full_mapping()
        # 15 structures → classes 1..15, plus background 0 → 16
        assert mapper.get_num_labels(mapping) == 16

    def test_empty_mapping(self, minimal_ontology_path):
        mapper = OntologyMapper(minimal_ontology_path)
        assert mapper.get_num_labels({}) == 0


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


@pytest.mark.skipif(
    not REAL_ONTOLOGY_PATH.exists(),
    reason="Real ontology file not present locally",
)
class TestRealOntologyCoarseMapping:
    """Diagnostic tests using the full structure_graph_1.json (1,327 structures).

    These tests verify the coarse mapping against the real ontology to
    diagnose the Cerebrum NaN IoU issue from the first training run.
    """

    @pytest.fixture
    def real_mapper(self) -> OntologyMapper:
        return OntologyMapper(REAL_ONTOLOGY_PATH)

    @pytest.fixture
    def real_coarse_mapping(self, real_mapper) -> dict[int, int]:
        return real_mapper.build_coarse_mapping()

    def test_total_structures_count(self, real_mapper):
        """The real ontology should have ~1,327 structures."""
        assert len(real_mapper.all_structure_ids) > 1300

    def test_coarse_mapping_covers_all_structures(
        self, real_mapper, real_coarse_mapping
    ):
        """Every structure ID should have a coarse class assignment."""
        assert set(real_coarse_mapping.keys()) == real_mapper.all_structure_ids

    def test_cerebrum_class_has_structures(self, real_coarse_mapping):
        """Class 1 (Cerebrum) should have hundreds of mapped structures."""
        class_counts = Counter(real_coarse_mapping.values())
        assert class_counts[1] > 600, (
            f"Expected >600 structures mapping to Cerebrum (class 1), "
            f"got {class_counts[1]}"
        )

    def test_all_coarse_classes_have_structures(self, real_coarse_mapping):
        """Every coarse class 0-5 should have at least one structure."""
        class_counts = Counter(real_coarse_mapping.values())
        for cls in range(6):
            assert class_counts[cls] > 0, (
                f"Class {cls} has no structures in coarse mapping"
            )

    def test_coarse_class_distribution(self, real_coarse_mapping):
        """Verify expected distribution: Cerebrum is the largest class."""
        class_counts = Counter(real_coarse_mapping.values())
        # Cerebrum should be the largest non-background class
        non_bg = {k: v for k, v in class_counts.items() if k != 0}
        largest_class = max(non_bg, key=non_bg.get)
        assert largest_class == 1, (
            f"Expected Cerebrum (1) to be largest class, "
            f"got class {largest_class} with {non_bg[largest_class]} structures"
        )

    def test_known_cerebrum_descendants_map_to_class_1(
        self, real_coarse_mapping
    ):
        """Spot-check that well-known cerebrum structures map correctly."""
        # These are real Allen Brain structure IDs under Cerebrum
        cerebrum_ids = {
            567: "Cerebrum",
            688: "Cerebral cortex",
            695: "Cortical plate",
            315: "Isocortex",
            184: "Frontal pole, cerebral cortex",
            453: "Cerebral nuclei",
            803: "Striatum",
        }
        for sid, name in cerebrum_ids.items():
            if sid in real_coarse_mapping:
                assert real_coarse_mapping[sid] == 1, (
                    f"{name} (ID {sid}) mapped to class "
                    f"{real_coarse_mapping[sid]}, expected 1 (Cerebrum)"
                )

    def test_print_class_distribution(self, real_coarse_mapping, real_mapper):
        """Print full class distribution for diagnostic review."""
        class_counts = Counter(real_coarse_mapping.values())
        names = real_mapper.get_class_names(real_coarse_mapping)
        print("\n=== Real Ontology Coarse Class Distribution ===")
        for cls in sorted(class_counts.keys()):
            name = names[cls] if cls < len(names) else "Unknown"
            print(f"  Class {cls} ({name}): {class_counts[cls]} structures")
        print(f"  Total: {sum(class_counts.values())} structures")


ANNOTATION_25_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "allen_brain_data"
    / "ccfv3"
    / "annotation_25.nrrd"
)


@pytest.mark.skipif(
    not REAL_ONTOLOGY_PATH.exists() or not ANNOTATION_25_PATH.exists(),
    reason="Real ontology or annotation_25.nrrd not present locally",
)
class TestRealAnnotationClassDistribution:
    """Diagnostic: check pixel-level class distribution in real annotation volume.

    Uses the 25μm annotation (85MB, faster to load) to verify that
    Cerebrum (class 1) pixels actually exist in the remapped volume.
    """

    @pytest.fixture(scope="class")
    def real_annotation_data(self):
        """Load 25μm annotation + ontology, build coarse mapping."""
        import nrrd

        mapper = OntologyMapper(REAL_ONTOLOGY_PATH)
        annotation, _ = nrrd.read(str(ANNOTATION_25_PATH))
        coarse_mapping = mapper.build_coarse_mapping()
        return annotation, mapper, coarse_mapping

    def test_annotation_volume_has_cerebrum_structure_ids(
        self, real_annotation_data
    ):
        """The raw annotation volume should contain structure IDs
        that map to Cerebrum (class 1)."""
        annotation, mapper, coarse_mapping = real_annotation_data
        unique_ids = np.unique(annotation)
        cerebrum_ids_in_volume = [
            sid for sid in unique_ids
            if coarse_mapping.get(int(sid), 0) == 1
        ]
        print(f"\nUnique structure IDs in annotation: {len(unique_ids)}")
        print(f"Cerebrum structure IDs in annotation: {len(cerebrum_ids_in_volume)}")
        assert len(cerebrum_ids_in_volume) > 0, (
            "No cerebrum structure IDs found in annotation volume"
        )

    def test_remapped_volume_has_all_coarse_classes(
        self, real_annotation_data
    ):
        """After remapping, all 6 coarse classes should have pixels."""
        annotation, mapper, coarse_mapping = real_annotation_data
        remapped = mapper.remap_mask(annotation, coarse_mapping)
        unique_classes = set(np.unique(remapped))
        print(f"\nUnique classes in remapped volume: {sorted(unique_classes)}")
        for cls in range(6):
            assert cls in unique_classes, (
                f"Class {cls} missing from remapped volume"
            )

    def test_cerebrum_pixel_fraction(self, real_annotation_data):
        """Cerebrum should be a significant fraction of non-background pixels."""
        annotation, mapper, coarse_mapping = real_annotation_data
        remapped = mapper.remap_mask(annotation, coarse_mapping)
        total = remapped.size
        per_class = {}
        for cls in range(6):
            count = int((remapped == cls).sum())
            per_class[cls] = count

        names = mapper.get_class_names(coarse_mapping)
        print("\n=== Pixel-level Class Distribution (25μm volume) ===")
        for cls in range(6):
            pct = per_class[cls] / total * 100
            print(f"  Class {cls} ({names[cls]}): {per_class[cls]:,} pixels ({pct:.1f}%)")
        print(f"  Total: {total:,} pixels")

        # Cerebrum should have > 0 pixels
        assert per_class[1] > 0, "Cerebrum has 0 pixels in remapped volume"
        # Cerebrum should be substantial (>5% of total volume)
        cerebrum_pct = per_class[1] / total * 100
        assert cerebrum_pct > 5, (
            f"Cerebrum is only {cerebrum_pct:.1f}% of volume — unexpectedly small"
        )

    def test_cerebrum_absent_from_posterior_splits(self, real_annotation_data):
        """DIAGNOSTIC: spatial split puts all cerebrum in train.

        The mouse brain cerebrum is concentrated in the anterior portion.
        An 80/10/10 contiguous spatial split along the AP axis puts the
        posterior 20% into val+test, which has no cerebrum tissue.
        THIS IS THE ROOT CAUSE of the Cerebrum NaN IoU.
        """
        annotation, mapper, coarse_mapping = real_annotation_data
        n_slices = annotation.shape[0]

        # Filter valid slices (>10% brain pixels, matching CCFv3Slicer logic)
        total_pixels = annotation.shape[1] * annotation.shape[2]
        threshold = total_pixels * 0.10
        valid = [
            ap for ap in range(n_slices)
            if np.count_nonzero(annotation[ap, :, :]) >= threshold
        ]

        # Spatial split (80/10/10) on valid slices
        n_train = int(len(valid) * 0.8)
        n_val = int(len(valid) * 0.1)

        splits = {
            "train": valid[:n_train],
            "val": valid[n_train:n_train + n_val],
            "test": valid[n_train + n_val:],
        }

        print(f"\nValid slices: {len(valid)} / {n_slices}")
        print(f"Split sizes: train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

        per_split_cerebrum = {}
        per_split_all_classes = {}
        for split_name, indices in splits.items():
            class_pixels = Counter()
            for ap in indices:
                remapped = mapper.remap_mask(
                    annotation[ap, :, :], coarse_mapping
                )
                for cls in range(6):
                    class_pixels[cls] += int((remapped == cls).sum())
            per_split_cerebrum[split_name] = class_pixels[1]
            per_split_all_classes[split_name] = dict(class_pixels)

        names = mapper.get_class_names(coarse_mapping)
        for split_name in ["train", "val", "test"]:
            print(f"\n  {split_name}:")
            for cls in range(6):
                count = per_split_all_classes[split_name].get(cls, 0)
                print(f"    Class {cls} ({names[cls]}): {count:,}")

        # This CONFIRMS the root cause: cerebrum absent from val and test
        assert per_split_cerebrum["train"] > 0, "Cerebrum should be in train"
        assert per_split_cerebrum["val"] == 0, (
            "Expected 0 cerebrum in val (posterior slices) — "
            "this confirms the spatial split is the root cause"
        )
        assert per_split_cerebrum["test"] == 0, (
            "Expected 0 cerebrum in test (most posterior slices) — "
            "confirms spatial split root cause"
        )

class TestGetClassNamesDepthMapping:
    """Test get_class_names() with depth and full mappings.

    The buggy implementation picks an arbitrary structure_id per class
    (depends on set iteration order). For depth-2, class 4 could be
    named "Cortical plate" instead of "Cerebrum". The fix prefers the
    shallowest-depth structure for each class.
    """

    def test_depth2_names_use_ancestor_not_descendant(
        self, minimal_ontology_path
    ):
        """Depth-2 class for Cerebrum should be named 'Cerebrum', not a descendant."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_depth_mapping(target_depth=2)
        names = mapper.get_class_names(mapping)
        # Cerebrum (567) is the depth-2 ancestor; descendants 688, 695 also
        # map to the same class. The name should be "Cerebrum" (depth 2),
        # NOT "Cerebral cortex" (depth 3) or "Cortical plate" (depth 4).
        assert names[mapping[567]] == "Cerebrum"

    def test_depth2_names_brainstem(self, minimal_ontology_path):
        """Depth-2 class for Brain stem should be named 'Brain stem', not 'Midbrain'."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_depth_mapping(target_depth=2)
        names = mapper.get_class_names(mapping)
        assert names[mapping[343]] == "Brain stem"

    def test_depth2_background_is_named(self, minimal_ontology_path):
        """Background class (0) should always be named 'Background'."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_depth_mapping(target_depth=2)
        names = mapper.get_class_names(mapping)
        assert names[0] == "Background"

    def test_full_mapping_each_structure_named(self, minimal_ontology_path):
        """In full mapping, each class gets its own structure's name."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_full_mapping()
        names = mapper.get_class_names(mapping)
        for sid, cid in mapping.items():
            expected_name = mapper.get_structure_name(sid)
            assert names[cid] == expected_name, (
                f"Class {cid} (sid={sid}) should be named '{expected_name}', "
                f"got '{names[cid]}'"
            )

    @pytest.mark.skipif(
        not REAL_ONTOLOGY_PATH.exists(),
        reason="Real ontology file not present locally",
    )
    def test_real_depth2_names(self):
        """Real ontology: depth-2 ancestors should have correct names."""
        mapper = OntologyMapper(REAL_ONTOLOGY_PATH)
        mapping = mapper.build_depth_mapping(target_depth=2)
        names = mapper.get_class_names(mapping)
        assert names[mapping[567]] == "Cerebrum"
        assert names[mapping[343]] == "Brain stem"
        assert names[mapping[512]] == "Cerebellum"


class TestDepthMappingIntegration:
    """Integration tests for depth-2 mapping class counts and coverage."""

    def test_depth2_class_count_minimal(self, minimal_ontology_path):
        """Minimal fixture: 6 depth-2 ancestors + background = 7 unique classes."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_depth_mapping(target_depth=2)
        unique_classes = set(mapping.values())
        assert len(unique_classes) == 7

    def test_depth2_get_num_labels(self, minimal_ontology_path):
        """get_num_labels should return 7 for minimal fixture depth-2 mapping."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_depth_mapping(target_depth=2)
        assert mapper.get_num_labels(mapping) == 7

    @pytest.mark.skipif(
        not REAL_ONTOLOGY_PATH.exists(),
        reason="Real ontology file not present locally",
    )
    def test_real_depth2_class_count(self):
        """Real ontology: 18 depth-2 structures + background = 19 total classes."""
        mapper = OntologyMapper(REAL_ONTOLOGY_PATH)
        mapping = mapper.build_depth_mapping(target_depth=2)
        assert mapper.get_num_labels(mapping) == 19

    def test_depth2_all_structures_mapped(self, minimal_ontology_path):
        """Every structure ID should have a mapping entry."""
        mapper = OntologyMapper(minimal_ontology_path)
        mapping = mapper.build_depth_mapping(target_depth=2)
        assert set(mapping.keys()) == mapper.all_structure_ids


@pytest.mark.skipif(
    not REAL_ONTOLOGY_PATH.exists() or not ANNOTATION_25_PATH.exists(),
    reason="Real ontology or annotation_25.nrrd not present locally",
)
class TestRealAnnotationInterleaved:
    """Validation that interleaved split fixes the Cerebrum NaN issue."""

    @pytest.fixture(scope="class")
    def real_annotation_data(self):
        """Load 25μm annotation + ontology, build coarse mapping."""
        import nrrd

        mapper = OntologyMapper(REAL_ONTOLOGY_PATH)
        annotation, _ = nrrd.read(str(ANNOTATION_25_PATH))
        coarse_mapping = mapper.build_coarse_mapping()
        return annotation, mapper, coarse_mapping

    def test_interleaved_split_has_cerebrum_in_all_splits(
        self, real_annotation_data
    ):
        """VALIDATION: interleaved split fixes the Cerebrum NaN issue.

        With interleaved splitting, every 10th slice goes to val/test,
        ensuring all brain regions (including anterior Cerebrum) appear
        in all splits.
        """
        from histological_image_analysis.ccfv3_slicer import CCFv3Slicer

        annotation, mapper, coarse_mapping = real_annotation_data

        # Use a synthetic image volume (we only care about annotation)
        image = np.ones_like(annotation, dtype=np.float32)
        slicer = CCFv3Slicer.from_arrays(image, annotation, mapper)

        splits = slicer.get_split_indices(split_strategy="interleaved")

        print(f"\nInterleaved split sizes: "
              f"train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])}")

        names = mapper.get_class_names(coarse_mapping)
        for split_name in ["train", "val", "test"]:
            class_pixels = Counter()
            for ap in splits[split_name]:
                remapped = mapper.remap_mask(
                    annotation[ap, :, :], coarse_mapping
                )
                for cls in range(6):
                    class_pixels[cls] += int((remapped == cls).sum())

            print(f"\n  {split_name} (interleaved):")
            for cls in range(6):
                count = class_pixels.get(cls, 0)
                print(f"    Class {cls} ({names[cls]}): {count:,}")

            # ALL classes should be present in ALL splits
            for cls in range(6):
                assert class_pixels.get(cls, 0) > 0, (
                    f"Class {cls} ({names[cls]}) missing from "
                    f"{split_name} split with interleaved strategy"
                )
