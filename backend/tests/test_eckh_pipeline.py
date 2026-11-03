"""Unit tests for ACNE-ECKH dataset cleaning pipeline and label mapping."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from backend.training.acne_eckh_pipeline import (
    HAYASHI_THRESHOLDS,
    AcneEckhInspectionPipeline,
    map_clinical_grade_to_severity,
    map_lesion_count_to_severity,
)


def test_hayashi_lesion_count_mapping():
    # Mild: <= 5 lesions
    assert map_lesion_count_to_severity(0) == 0
    assert map_lesion_count_to_severity(3) == 0
    assert map_lesion_count_to_severity(5) == 0

    # Moderate: 6 to 20 lesions
    assert map_lesion_count_to_severity(6) == 1
    assert map_lesion_count_to_severity(12) == 1
    assert map_lesion_count_to_severity(20) == 1

    # Severe: 21 to 50 lesions
    assert map_lesion_count_to_severity(21) == 2
    assert map_lesion_count_to_severity(35) == 2
    assert map_lesion_count_to_severity(50) == 2

    # Very Severe: > 50 lesions
    assert map_lesion_count_to_severity(51) == 3
    assert map_lesion_count_to_severity(100) == 3


def test_clinical_grade_string_mapping():
    assert map_clinical_grade_to_severity("mild") == 0
    assert map_clinical_grade_to_severity("Grade-1") == 0
    assert map_clinical_grade_to_severity("moderate") == 1
    assert map_clinical_grade_to_severity("Grade II") == 1
    assert map_clinical_grade_to_severity("severe") == 2
    assert map_clinical_grade_to_severity("Grade_3") == 2
    assert map_clinical_grade_to_severity("very_severe") == 3
    assert map_clinical_grade_to_severity("Grade 4") == 3
    assert map_clinical_grade_to_severity("unknown_category") is None


def test_eckh_pipeline_inspection_and_deduplication():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create sample folder structure
        mild_dir = tmp_path / "mild"
        mild_dir.mkdir()
        severe_dir = tmp_path / "severe"
        severe_dir.mkdir()

        # Create two unique synthetic images
        img1 = Image.new("RGB", (128, 128), color=(200, 150, 100))
        img1_path = mild_dir / "sample1.jpg"
        img1.save(img1_path)

        img2 = Image.new("RGB", (128, 128), color=(100, 150, 200))
        img2_path = severe_dir / "sample2.jpg"
        img2.save(img2_path)

        # Create an exact duplicate of img1
        img1_dup_path = mild_dir / "sample1_copy.jpg"
        img1.save(img1_dup_path)

        # Create a tiny unusable image (< 64x64)
        tiny_img = Image.new("RGB", (32, 32), color=(50, 50, 50))
        tiny_path = mild_dir / "tiny.jpg"
        tiny_img.save(tiny_path)

        # Create a corrupted non-image file
        corrupt_path = severe_dir / "corrupted.jpg"
        corrupt_path.write_bytes(b"not a valid jpeg file content")

        # Run pipeline
        pipeline = AcneEckhInspectionPipeline()
        report = pipeline.inspect_and_clean_directory(tmp_path)

        assert report["status"] == "audit_complete"
        assert report["valid_images"] == 3  # img1, img2, img1_copy
        assert report["corrupted_images"] == 1  # corrupted.jpg
        assert len(report["internal_duplicate_groups"]) == 1
        assert len(report["accepted_records"]) == 2  # deduplicated down to 2 unique images
        assert report["label_distribution"]["mild"] == 1
        assert report["label_distribution"]["severe"] == 1

