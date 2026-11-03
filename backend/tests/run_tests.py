from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
# Also support root parent for Skinora.backend package imports
root_parent = root.parent
if str(root_parent) not in sys.path:
    sys.path.insert(0, str(root_parent))

"""Standalone test runner for Skinora backend test suite."""
import traceback

print("=" * 60)
print("RUNNING SKINORA BACKEND TEST SUITE")
print("=" * 60)

passed = 0
failed = 0

# 1. Test model
try:
    from Skinora.backend.tests.test_model import (
        test_build_model_frozen,
        test_build_model_partial_layer4,
        test_parameter_groups,
        test_checkpoint_roundtrip,
    )
except ImportError:
    from backend.tests.test_model import (
        test_build_model_frozen,
        test_build_model_partial_layer4,
        test_parameter_groups,
        test_checkpoint_roundtrip,
    )

model_tests = [
    ("test_build_model_frozen", test_build_model_frozen),
    ("test_build_model_partial_layer4", test_build_model_partial_layer4),
    ("test_parameter_groups", test_parameter_groups),
    ("test_checkpoint_roundtrip", test_checkpoint_roundtrip),
]

for name, test_fn in model_tests:
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

# 2. Test evaluation
try:
    from Skinora.backend.tests.test_evaluation import (
        test_compute_ece,
        test_plot_confusion_matrix,
        test_plot_training_curves,
    )
except ImportError:
    from backend.tests.test_evaluation import (
        test_compute_ece,
        test_plot_confusion_matrix,
        test_plot_training_curves,
    )

eval_tests = [
    ("test_compute_ece", test_compute_ece),
    ("test_plot_confusion_matrix", test_plot_confusion_matrix),
    ("test_plot_training_curves", test_plot_training_curves),
]

for name, test_fn in eval_tests:
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

# 3. Test API
try:
    from Skinora.backend.tests.test_api import (
        test_root_endpoint,
        test_health_endpoint,
        test_model_status_endpoint,
        test_model_evaluation_endpoint,
        test_analyze_questionnaire_only,
        test_analyze_with_synthetic_photo,
        test_analyze_invalid_file,
        test_assessments_history,
    )
except ImportError:
    from backend.tests.test_api import (
        test_root_endpoint,
        test_health_endpoint,
        test_model_status_endpoint,
        test_model_evaluation_endpoint,
        test_analyze_questionnaire_only,
        test_analyze_with_synthetic_photo,
        test_analyze_invalid_file,
        test_assessments_history,
    )

api_tests = [
    ("test_root_endpoint", test_root_endpoint),
    ("test_health_endpoint", test_health_endpoint),
    ("test_model_status_endpoint", test_model_status_endpoint),
    ("test_model_evaluation_endpoint", test_model_evaluation_endpoint),
    ("test_analyze_questionnaire_only", test_analyze_questionnaire_only),
    ("test_analyze_with_synthetic_photo", test_analyze_with_synthetic_photo),
    ("test_analyze_invalid_file", test_analyze_invalid_file),
    ("test_assessments_history", test_assessments_history),
]

for name, test_fn in api_tests:
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

# 4. Test Dataset Integrity & Split Leakage
try:
    from Skinora.backend.tests.test_dataset_integrity import (
        test_clean_splits_manifest_structure,
        test_zero_hash_leakage,
        test_duplicate_clusters_isolated,
        test_contradictory_labels_quarantined_to_train,
        test_production_checkpoint_preserved,
        test_class_id_mapping_stability,
    )
except ImportError:
    from backend.tests.test_dataset_integrity import (
        test_clean_splits_manifest_structure,
        test_zero_hash_leakage,
        test_duplicate_clusters_isolated,
        test_contradictory_labels_quarantined_to_train,
        test_production_checkpoint_preserved,
        test_class_id_mapping_stability,
    )

integrity_tests = [
    ("test_clean_splits_manifest_structure", test_clean_splits_manifest_structure),
    ("test_zero_hash_leakage", test_zero_hash_leakage),
    ("test_duplicate_clusters_isolated", test_duplicate_clusters_isolated),
    ("test_contradictory_labels_quarantined_to_train", test_contradictory_labels_quarantined_to_train),
    ("test_production_checkpoint_preserved", test_production_checkpoint_preserved),
    ("test_class_id_mapping_stability", test_class_id_mapping_stability),
]

for name, test_fn in integrity_tests:
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

# 5. Test Camera Capture & Crop API Compatibility
try:
    from Skinora.backend.tests.test_camera_crop_compat import (
        test_cropped_square_image_accepted_by_api,
        test_camera_capture_filename_and_aspect_ratio_accepted,
    )
except ImportError:
    from backend.tests.test_camera_crop_compat import (
        test_cropped_square_image_accepted_by_api,
        test_camera_capture_filename_and_aspect_ratio_accepted,
    )

crop_tests = [
    ("test_cropped_square_image_accepted_by_api", test_cropped_square_image_accepted_by_api),
    ("test_camera_capture_filename_and_aspect_ratio_accepted", test_camera_capture_filename_and_aspect_ratio_accepted),
]

for name, test_fn in crop_tests:
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

# 6. Test ACNE-ECKH Pipeline & Label Mapping
try:
    from Skinora.backend.tests.test_eckh_pipeline import (
        test_hayashi_lesion_count_mapping,
        test_clinical_grade_string_mapping,
        test_eckh_pipeline_inspection_and_deduplication,
    )
except ImportError:
    from backend.tests.test_eckh_pipeline import (
        test_hayashi_lesion_count_mapping,
        test_clinical_grade_string_mapping,
        test_eckh_pipeline_inspection_and_deduplication,
    )

eckh_tests = [
    ("test_hayashi_lesion_count_mapping", test_hayashi_lesion_count_mapping),
    ("test_clinical_grade_string_mapping", test_clinical_grade_string_mapping),
    ("test_eckh_pipeline_inspection_and_deduplication", test_eckh_pipeline_inspection_and_deduplication),
]

for name, test_fn in eckh_tests:
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

# 7. Test ACNE04 Dataset Inspection & Preparation
try:
    from Skinora.backend.tests.test_acne04_inspection import (
        test_inspection_artifacts_exist,
        test_dataset_size_and_missing_values,
        test_label_consistency_and_hayashi_rules,
        test_bounding_box_validity,
        test_official_five_folds_counts,
        test_duplicate_groups_and_leakage,
        test_clean_splits_manifest_verification,
        test_production_checkpoint_preserved as test_prod_ckpt_acne04,
    )
except ImportError:
    from backend.tests.test_acne04_inspection import (
        test_inspection_artifacts_exist,
        test_dataset_size_and_missing_values,
        test_label_consistency_and_hayashi_rules,
        test_bounding_box_validity,
        test_official_five_folds_counts,
        test_duplicate_groups_and_leakage,
        test_clean_splits_manifest_verification,
        test_production_checkpoint_preserved as test_prod_ckpt_acne04,
    )

acne04_tests = [
    ("test_inspection_artifacts_exist", test_inspection_artifacts_exist),
    ("test_dataset_size_and_missing_values", test_dataset_size_and_missing_values),
    ("test_label_consistency_and_hayashi_rules", test_label_consistency_and_hayashi_rules),
    ("test_bounding_box_validity", test_bounding_box_validity),
    ("test_official_five_folds_counts", test_official_five_folds_counts),
    ("test_duplicate_groups_and_leakage", test_duplicate_groups_and_leakage),
    ("test_clean_splits_manifest_verification", test_clean_splits_manifest_verification),
    ("test_production_checkpoint_preserved", test_prod_ckpt_acne04),
]

for name, test_fn in acne04_tests:
    try:
        test_fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        failed += 1

print("=" * 60)
print(f"TEST RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)
sys.exit(0)
