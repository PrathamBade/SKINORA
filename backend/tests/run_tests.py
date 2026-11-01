from pathlib import Path
import sys
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
"""Standalone test runner for Skinora backend test suite."""
import sys
import traceback

print("=" * 60)
print("RUNNING SKINORA BACKEND TEST SUITE")
print("=" * 60)

passed = 0
failed = 0

# 1. Test model
from Skinora.backend.tests.test_model import (
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
from Skinora.backend.tests.test_evaluation import (
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

print("=" * 60)
print(f"TEST RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)
sys.exit(0)
