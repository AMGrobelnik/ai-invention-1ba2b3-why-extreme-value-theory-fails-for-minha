#!/usr/bin/env python3
"""
Test script to verify bootstrap CI implementation.
"""
import sys
sys.path.insert(0, '/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_art/gen_art_experiment_1')

from method import bootstrap_ci_correct, bootstrap_ci_incorrect, generate_shingles
import random

# Test with simple data
set_a = {'abc', 'def', 'ghi', 'jkl'}
set_b = {'abc', 'def', 'mno', 'pqr'}

print("Testing bootstrap_ci_correct...")
try:
    result = bootstrap_ci_correct(set_a, set_b, num_hashes=32, B=100, seed=42)
    print(f"Success! point={result.point_estimate:.3f}, CI=[{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"Contains point: {result.contains_point}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
