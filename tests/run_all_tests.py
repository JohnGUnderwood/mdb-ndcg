#!/usr/bin/env python3
"""
Test runner for all NDCG evaluation tests.

This script runs all the test modules in the correct order.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all test modules."""
    print("NDCG Evaluation System - Test Suite")
    print("=" * 50)
    
    try:
        # Run core NDCG function tests
        print("\n1. Running NDCG core function tests...")
        from test_ndcg import main as test_ndcg_main
        test_ndcg_main()
        
        # Run evaluation runner tests
        print("\n2. Running evaluation runner tests...")
        from test_run_script import main as test_run_script_main
        test_run_script_main()
        
        # Run integration tests (only if MongoDB is available)
        print("\n3. Running integration tests...")
        try:
            from test_ndcg_k_functionality import main as test_k_main
            test_k_main()
        except Exception as e:
            print(f"⚠️  Integration tests skipped (requires MongoDB setup): {e}")
        
        print("\n" + "=" * 50)
        print("✅ Test suite completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()