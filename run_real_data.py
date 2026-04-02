# run_real_data.py

print("\n" + "=" * 80)
print("RUNNING REAL DATA ANALYSIS")
print("=" * 80)

# =============================
# (1) YOUR METHOD (HORSESHOE)
# =============================
print("\n--- YOUR METHOD (Regularized Horseshoe) ---")

import real_data_main

real_data_main.main()

# =============================
# (2) COMPARISON (LAPLACE VS HORSESHOE)
# =============================
print("\n--- COMPARISON (Laplace vs Horseshoe) ---")

import real_data_comparison

real_data_comparison.main()

print("\n" + "=" * 80)
print("REAL DATA ANALYSIS COMPLETED")
print("=" * 80)
