All the code used in this project is available in this GitHub repository, structured into different component files according to their functions. All results for our proposed AutoQEC framework can be reproduced using this repository.

### File Overview

- **`run_baseline_EfficientSU2.py`**: This file evaluates the performance of the baseline method `EfficientSU2`. The performance metrics include molecule energy and decomposed gate counts. This serves as a comparison baseline for the overall AutoQEC framework. The results are presented in Table 1.

- **`run_stage1_FT_gates_search.py`**: This file obtains the supported FT gateset for a given QEC code using different FT gate training methods, including noise-free training (`NF`), hardware-specific noisy training (`HN`), and our proposed error-bounded training method (`AutoQEC-I`). It constitutes the first stage of the AutoQEC framework, and serves as a prerequisite for the second stage. The searched FT gate sets are listed in Table 1.

- **`run_stage2_VQE_circuit_search.py`**: This file searches high-performance quantum circuits for VQE applications using an available non-universal FT gateset generated from `AutoQEC-I`. Two circuit search methods are supported: random sampling (`Rand`) and evolutionary-based search (`AutoQEC-II`). This constitutes the second stage of the AutoQEC framework. The resulting circuit performances, including molecular energies and gate counts, are summarized in Table 1. The search space exploration for the `Rand` method is demonstrated in Figure 4.

- **`run_stage3_FTQC_implementation.py`**: This file implements the QEC-based FTQC for H₂ molecule. The underlying circuit is obtained using the `AutoQEC-II` search method. Results are shown in Table 2.

- **`evaluate_FT_gates_training_methods.py`**: This file evaluates different FT gate training methods (i.e., `NF`, `HN`, and `AutoQEC-I`) under various noise levels. Evaluation results are presented in Figure 3, and the noise levels are listed in this file.

- **`prove_errors_indistinguishability.md`**: This file provides the detailed proofs of Lemma 1 and Theorem 2, as part of Component I of AutoQEC.

- **Other `a_`-prefixed files**: These files define core classes and functions used throughout AutoQEC.
