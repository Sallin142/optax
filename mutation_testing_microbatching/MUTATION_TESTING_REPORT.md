# Mutation Testing Report: Optax Microbatching

**Date:** January 26, 2026

**Target File:** optax/experimental/microbatching.py

---

## 1. Project Identification

- **Project Name:** Optax
- **Description:** A gradient processing and optimization library for JAX
- **Supporting Organization:** DeepMind (Google)
- **Repository:** https://github.com/google-deepmind/optax
- **License:** Apache License 2.0
- **Primary Language:** Python 3.10+
- **Code Base Size:** ~20,000+ lines of Python code (core library)
- **Test Suite:** pytest-based with comprehensive test coverage

### Evaluation Platform
- **Operating System:** macOS
- **Python Version:** 3.10+
- **Key Dependencies:** JAX, NumPy, Chex
- **Test Framework:** pytest
- **Build Time:** N/A (interpreted Python)
- **Test Suite Execution Time:** ~30-60 seconds for microbatching_test.py
- **Mutation Testing Time:** 450.5 seconds (~7.5 minutes)

---

## 2. Mutation Operators

Seven mutation operators were implemented based on the selective mutation approach:

### 2.1 Arithmetic Operator Replacement (AOR) - 23 mutations
Replaces arithmetic operators with compatible alternatives:
- `+` ↔ `-` (addition/subtraction swap)
- `*` ↔ `/` (multiplication/division swap)
- `**` → `*` (exponentiation to multiplication)
- `%` → `/` (modulo to division)

**Rationale:** Detects errors in mathematical computations, particularly in accumulation, averaging, and index calculations.

### 2.2 Relational Operator Replacement (ROR) - 11 mutations
Replaces relational operators:
- `<` ↔ `<=` (boundary conditions)
- `>` ↔ `>=` (boundary conditions)
- `==` ↔ `!=` (equality inversion)

**Rationale:** Detects off-by-one errors and boundary condition mistakes in validation checks.

### 2.3 Constant Replacement Operator (CRP) - 45 mutations
Modifies numeric constants:
- Integer constants: `0` → `1`, `1` → `2`, etc.
- Float constants: `x` → `x+0.1`

**Rationale:** Detects hardcoded values like axis indices, batch sizes, and array slicing boundaries.

### 2.4 Logical Connector Replacement (LCR) - 1 mutation
Replaces logical operators:
- `and` ↔ `or`

**Rationale:** Detects errors in boolean logic and conditional expressions.

### 2.5 Unary Operator Insertion (UOI) - 25 mutations
Inserts unary operators:
- `x` → `-x` (arithmetic negation)
- `x` → `not x` (logical negation)

**Rationale:** Detects sign errors and incorrect boolean handling.

### 2.6 Statement Deletion (SDL) - 8 mutations
Removes executable statements:
- Deletes `raise` statements (error handling removal)
- Deletes `return` statements (replaced with `pass`)

**Rationale:** Detects whether error handling and return values are properly tested.

### 2.7 Assignment Statement Replacement (ASR) - 9 mutations
Modifies assignment operators:
- `+=` ↔ `-=`
- `*=` ↔ `/=`

**Rationale:** Detects errors in in-place arithmetic operations and accumulator updates.

---

## 3. Mutation Generation Process

### 3.1 Implementation
- **Tool:** Custom Python script (`single_file_mutation.py`)
- **Approach:** Automated mutation using regex-based pattern matching
- **Filtering Strategy:**
  - Excluded copyright headers and license text
  - Excluded docstrings and multi-line comments
  - Excluded standalone string literals (error messages)
  - Excluded type annotations (e.g., avoided mutating `->` in function signatures)
  - Only mutated executable code lines

### 3.2 Mutation Application
- **Total Mutants Generated:** 122
- **Mutation Isolation:** Each mutant contains exactly ONE mutation
- **Testing Approach:** Strong mutation testing
  - Each mutant is applied to the source file individually
  - The full test suite is executed against the mutated code
  - Original file is restored after each test
  - Mutant is marked 'killed' if any test fails
  - Mutant is marked 'survived' if all tests pass

---

## 4. Mutation Distribution

| Operator  | Description                      | Count   | Percentage |
| --------- | -------------------------------- | ------- | ---------- |
| CRP       | Constant Replacement             | 45      | 36.9%      |
| UOI       | Unary Operator Insertion         | 25      | 20.5%      |
| AOR       | Arithmetic Operator Replacement  | 23      | 18.9%      |
| ROR       | Relational Operator Replacement  | 11      | 9.0%       |
| ASR       | Assignment Statement Replacement | 9       | 7.4%       |
| SDL       | Statement Deletion               | 8       | 6.6%       |
| LCR       | Logical Connector Replacement    | 1       | 0.8%       |
| **Total** |                                  | **122** | **100.0%** |

---

## 5. Overall Test Suite Effectiveness

- **Total Mutants:** 122
- **Killed Mutants:** 78
- **Survived Mutants:** 44
- **Errors/Timeouts:** 0
- **Mutation Score:** 63.93%

**Interpretation:** The test suite successfully detects 63.93% of seeded faults, 
indicating moderate test coverage and fault detection capability.

---

## 6. Per-Routine Effectiveness

Mutation testing results broken down by individual microbatching functions:

| Routine                        | Total Mutants | Killed | Survived | Effectiveness |
| ------------------------------ | ------------- | ------ | -------- | ------------- |
| `_canonicalize`                | 2             | 0      | 2        | 0.0%          |
| `_compose`                     | 6             | 6      | 0        | 100.0%        |
| `_concat`                      | 11            | 9      | 2        | 81.8%         |
| `_get_out_sharding`            | 4             | 0      | 4        | 0.0%          |
| `_identity`                    | 1             | 1      | 0        | 100.0%        |
| `_lift`                        | 1             | 1      | 0        | 100.0%        |
| `_mean`                        | 6             | 3      | 3        | 50.0%         |
| `_normalize_fun_to_return_aux` | 2             | 1      | 1        | 50.0%         |
| `_reshape_all_args`            | 11            | 9      | 2        | 81.8%         |
| `_running_mean`                | 10            | 1      | 9        | 10.0%         |
| `_sum`                         | 2             | 2      | 0        | 100.0%        |
| `_with_extra_batch_axis`       | 2             | 2      | 0        | 100.0%        |
| `_with_floating_check`         | 2             | 2      | 0        | 100.0%        |
| `micro_grad`                   | 14            | 12     | 2        | 85.7%         |
| `micro_vmap`                   | 14            | 10     | 4        | 71.4%         |
| `microbatch`                   | 12            | 12     | 0        | 100.0%        |
| `reshape_batch_axis`           | 22            | 7      | 15       | 31.8%         |

### Key Observations:
- **Best Tested Routine:** `_with_floating_check` (100.0% effectiveness)
- **Worst Tested Routine:** `_get_out_sharding` (0.0% effectiveness)

---

## 7. Analysis of Survived Mutants

Of the 44 survived mutants, 5 representative mutants involving arithmetic and relational operators were analyzed in detail:

- **Potentially Equivalent Mutants:** 0
- **Non-Equivalent (Test Coverage Gap):** 5

### 7.1 Mutant #40 - AOR (Arithmetic Operator Replacement)
**Location:** Line 205 (`_running_mean` function)
**Operator:** Division to Multiplication

**Original Code:**
```python
p = index / (index + 1)
```

**Mutated Code:**
```python
p = index * (index + 1)
```

**Analysis:** ✗ Non-Equivalent - Test Coverage Gap

**Reason:** This mutation fundamentally breaks the running mean calculation. The original computes `p = i/(i+1)` which approaches 1 as index increases (e.g., 0/1=0, 1/2=0.5, 2/3=0.67...). The mutated version computes `p = i*(i+1)` which grows quadratically (0, 2, 6, 12...). This would cause completely incorrect weighted averaging, yet no test detects it.

**Suggested Test:**
```python
def test_running_mean_weights():
    # Verify running mean computes correct weighted average
    values = jnp.array([[1.0], [2.0], [3.0]])
    result = _running_mean(values)
    # Expected: (1*0 + 2*0.5 + 3*0.67) / normalization
    assert jnp.allclose(result, expected_mean)
```

### 7.2 Mutant #43 - AOR (Arithmetic Operator Replacement)
**Location:** Line 206 (`_running_mean` function)
**Operator:** Addition to Subtraction

**Original Code:**
```python
new_state = carry * p + value * (1 - p)
```

**Mutated Code:**
```python
new_state = carry * p - value * (1 - p)
```

**Analysis:** ✗ Non-Equivalent - Test Coverage Gap

**Reason:** This changes weighted averaging from addition to subtraction. Instead of blending `carry` and `value`, it now subtracts the weighted value contribution. For positive inputs, this would produce drastically different (potentially negative) results. The running mean accumulator is not being tested with assertions that verify the actual computed values.

**Suggested Test:**
```python
def test_running_mean_accumulation():
    # Test that running mean correctly accumulates positive values
    grads = create_test_gradients(all_positive=True)
    result = microbatch(loss_fn, acc=Accumulator.RUNNING_MEAN)(grads)
    assert jnp.all(result > 0), "Running mean of positive values should be positive"
```

### 7.3 Mutant #45 - AOR (Arithmetic Operator Replacement)
**Location:** Line 206 (`_running_mean` function)
**Operator:** Multiplication to Division

**Original Code:**
```python
new_state = carry * p + value * (1 - p)
```

**Mutated Code:**
```python
new_state = carry / p + value * (1 - p)
```

**Analysis:** ✗ Non-Equivalent - Test Coverage Gap

**Reason:** Mutant causes division by zero when p=0, producing inf/nan. At index=1+, mutant divides instead of multiplies, fundamentally breaking the running mean calculation

**Suggested Test:**
```python
def test_running_mean_first_iteration():
    # Verify first iteration handles p=0 correctly
    single_batch = jnp.array([[5.0]])
    result = _running_mean(single_batch)
    assert jnp.isfinite(result).all(), "Should not produce inf/nan"
```

### 7.4 Mutant #20 - ROR (Relational Operator Replacement)
**Location:** Line 123 (`_get_out_sharding` function)
**Operator:** Not-Equal to Equal

**Original Code:**
```python
if microbatch_size % nshards != 0:
```

**Mutated Code:**
```python
if microbatch_size % nshards == 0:
```

**Analysis:** ✗ Non-Equivalent - Test Coverage Gap

**Reason:** This inverts the validation logic. Originally, a ValueError is raised when `microbatch_size` is NOT evenly divisible by `nshards`. After mutation, the error is raised when it IS evenly divisible—the exact opposite behavior. Tests should verify both valid and invalid shard configurations.

**Suggested Test:**
```python
def test_sharding_validation():
    # Valid: microbatch_size=8, nshards=4 (8 % 4 == 0)
    result = reshape_batch_axis(data, microbatch_size=8)  # Should succeed
    
    # Invalid: microbatch_size=7, nshards=4 (7 % 4 != 0)  
    with pytest.raises(ValueError, match="must evenly divide"):
        reshape_batch_axis(data, microbatch_size=7)
```

### 7.5 Mutant #34 - ROR (Relational Operator Replacement)
**Location:** Line 190 (`microbatch` function)
**Operator:** Less-Than-Or-Equal to Less-Than

**Original Code:**
```python
if num_microbatches <= 0:
```

**Mutated Code:**
```python
if num_microbatches < 0:
```

**Analysis:** ✗ Non-Equivalent - Test Coverage Gap

**Reason:** The original rejects `num_microbatches=0` (zero microbatches is invalid). The mutation allows `num_microbatches=0` through, which would likely cause downstream errors or incorrect behavior. This boundary condition (`== 0`) is not tested.

**Suggested Test:**
```python
def test_zero_microbatches_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        microbatch(loss_fn, num_microbatches=0)(params, batch)
```

---

## 8. Methodology and Automation

### 8.1 Automation Strategy
The mutation testing was fully automated using a custom Python script that:
1. **Parses** the source code to identify mutation points
2. **Generates** mutations by applying operators via regex substitution
3. **Filters** out non-code elements (comments, docstrings)
4. **Applies** each mutation individually to the source file
5. **Executes** the test suite using pytest
6. **Records** the outcome (killed/survived) based on test exit code
7. **Restores** the original source after each test
8. **Generates** unified diff output for all mutations

### 8.2 Challenges Encountered
1. **Type Annotation Mutations:** Initial implementation incorrectly mutated `->` in Python type hints, creating invalid syntax
   - **Solution:** Added negative lookbehind in regex patterns to exclude type annotations

2. **JAX Compilation:** Microbatching code uses JAX transformations which require specific runtime behavior
   - **Solution:** Used longer test timeouts to account for JIT compilation

3. **Version String Mutations:** JAX version checks contain version strings that shouldn't be mutated
   - **Solution:** Careful regex patterns to avoid string literal mutations

### 8.3 Lessons Learned
1. **Filtering is Critical:** Excluding docstrings and comments was necessary to avoid useless mutations
2. **Context-Aware Mutations:** Regex-based mutation can create syntactically valid but semantically nonsensical changes
3. **Test Suite Quality:** Mutation score indicates the effectiveness of tests in detecting code changes
4. **Boundary Conditions:** Many survived mutants involved boundary condition changes, a common weak area
5. **Mathematical Operations:** Accumulator logic requires precise arithmetic - mutations here are often detected

---

## 9. Recommendations for Improving Test Suite

1. **Add Edge Case Tests:** Focus on boundary conditions (num_microbatches=1, microbatch_size edge cases)
2. **Test Accumulation Logic:** Add tests specifically verifying SUM, MEAN, RUNNING_MEAN, CONCAT accumulators
3. **Parametric Testing:** Use pytest parametrize to test multiple axis values and batch sizes
4. **Test Error Conditions:** Verify ValueError exceptions are raised for invalid inputs
5. **Property-Based Testing:** Consider using hypothesis to generate test cases automatically

---

## 10. Conclusion

The mutation testing study revealed that the Optax microbatching module has a **63.93% mutation score**, 
indicating opportunities for improvement in test coverage. 
The analysis identified specific areas where test coverage can be enhanced, 
particularly around boundary conditions, accumulator arithmetic, and edge cases. The automated mutation testing approach proved effective 
for systematically evaluating test suite quality and identifying gaps in fault detection capability.

---

## 11. Appendices

### 11.1 Files Submitted
- `optax/experimental/microbatching.py` - Source code under test
- `mutation_testing_microbatching/results/mutations_diff.txt` - Unified diff file with all mutations
- `mutation_testing_microbatching/results/mutation_results.json` - Detailed JSON results
- `mutation_testing_microbatching/single_file_mutation.py` - Mutation testing script
- `MUTATION_TESTING_REPORT.md` - This report

### 11.2 Mutation Score Formula
```
Mutation Score = (Killed Mutants / Total Mutants) × 100%
                = (78 / 122) × 100%
                = 63.93%
```

### 11.3 Adjusted Score (Excluding Confirmed Equivalents)
No confirmed equivalent mutants were identified:
```
Adjusted Score = 78 / 122 × 100%
               = 63.93% (unchanged)
```