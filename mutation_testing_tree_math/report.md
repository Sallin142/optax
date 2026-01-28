# Mutation #10 - ROR: Relational Operator Replacement: == to >= (`_vdot`)

```diff
--- original
+++ mutated
@@ -129,1 +129,1 @@
-  assert a.shape == b.shape
+  assert a.shape >= b.shape
```

This mutant is a bug in the code. `_vdot` is a function that is used by `tree_vdot`, which computes the inner product of 2 arrays. To compute the inner product of 2 arrays, you must ensure that the 2 arrays have the same dimension (e.g. array 1 = 20x1, array 2 = 20x1). The code in `_vdot` that follows always assumes that a.shape will be the same as b.shape. With this mutation, all legal array values are permitted, but so do illegal ones where the first provided array a is allowed to be larger in dimension than the 2nd array b that is given.

The following test will catch this mutation:

```python
  def test_mutant10(self):
    rng = np.random.RandomState(0)
    array_1 = rng.randn(21)
    array_2 = rng.randn(20)

    try:
      tu.tree_vdot(array_1, array_2)
    except:
      return

    raise AssertionError("Arrays of different shapes were permitted.")
```

# Mutation #25 - ROR: Relational Operator Replacement: == to <= (`tree_max`)

```diff
--- original
+++ mutated
@@ -212,1 +212,1 @@
-    if jnp.size(array) == 0:
+    if jnp.size(array) <= 0:
```

This mutant survives because it is an equivalent mutant and is therefore not a bug. The purpose of this if statement is to ensure that the array provided is not empty, so `<= 0` and `== 0` will serve the same purpose.

# Mutation #43 - CRP: Constant Replacement: 2 to 3 (`tree_norm`)

```diff
--- original
+++ mutated
@@ -291,1 +291,1 @@
-  if ord is None or ord == 2:
+  if ord is None or ord == 3:
```

This mutant is a bug in the code. According to the documentation, "ord is the order of the vector norm to compute from". The supported ord values are: None, 1, 2, and inf. Providing an ord value that is not supported is supposed to produce a ValueError. With the mutation however, it makes the legal ord value 2 produce a ValueError instead

The following test will catch this mutation:

```python
  def test_mutant43(self):
    ord = 2  # valid ord

    expected = jnp.sqrt(jnp.vdot(self.array_a, self.array_a).real)
    got = tu.tree_norm(self.array_a, ord=ord)  # ValueError
    np.testing.assert_allclose(expected, got)
```

# Mutation #55 - ROR: Relational Operator Replacement: == to <= (`tree_norm`)

```diff
--- original
+++ mutated
@@ -295,1 +295,1 @@
-  elif ord == 1:
+  elif ord <= 1:
```

This mutant is a bug in the code. Similar to mutant 43, ord is the order of the vector norm to compute from, and the supported values are None, 1, 2, and inf. Unlike the scenario in mutant 43 however, all legal values remain legal values, however, this mutation permits values such as 0 or any negative number, which are not supported. The correct behavior should be to return a ValueError, but the mutant will proceed to sum the tree instead. 

The following test will catch this mutation

```python
  def test_mutant55(self):
    ord = 0

    try:
      tu.tree_norm(self.array_a, ord=ord)
    except ValueError:
      return

    raise AssertionError(f"Unsupported ord value {ord} was permitted.")
```

# Mutation #63 - ROR: Relational Operator Replacement: == to >= (`tree_norm`)

```diff
--- original
+++ mutated
@@ -297,1 +297,1 @@
-  elif ord == jnp.inf or ord in ('inf', 'infinity'):
+  elif ord >= jnp.inf or ord in ('inf', 'infinity'):
```

This mutant survives because it is an equivalent mutant and is therefore not a bug. For `tree_norm`, all legal ord values are still permitted (None, 1, 2, inf), and all illegal ord values still produce an error. The value of ord cannot be greater than infinity, so it is similar to the size check in mutant 25 where sizes of an array cannot be less than 0.

# Conclusion

| Operator | Count | Percentage |
|----------|-------|------------|
| AOR | 27 | 27.0% |
| ROR | 30 | 30.0% |
| CRP | 41 | 41.0% |
| LCR | 2 | 2.0% |
| **Total** | **100** | **100.0%** |

| Routine | Total Mutants | Killed | Survived | Effectiveness |
|---------|---------------|--------|----------|---------------|
| `_square` | 2 | 2 | 0 | 100.0% |
| `_vdot` | 5 | 3 | 2 | 60.0% |
| `tree_add_scale` | 4 | 4 | 0 | 100.0% |
| `tree_allclose` | 4 | 1 | 3 | 25.0% |
| `tree_bias_correction` | 5 | 1 | 4 | 20.0% |
| `tree_max` | 9 | 8 | 1 | 88.9% |
| `tree_min` | 8 | 7 | 1 | 87.5% |
| `tree_norm` | 25 | 18 | 7 | 72.0% |
| `tree_scale` | 2 | 2 | 0 | 100.0% |
| `tree_sum` | 6 | 3 | 3 | 50.0% |
| `tree_update_infinity_moment` | 4 | 4 | 0 | 100.0% |
| `tree_update_moment` | 3 | 0 | 3 | 0.0% |
| `tree_update_moment_per_elem_norm` | 20 | 0 | 20 | 0.0% |
| `tree_vdot` | 3 | 3 | 0 | 100.0% |

- Total mutations: 100
  - AOR: 27
  - ROR: 30
  - CRP: 41
  - LCR: 2
- Mutation score: 56 / 100 = 56.00%
