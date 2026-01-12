# Computational Kernels Module

The `biosparse.kernel` module provides high-performance computational kernels for bioinformatics, implemented using Numba JIT with optimization hints. All kernels work directly with the project's CSR sparse matrix type.

## Overview

### Submodules

| Module | Description |
|--------|-------------|
| `hvg` | Highly Variable Gene (HVG) selection |
| `mwu` | Mann-Whitney U test |
| `ttest` | Student's and Welch's t-test |
| `mmd` | Maximum Mean Discrepancy |
| `math` | Statistical utility functions |

### Design Pattern: One-vs-All

All group-based kernels follow a consistent convention:

- **group_ids**: Integer array where `0 = reference group`, `1, 2, 3... = target groups`
- **One-vs-all**: Computes reference vs each target simultaneously
- **Output shape**: `(n_rows, n_targets)` for multi-target results

```python
# Example: 3 groups (ref=0, target1=1, target2=2)
group_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2])  # Per-sample group assignment
n_targets = 2  # Number of target groups (excludes reference)

# Results have shape (n_genes, n_targets)
# result[:, 0] = ref vs target1
# result[:, 1] = ref vs target2
```

---

## Highly Variable Genes (HVG)

The `biosparse.kernel.hvg` module provides functions for identifying highly variable genes.

### Functions

#### `compute_moments(csr, ddof)`
Compute per-row mean and variance.

```python
from biosparse.kernel import hvg
from biosparse import CSRF64
import scipy.sparse as sp

# Create sparse matrix (genes x cells)
scipy_mat = sp.random(10000, 5000, density=0.1, format='csr')
csr = CSRF64.from_scipy(scipy_mat)

# Compute moments
means, variances = hvg.compute_moments(csr, ddof=1)
print(f"Means shape: {means.shape}")       # (10000,)
print(f"Variances shape: {variances.shape}")  # (10000,)
```

**Parameters:**
- `csr`: CSR sparse matrix (genes × cells)
- `ddof`: Delta degrees of freedom for variance (typically 1)

**Returns:** `(means, variances)` - Per-row mean and variance arrays

---

#### `compute_dispersion(means, vars)`
Compute dispersion = variance / mean.

```python
dispersions = hvg.compute_dispersion(means, variances)
```

**Parameters:**
- `means`: Per-gene mean values
- `vars`: Per-gene variance values

**Returns:** Dispersion values array

---

#### `normalize_dispersion(dispersions, means, min_mean, max_mean)`
Z-score normalize dispersion values.

```python
normalized = hvg.normalize_dispersion(
    dispersions,
    means,
    min_mean=0.0125,  # Filter genes with mean < 0.0125
    max_mean=3.0      # Filter genes with mean > 3.0
)
# Genes outside range get -inf
```

**Parameters:**
- `dispersions`: Raw dispersion values
- `means`: Per-gene means
- `min_mean`: Minimum mean threshold
- `max_mean`: Maximum mean threshold

**Returns:** Normalized dispersion (genes outside range = -inf)

---

#### `select_top_k(scores, k)`
Select top k genes by score.

```python
indices, mask = hvg.select_top_k(normalized, k=2000)
print(f"Top gene indices: {indices}")  # Shape: (2000,)
print(f"Selection mask: {mask}")       # Shape: (n_genes,), dtype=uint8
```

**Parameters:**
- `scores`: Score values per gene
- `k`: Number of top genes to select

**Returns:** `(indices, mask)` - Top k indices and binary mask

---

#### `select_hvg_by_dispersion(csr, n_top)`
Complete HVG selection pipeline.

```python
indices, mask, dispersions = hvg.select_hvg_by_dispersion(csr, n_top=2000)
print(f"Selected {len(indices)} highly variable genes")
```

**Parameters:**
- `csr`: CSR sparse matrix (genes × cells)
- `n_top`: Number of top genes to select

**Returns:** `(indices, mask, dispersions)` - Selection results

---

#### `compute_clipped_moments(csr, clip_vals)`
Compute moments with value clipping (for VST).

```python
clip_values = np.percentile(csr.to_dense(), 99, axis=1)
means, variances = hvg.compute_clipped_moments(csr, clip_values)
```

---

### Complete HVG Example

```python
import numpy as np
import scipy.sparse as sp
from biosparse import CSRF64
from biosparse.kernel import hvg

# Create expression matrix (genes x cells)
scipy_mat = sp.random(10000, 5000, density=0.1, format='csr', dtype='float64')
csr = CSRF64.from_scipy(scipy_mat)

# Method 1: Manual pipeline
means, variances = hvg.compute_moments(csr, ddof=1)
dispersions = hvg.compute_dispersion(means, variances)
norm_disp = hvg.normalize_dispersion(dispersions, means, 0.0125, 3.0)
top_indices, top_mask = hvg.select_top_k(norm_disp, k=2000)

# Method 2: All-in-one
top_indices, top_mask, dispersions = hvg.select_hvg_by_dispersion(csr, n_top=2000)

print(f"Selected {np.sum(top_mask)} highly variable genes")
```

---

## Mann-Whitney U Test

The `biosparse.kernel.mwu` module provides the Mann-Whitney U test for sparse matrices.

### Functions

#### `mwu_test(csr, group_ids, n_targets)`
Perform Mann-Whitney U test: reference vs all targets.

```python
from biosparse.kernel import mwu
import numpy as np

# Create sparse matrix (genes x cells)
csr = CSRF64.from_scipy(scipy_mat)

# Group assignments: 0=reference, 1=target1, 2=target2
n_cells = csr.ncols
group_ids = np.random.randint(0, 3, size=n_cells).astype(np.int64)
n_targets = 2

# Run test
u_stats, p_values, log2_fc, auroc = mwu.mwu_test(csr, group_ids, n_targets)

print(f"U-statistics shape: {u_stats.shape}")  # (n_genes, n_targets)
print(f"P-values shape: {p_values.shape}")      # (n_genes, n_targets)
print(f"Log2 FC shape: {log2_fc.shape}")        # (n_genes, n_targets)
print(f"AUROC shape: {auroc.shape}")            # (n_genes, n_targets)
```

**Parameters:**
- `csr`: CSR sparse matrix (genes × cells)
- `group_ids`: Group assignment for each cell (0 = ref, 1..n = targets)
- `n_targets`: Number of target groups

**Returns:** `(u_stats, p_values, log2_fc, auroc)` - Each with shape `(n_rows, n_targets)`

---

#### `count_groups(group_ids, n_groups)`
Count elements in each group.

```python
counts = mwu.count_groups(group_ids, n_groups=3)
print(f"Group sizes: {counts}")  # e.g., [100, 50, 75]
```

---

### Complete MWU Example

```python
import numpy as np
import scipy.sparse as sp
from biosparse import CSRF64
from biosparse.kernel import mwu

# Simulate expression data
n_genes, n_cells = 5000, 1000
scipy_mat = sp.random(n_genes, n_cells, density=0.1, format='csr', dtype='float64')
csr = CSRF64.from_scipy(scipy_mat)

# Define groups: control (0) vs treatment1 (1) vs treatment2 (2)
group_ids = np.array(
    [0] * 400 +  # 400 control cells
    [1] * 300 +  # 300 treatment1 cells
    [2] * 300,   # 300 treatment2 cells
    dtype=np.int64
)
n_targets = 2

# Run Mann-Whitney U test
u_stats, p_values, log2_fc, auroc = mwu.mwu_test(csr, group_ids, n_targets)

# Find significant genes for treatment1 vs control
significant = p_values[:, 0] < 0.05
upregulated = (p_values[:, 0] < 0.05) & (log2_fc[:, 0] > 1.0)
downregulated = (p_values[:, 0] < 0.05) & (log2_fc[:, 0] < -1.0)

print(f"Significant genes (treatment1 vs control): {np.sum(significant)}")
print(f"Upregulated (log2FC > 1): {np.sum(upregulated)}")
print(f"Downregulated (log2FC < -1): {np.sum(downregulated)}")
```

---

## T-Test

The `biosparse.kernel.ttest` module provides Student's and Welch's t-test.

### Functions

#### `ttest(csr, group_ids, n_targets, use_welch=True)`
Perform t-test: reference vs all targets.

```python
from biosparse.kernel import ttest

# Run Welch's t-test (default)
t_stats, p_values, log2_fc = ttest.ttest(csr, group_ids, n_targets, use_welch=True)

# Run Student's t-test
t_stats, p_values, log2_fc = ttest.ttest(csr, group_ids, n_targets, use_welch=False)
```

**Parameters:**
- `csr`: CSR sparse matrix (genes × cells)
- `group_ids`: Group assignment (0 = ref, 1..n = targets)
- `n_targets`: Number of target groups
- `use_welch`: If True, use Welch's t-test; else Student's

**Returns:** `(t_stats, p_values, log2_fc)` - Each with shape `(n_rows, n_targets)`

---

#### `welch_ttest(csr, group_ids, n_targets)`
Convenience wrapper for Welch's t-test.

```python
t_stats, p_values, log2_fc = ttest.welch_ttest(csr, group_ids, n_targets)
```

---

#### `student_ttest(csr, group_ids, n_targets)`
Convenience wrapper for Student's t-test.

```python
t_stats, p_values, log2_fc = ttest.student_ttest(csr, group_ids, n_targets)
```

---

### Welch's vs Student's T-Test

| Test | Assumption | When to Use |
|------|------------|-------------|
| **Welch's** | Unequal variances | Default choice; more robust |
| **Student's** | Equal variances | When you know variances are equal |

```python
# Welch's t-test (recommended)
t_w, p_w, fc_w = ttest.welch_ttest(csr, group_ids, n_targets)

# Student's t-test
t_s, p_s, fc_s = ttest.student_ttest(csr, group_ids, n_targets)
```

---

### Complete T-Test Example

```python
import numpy as np
import scipy.sparse as sp
from biosparse import CSRF64
from biosparse.kernel import ttest

# Simulate expression data
n_genes, n_cells = 5000, 1000
scipy_mat = sp.random(n_genes, n_cells, density=0.1, format='csr', dtype='float64')
csr = CSRF64.from_scipy(scipy_mat)

# Define groups
group_ids = np.array([0] * 500 + [1] * 500, dtype=np.int64)
n_targets = 1

# Run Welch's t-test
t_stats, p_values, log2_fc = ttest.welch_ttest(csr, group_ids, n_targets)

# Volcano plot data
significant = p_values[:, 0] < 0.05
large_effect = np.abs(log2_fc[:, 0]) > 1.0
de_genes = significant & large_effect

print(f"Differentially expressed genes: {np.sum(de_genes)}")

# Get top genes by absolute t-statistic
top_idx = np.argsort(np.abs(t_stats[:, 0]))[::-1][:10]
print(f"Top 10 genes by |t|: {top_idx}")
```

---

## Maximum Mean Discrepancy (MMD)

The `biosparse.kernel.mmd` module computes MMD² with RBF (Gaussian) kernel.

### Functions

#### `mmd_rbf(csr, group_ids, n_targets, gamma)`
Compute MMD² with RBF kernel.

```python
from biosparse.kernel import mmd

# gamma = 1 / (2 * sigma^2), typical values: 0.01 - 1.0
mmd_values = mmd.mmd_rbf(csr, group_ids, n_targets, gamma=0.1)
print(f"MMD² shape: {mmd_values.shape}")  # (n_rows, n_targets)
```

**Parameters:**
- `csr`: CSR sparse matrix (features × samples)
- `group_ids`: Group assignment (0 = ref, 1..n = targets)
- `n_targets`: Number of target groups
- `gamma`: RBF kernel parameter (1 / (2σ²))

**Returns:** MMD² values with shape `(n_rows, n_targets)`

---

### Understanding MMD

MMD (Maximum Mean Discrepancy) measures the difference between two distributions in a reproducing kernel Hilbert space.

```
MMD²(P, Q) = E[k(X, X')] + E[k(Y, Y')] - 2*E[k(X, Y)]

where:
- X, X' ~ P (reference distribution)
- Y, Y' ~ Q (target distribution)
- k(·, ·) is the RBF kernel: k(x, y) = exp(-γ * ||x - y||²)
```

---

### Complete MMD Example

```python
import numpy as np
import scipy.sparse as sp
from biosparse import CSRF64
from biosparse.kernel import mmd

# Simulate feature data
n_features, n_samples = 1000, 500
scipy_mat = sp.random(n_features, n_samples, density=0.1, format='csr', dtype='float64')
csr = CSRF64.from_scipy(scipy_mat)

# Define groups
group_ids = np.array(
    [0] * 200 +  # Reference
    [1] * 150 +  # Target 1
    [2] * 150,   # Target 2
    dtype=np.int64
)
n_targets = 2

# Compute MMD² with different gamma values
for gamma in [0.01, 0.1, 1.0]:
    mmd_values = mmd.mmd_rbf(csr, group_ids, n_targets, gamma=gamma)
    print(f"gamma={gamma}: mean MMD²={np.mean(mmd_values):.4f}")

# Find features with largest distribution difference
mmd_values = mmd.mmd_rbf(csr, group_ids, n_targets, gamma=0.1)
top_features = np.argsort(mmd_values[:, 0])[::-1][:10]
print(f"Top 10 features by MMD² (ref vs target1): {top_features}")
```

---

## Combining Kernels

### Differential Expression Pipeline

```python
import numpy as np
import scipy.sparse as sp
from biosparse import CSRF64
from biosparse.kernel import mwu, ttest, hvg

# Load expression matrix (genes x cells)
scipy_mat = sp.random(10000, 2000, density=0.1, format='csr', dtype='float64')
csr = CSRF64.from_scipy(scipy_mat)

# Define cell groups
group_ids = np.array([0] * 1000 + [1] * 1000, dtype=np.int64)
n_targets = 1

# Step 1: Filter to highly variable genes
top_idx, mask, _ = hvg.select_hvg_by_dispersion(csr, n_top=5000)

# Step 2: Run both statistical tests
# MWU (non-parametric)
u_stats, p_mwu, log2_fc_mwu, auroc = mwu.mwu_test(csr, group_ids, n_targets)

# Welch's t-test (parametric)
t_stats, p_ttest, log2_fc_ttest = ttest.welch_ttest(csr, group_ids, n_targets)

# Step 3: Combine results
# Use MWU for sparse data, t-test for interpretation
significant = (p_mwu[:, 0] < 0.05) & (p_ttest[:, 0] < 0.05)
upregulated = significant & (log2_fc_mwu[:, 0] > 1.0)
downregulated = significant & (log2_fc_mwu[:, 0] < -1.0)

print(f"Upregulated genes: {np.sum(upregulated)}")
print(f"Downregulated genes: {np.sum(downregulated)}")
```

---

## Performance Tips

### 1. Use Appropriate Data Types

```python
# Ensure int64 for group_ids
group_ids = group_ids.astype(np.int64)

# Results are always float64
```

### 2. Batch Processing

All kernels process all rows in parallel. Avoid calling repeatedly on single rows:

```python
# BAD: Row-by-row
for i in range(n_genes):
    result[i] = process_gene(csr_row[i])

# GOOD: Batch processing
results = mwu.mwu_test(csr, group_ids, n_targets)
```

### 3. Memory Considerations

Output arrays have shape `(n_rows, n_targets)`. For large matrices:

```python
# Estimate memory
n_genes = 20000
n_targets = 10
memory_per_output = n_genes * n_targets * 8  # 8 bytes per float64

# mwu_test returns 4 arrays
total_memory = memory_per_output * 4
print(f"Memory needed: {total_memory / 1e6:.1f} MB")
```

---

## API Reference Summary

### hvg

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_moments(csr, ddof)` | `(means, vars)` | Per-row mean and variance |
| `compute_dispersion(means, vars)` | `dispersions` | Variance/mean ratio |
| `normalize_dispersion(disp, means, min, max)` | `norm_disp` | Z-score normalized |
| `select_top_k(scores, k)` | `(indices, mask)` | Top k selection |
| `select_hvg_by_dispersion(csr, n_top)` | `(idx, mask, disp)` | Complete pipeline |
| `compute_clipped_moments(csr, clips)` | `(means, vars)` | Clipped moments |

### mwu

| Function | Returns | Description |
|----------|---------|-------------|
| `mwu_test(csr, groups, n_targets)` | `(U, p, log2fc, auroc)` | MWU test |
| `count_groups(group_ids, n_groups)` | `counts` | Group sizes |

### ttest

| Function | Returns | Description |
|----------|---------|-------------|
| `ttest(csr, groups, n_targets, welch)` | `(t, p, log2fc)` | T-test |
| `welch_ttest(csr, groups, n_targets)` | `(t, p, log2fc)` | Welch's t-test |
| `student_ttest(csr, groups, n_targets)` | `(t, p, log2fc)` | Student's t-test |

### mmd

| Function | Returns | Description |
|----------|---------|-------------|
| `mmd_rbf(csr, groups, n_targets, gamma)` | `mmd2` | MMD² with RBF kernel |
