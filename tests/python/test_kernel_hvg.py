"""Tests for kernel.hvg module - Highly Variable Gene selection.

Tests all HVG flavors:
    - hvg_seurat: Seurat flavor (binning + mean/std)
    - hvg_cell_ranger: Cell Ranger flavor (percentile + MAD)
    - hvg_seurat_v3: Seurat V3 flavor (VST + LOESS)
    - hvg_pearson_residuals: Pearson residuals flavor
"""

import pytest
import numpy as np

# Check dependencies
try:
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from biosparse._binding import CSRF64
    import biosparse._numba
    BINDING_AVAILABLE = True
except Exception:
    BINDING_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not available"),
    pytest.mark.skipif(not BINDING_AVAILABLE, reason="Rust FFI bindings not available"),
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gene_expression_matrix():
    """Create a gene expression CSR matrix (genes x cells)."""
    np.random.seed(42)
    # 500 genes, 100 cells, ~15% density
    mat = sp.random(500, 100, density=0.15, format='csr', dtype=np.float64)
    mat.data = np.abs(mat.data) * 10  # Make values positive
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def count_matrix():
    """Create a count matrix (raw counts for Seurat V3 / Pearson)."""
    np.random.seed(42)
    # 500 genes, 100 cells
    # Simulate count data with negative binomial-like distribution
    mat = sp.random(500, 100, density=0.2, format='csr', dtype=np.float64)
    # Convert to positive integers
    mat.data = np.floor(np.abs(mat.data) * 20).astype(np.float64)
    mat.eliminate_zeros()
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def small_matrix():
    """Create a small matrix for detailed testing."""
    np.random.seed(123)
    mat = sp.random(50, 20, density=0.3, format='csr', dtype=np.float64)
    mat.data = np.abs(mat.data) * 5
    return CSRF64.from_scipy(mat), mat


# =============================================================================
# Test Basic Statistics
# =============================================================================

class TestComputeMoments:
    """Test compute_moments function."""
    
    def test_compute_moments_basic(self, gene_expression_matrix):
        """compute_moments should return valid means and variances."""
        from biosparse.kernel.hvg import compute_moments
        
        csr, scipy_mat = gene_expression_matrix
        
        means, vars = compute_moments(csr, ddof=1)
        
        # Check shapes
        assert means.shape == (500,)
        assert vars.shape == (500,)
        
        # Means should be non-negative
        assert np.all(means >= 0.0)
        
        # Variances should be non-negative
        assert np.all(vars >= 0.0)
    
    def test_compute_moments_vs_numpy(self, gene_expression_matrix):
        """compute_moments should match numpy computation."""
        from biosparse.kernel.hvg import compute_moments
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        means, vars = compute_moments(csr, ddof=1)
        
        # Compare with numpy (first 50 genes)
        for gene_idx in range(50):
            row = dense[gene_idx, :]
            expected_mean = np.mean(row)
            expected_var = np.var(row, ddof=1)
            
            np.testing.assert_allclose(means[gene_idx], expected_mean, rtol=1e-10,
                err_msg=f"Gene {gene_idx}: mean mismatch")
            np.testing.assert_allclose(vars[gene_idx], expected_var, rtol=1e-10,
                err_msg=f"Gene {gene_idx}: variance mismatch")


class TestComputeDispersion:
    """Test compute_dispersion function."""
    
    def test_compute_dispersion_basic(self):
        """compute_dispersion should compute var/mean correctly."""
        from biosparse.kernel.hvg import compute_dispersion
        
        means = np.array([1.0, 2.0, 0.5, 0.0, 10.0])
        vars = np.array([2.0, 8.0, 0.5, 1.0, 50.0])
        
        disp = compute_dispersion(means, vars)
        
        # Expected: var/mean, but 0 for mean ~ 0
        expected = np.array([2.0, 4.0, 1.0, 0.0, 5.0])
        
        np.testing.assert_allclose(disp, expected, rtol=1e-10)


class TestSelectTopK:
    """Test select_top_k function."""
    
    def test_select_top_k_basic(self):
        """select_top_k should select correct indices."""
        from biosparse.kernel.hvg import select_top_k
        
        scores = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        
        indices, mask = select_top_k(scores, 3)
        
        # Top 3 should be indices with values 7, 5, 3
        assert set(indices.tolist()) == {3, 1, 2}
        
        # Mask should have 3 ones
        assert np.sum(mask) == 3
        assert mask[3] == 1
        assert mask[1] == 1
        assert mask[2] == 1
    
    def test_select_top_k_all(self):
        """select_top_k with k=n should select all."""
        from biosparse.kernel.hvg import select_top_k
        
        scores = np.array([1.0, 5.0, 3.0])
        
        indices, mask = select_top_k(scores, 3)
        
        assert len(indices) == 3
        assert np.all(mask == 1)


# =============================================================================
# Test HVG Seurat Flavor
# =============================================================================

class TestHVGSeurat:
    """Test hvg_seurat function (binning + mean/std)."""
    
    def test_hvg_seurat_basic(self, gene_expression_matrix):
        """hvg_seurat should return valid results."""
        from biosparse.kernel.hvg import hvg_seurat
        
        csr, scipy_mat = gene_expression_matrix
        
        indices, mask, means, dispersions, dispersions_norm = hvg_seurat(
            csr, n_top_genes=100, n_bins=20
        )
        
        # Check shapes
        assert len(indices) == 100
        assert len(mask) == 500
        assert len(means) == 500
        assert len(dispersions) == 500
        assert len(dispersions_norm) == 500
        
        # Mask should have exactly 100 ones
        assert np.sum(mask) == 100
        
        # Indices should correspond to mask
        for idx in indices:
            assert mask[idx] == 1
    
    def test_hvg_seurat_dispersion_ranking(self, gene_expression_matrix):
        """Selected genes should have high normalized dispersion."""
        from biosparse.kernel.hvg import hvg_seurat
        
        csr, scipy_mat = gene_expression_matrix
        
        indices, mask, _, _, dispersions_norm = hvg_seurat(
            csr, n_top_genes=50, n_bins=20
        )
        
        # Get valid dispersions (not -inf)
        valid_mask = dispersions_norm > -np.inf
        if np.any(valid_mask):
            # Selected genes should have high dispersion among valid ones
            selected_disps = dispersions_norm[mask.astype(bool)]
            valid_selected = selected_disps[selected_disps > -np.inf]
            
            if len(valid_selected) > 0:
                # At least some selected genes should be above median
                median_disp = np.median(dispersions_norm[valid_mask])
                assert np.mean(valid_selected) >= median_disp - 1.0


# =============================================================================
# Test HVG Cell Ranger Flavor
# =============================================================================

class TestHVGCellRanger:
    """Test hvg_cell_ranger function (percentile + MAD)."""
    
    def test_hvg_cell_ranger_basic(self, gene_expression_matrix):
        """hvg_cell_ranger should return valid results."""
        from biosparse.kernel.hvg import hvg_cell_ranger
        
        csr, scipy_mat = gene_expression_matrix
        
        indices, mask, means, dispersions, dispersions_norm = hvg_cell_ranger(
            csr, n_top_genes=100
        )
        
        # Check shapes
        assert len(indices) == 100
        assert len(mask) == 500
        assert len(means) == 500
        assert len(dispersions) == 500
        assert len(dispersions_norm) == 500
        
        # Mask should have exactly 100 ones
        assert np.sum(mask) == 100


# =============================================================================
# Test HVG Seurat V3 Flavor
# =============================================================================

class TestHVGSeuratV3:
    """Test hvg_seurat_v3 function (VST + LOESS)."""
    
    def test_hvg_seurat_v3_basic(self, count_matrix):
        """hvg_seurat_v3 should return valid results."""
        from biosparse.kernel.hvg import hvg_seurat_v3
        
        csr, scipy_mat = count_matrix
        
        indices, mask, means, variances, variances_norm = hvg_seurat_v3(
            csr, n_top_genes=100, span=0.3
        )
        
        # Check shapes
        assert len(indices) == 100
        assert len(mask) == 500
        assert len(means) == 500
        assert len(variances) == 500
        assert len(variances_norm) == 500
        
        # Mask should have exactly 100 ones
        assert np.sum(mask) == 100
        
        # Means should be non-negative
        assert np.all(means >= 0.0)
        
        # Variances should be non-negative
        assert np.all(variances >= 0.0)
    
    def test_hvg_seurat_v3_different_span(self, count_matrix):
        """hvg_seurat_v3 should work with different span values."""
        from biosparse.kernel.hvg import hvg_seurat_v3
        
        csr, scipy_mat = count_matrix
        
        # Test with different span values
        for span in [0.2, 0.3, 0.5]:
            indices, mask, _, _, _ = hvg_seurat_v3(csr, n_top_genes=50, span=span)
            assert len(indices) == 50
            assert np.sum(mask) == 50


# =============================================================================
# Test HVG Pearson Residuals Flavor
# =============================================================================

class TestHVGPearsonResiduals:
    """Test hvg_pearson_residuals function."""
    
    def test_hvg_pearson_basic(self, count_matrix):
        """hvg_pearson_residuals should return valid results."""
        from biosparse.kernel.hvg import hvg_pearson_residuals
        
        csr, scipy_mat = count_matrix
        
        indices, mask, means, variances, residual_vars = hvg_pearson_residuals(
            csr, n_top_genes=100, theta=100.0
        )
        
        # Check shapes
        assert len(indices) == 100
        assert len(mask) == 500
        assert len(means) == 500
        assert len(variances) == 500
        assert len(residual_vars) == 500
        
        # Mask should have exactly 100 ones
        assert np.sum(mask) == 100
        
        # Residual variances should be non-negative
        assert np.all(residual_vars >= 0.0)
    
    def test_hvg_pearson_different_theta(self, count_matrix):
        """hvg_pearson_residuals should work with different theta values."""
        from biosparse.kernel.hvg import hvg_pearson_residuals
        
        csr, scipy_mat = count_matrix
        
        # Test with different theta values
        for theta in [10.0, 100.0, 1000.0]:
            indices, mask, _, _, residual_vars = hvg_pearson_residuals(
                csr, n_top_genes=50, theta=theta
            )
            assert len(indices) == 50
            assert np.sum(mask) == 50


# =============================================================================
# Test Legacy API
# =============================================================================

class TestSelectHVGByDispersion:
    """Test legacy select_hvg_by_dispersion function."""
    
    def test_select_hvg_basic(self, gene_expression_matrix):
        """select_hvg_by_dispersion should return valid results."""
        from biosparse.kernel.hvg import select_hvg_by_dispersion
        
        csr, scipy_mat = gene_expression_matrix
        
        indices, mask, dispersions = select_hvg_by_dispersion(csr, 100)
        
        # Check shapes
        assert len(indices) == 100
        assert len(mask) == 500
        assert len(dispersions) == 500
        
        # Mask should have exactly 100 ones
        assert np.sum(mask) == 100
        
        # Indices should correspond to mask
        for idx in indices:
            assert mask[idx] == 1
    
    def test_select_hvg_dispersion_order(self, gene_expression_matrix):
        """Selected genes should have highest dispersions."""
        from biosparse.kernel.hvg import select_hvg_by_dispersion
        
        csr, scipy_mat = gene_expression_matrix
        
        indices, mask, dispersions = select_hvg_by_dispersion(csr, 50)
        
        selected_disps = dispersions[indices]
        unselected_mask = mask == 0
        unselected_disps = dispersions[unselected_mask]
        
        # Minimum selected should be >= maximum unselected
        min_selected = np.min(selected_disps)
        max_unselected = np.max(unselected_disps)
        
        assert min_selected >= max_unselected


# =============================================================================
# Test Math Submodule - Descriptive Statistics
# =============================================================================

class TestDescriptiveStats:
    """Test math._descriptive functions."""
    
    def test_median(self):
        """median should compute correct median."""
        from biosparse.kernel.math import median
        
        # Odd length
        arr = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        assert median(arr) == 3.0
        
        # Even length
        arr = np.array([1.0, 5.0, 3.0, 7.0])
        assert median(arr) == 4.0  # (3+5)/2
    
    def test_mad(self):
        """mad should compute correct MAD."""
        from biosparse.kernel.math import median, mad
        
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # median = 3, deviations = [2, 1, 0, 1, 2], median of devs = 1
        result = mad(arr)
        assert result == 1.0
    
    def test_quantile(self):
        """quantile should compute correct quantiles."""
        from biosparse.kernel.math import quantile
        
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert quantile(arr, 0.0) == 1.0
        assert quantile(arr, 1.0) == 5.0
        assert quantile(arr, 0.5) == 3.0
    
    def test_percentile(self):
        """percentile should compute correct percentiles."""
        from biosparse.kernel.math import percentile
        
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert percentile(arr, 0.0) == 1.0
        assert percentile(arr, 100.0) == 5.0
        assert percentile(arr, 50.0) == 3.0


# =============================================================================
# Test Math Submodule - Regression
# =============================================================================

class TestRegression:
    """Test math._regression functions."""
    
    def test_loess_fit_linear(self):
        """loess_fit should fit linear data well."""
        from biosparse.kernel.math import loess_fit
        
        # Generate linear data
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 50)
        
        fitted = loess_fit(x, y, span=0.5, degree=1)
        
        # Fitted values should be close to true line
        true_y = 2.0 * x + 1.0
        rmse = np.sqrt(np.mean((fitted - true_y) ** 2))
        assert rmse < 0.5  # Reasonable fit
    
    def test_loess_fit_quadratic(self):
        """loess_fit with degree=2 should fit quadratic data."""
        from biosparse.kernel.math import loess_fit
        
        # Generate quadratic data
        x = np.linspace(-5, 5, 50)
        y = x ** 2 + np.random.normal(0, 0.5, 50)
        
        fitted = loess_fit(x, y, span=0.5, degree=2)
        
        # Fitted values should be close to parabola
        true_y = x ** 2
        rmse = np.sqrt(np.mean((fitted - true_y) ** 2))
        assert rmse < 2.0  # Reasonable fit
    
    def test_weighted_polyfit(self):
        """weighted_polyfit should fit weighted data correctly."""
        from biosparse.kernel.math import weighted_polyfit_1, weighted_polyfit_2
        
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x
        weights = np.ones(5, dtype=np.float64)
        
        a, b = weighted_polyfit_1(x, y, weights, 5)
        
        # Should recover y = 0 + 2x
        np.testing.assert_allclose(a, 0.0, atol=1e-10)
        np.testing.assert_allclose(b, 2.0, atol=1e-10)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for HVG selection."""
    
    def test_all_flavors_return_same_shape(self, count_matrix):
        """All HVG flavors should return consistent shapes."""
        from biosparse.kernel.hvg import (
            hvg_seurat, hvg_cell_ranger, hvg_seurat_v3, hvg_pearson_residuals
        )
        
        csr, scipy_mat = count_matrix
        n_top = 50
        
        results = [
            hvg_seurat(csr, n_top),
            hvg_cell_ranger(csr, n_top),
            hvg_seurat_v3(csr, n_top, span=0.3),
            hvg_pearson_residuals(csr, n_top, theta=100.0),
        ]
        
        for i, result in enumerate(results):
            indices, mask, _, _, _ = result
            assert len(indices) == n_top, f"Flavor {i} returned wrong number of indices"
            assert np.sum(mask) == n_top, f"Flavor {i} returned wrong mask count"
    
    def test_different_n_top_genes(self, gene_expression_matrix):
        """HVG selection should work with different n_top_genes values."""
        from biosparse.kernel.hvg import hvg_seurat
        
        csr, scipy_mat = gene_expression_matrix
        
        for n_top in [10, 50, 100, 200]:
            indices, mask, _, _, _ = hvg_seurat(csr, n_top)
            assert len(indices) == n_top
            assert np.sum(mask) == n_top
