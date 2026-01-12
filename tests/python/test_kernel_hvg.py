"""Tests for kernel.hvg module - Highly Variable Gene selection."""

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


@pytest.fixture
def gene_expression_matrix():
    """Create a gene expression CSR matrix (genes x cells)."""
    import scipy.sparse as sp
    np.random.seed(42)
    # 500 genes, 100 cells, ~15% density
    mat = sp.random(500, 100, density=0.15, format='csr', dtype=np.float64)
    mat.data = np.abs(mat.data) * 10  # Make values positive
    return CSRF64.from_scipy(mat), mat


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
    
    def test_compute_moments_ddof0(self, gene_expression_matrix):
        """compute_moments with ddof=0 should match population variance."""
        from biosparse.kernel.hvg import compute_moments
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        means, vars = compute_moments(csr, ddof=0)
        
        # Compare with numpy (first 20 genes)
        for gene_idx in range(20):
            row = dense[gene_idx, :]
            expected_var = np.var(row, ddof=0)
            
            np.testing.assert_allclose(vars[gene_idx], expected_var, rtol=1e-10)


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
    
    def test_compute_dispersion_zero_mean(self):
        """compute_dispersion should return 0 for zero mean."""
        from biosparse.kernel.hvg import compute_dispersion
        
        means = np.array([0.0, 1e-15, 1.0])
        vars = np.array([1.0, 1.0, 1.0])
        
        disp = compute_dispersion(means, vars)
        
        # First two should be 0 (mean too small), third should be 1.0
        assert disp[0] == 0.0
        assert disp[1] == 0.0
        np.testing.assert_allclose(disp[2], 1.0)


class TestNormalizeDispersion:
    """Test normalize_dispersion function."""
    
    def test_normalize_dispersion_basic(self):
        """normalize_dispersion should z-score normalize."""
        from biosparse.kernel.hvg import normalize_dispersion
        
        dispersions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        means = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        norm_disp = normalize_dispersion(dispersions, means, 0.0, 10.0)
        
        # Should be z-scored: (x - mean) / std
        expected_mean = np.mean(dispersions)
        expected_std = np.std(dispersions, ddof=0)
        expected = (dispersions - expected_mean) / expected_std
        
        np.testing.assert_allclose(norm_disp, expected, rtol=1e-10)
    
    def test_normalize_dispersion_filter(self):
        """normalize_dispersion should filter by mean range."""
        from biosparse.kernel.hvg import normalize_dispersion
        
        dispersions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        means = np.array([0.1, 0.5, 1.0, 2.0, 10.0])  # First and last outside range
        
        norm_disp = normalize_dispersion(dispersions, means, 0.2, 5.0)
        
        # First and last should be -inf (outside mean range)
        assert norm_disp[0] == -np.inf
        assert norm_disp[4] == -np.inf
        
        # Middle three should be normalized
        assert np.isfinite(norm_disp[1])
        assert np.isfinite(norm_disp[2])
        assert np.isfinite(norm_disp[3])


class TestSelectTopK:
    """Test select_top_k function."""
    
    def test_select_top_k_basic(self):
        """select_top_k should select correct indices."""
        from biosparse.kernel.hvg import select_top_k
        
        scores = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        
        indices, mask = select_top_k(scores, 3)
        
        # Top 3 indices should be 3, 1, 2 (sorted by score desc)
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


class TestSelectHVGByDispersion:
    """Test select_hvg_by_dispersion function."""
    
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
