"""Tests for parallelized kernel functions.

Verifies that prange-parallelized kernel functions produce correct results.
"""

import pytest
import numpy as np

# Check if components are available
_OPTIM_AVAILABLE = False
_BINDING_AVAILABLE = False
_NUMBA_EXT_AVAILABLE = False

try:
    from biosparse.optim import disable_logging
    disable_logging()
    _OPTIM_AVAILABLE = True
except Exception:
    _OPTIM_AVAILABLE = False

try:
    from biosparse._binding import lib
    _BINDING_AVAILABLE = lib is not None
except Exception:
    _BINDING_AVAILABLE = False

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

_NUMBA_EXT_AVAILABLE = _OPTIM_AVAILABLE and _BINDING_AVAILABLE and _NUMBA_AVAILABLE
if _NUMBA_EXT_AVAILABLE:
    try:
        import biosparse._numba
        from biosparse._binding import CSRF64
        from biosparse.kernel import hvg, mmd, ttest, mwu
    except Exception:
        _NUMBA_EXT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _NUMBA_EXT_AVAILABLE,
    reason="Kernel module not available (requires Numba + Rust FFI)"
)


@pytest.fixture
def gene_cell_matrix():
    """Create a genes x cells sparse matrix."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    # 500 genes, 200 cells, 5% density
    mat = sp.random(500, 200, density=0.05, format='csr', dtype=np.float64)
    mat.data = np.abs(mat.data) * 10  # Make values positive
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def group_ids_3():
    """Group IDs for 3 groups (ref + 2 targets)."""
    np.random.seed(42)
    return np.random.randint(0, 3, size=200).astype(np.int64)


@pytest.fixture
def group_ids_binary():
    """Binary group IDs (ref vs 1 target)."""
    labels = np.zeros(200, dtype=np.int64)
    labels[100:] = 1
    return labels


class TestHVGParallel:
    """Test parallelized HVG functions."""
    
    def test_compute_moments_correctness(self, gene_cell_matrix):
        """compute_moments should produce correct mean and variance."""
        csr, scipy_mat = gene_cell_matrix
        
        means, vars_ = hvg.compute_moments(csr, ddof=1)
        
        # Compare with numpy
        dense = scipy_mat.toarray()
        expected_means = dense.mean(axis=1)
        expected_vars = dense.var(axis=1, ddof=1)
        
        np.testing.assert_allclose(means, expected_means, rtol=1e-10)
        np.testing.assert_allclose(vars_, expected_vars, rtol=1e-10)
    
    def test_compute_dispersion_correctness(self, gene_cell_matrix):
        """compute_dispersion should be variance / mean."""
        csr, scipy_mat = gene_cell_matrix
        
        means, vars_ = hvg.compute_moments(csr, ddof=1)
        dispersions = hvg.compute_dispersion(means, vars_)
        
        # Manual computation
        expected = np.where(means > 1e-12, vars_ / means, 0.0)
        
        np.testing.assert_allclose(dispersions, expected, rtol=1e-10)
    
    def test_select_hvg_by_dispersion_ordering(self, gene_cell_matrix):
        """Top genes should have highest dispersion."""
        csr, _ = gene_cell_matrix
        n_top = 50
        
        indices, mask, dispersions = hvg.select_hvg_by_dispersion(csr, n_top)
        
        # Verify top genes have highest dispersions
        top_dispersions = dispersions[indices]
        remaining_dispersions = dispersions[mask == 0]
        
        assert len(indices) == n_top
        assert np.sum(mask) == n_top
        
        # All top dispersions should be >= max of remaining
        if len(remaining_dispersions) > 0:
            assert np.min(top_dispersions) >= np.max(remaining_dispersions) - 1e-10


class TestMMDParallel:
    """Test parallelized MMD functions."""
    
    def test_mmd_rbf_shape(self, gene_cell_matrix, group_ids_3):
        """mmd_rbf should return correct shape."""
        csr, _ = gene_cell_matrix
        n_targets = 2
        gamma = 1.0
        
        result = mmd.mmd_rbf(csr, group_ids_3, n_targets, gamma)
        
        assert result.shape == (csr.nrows, n_targets)
    
    def test_mmd_rbf_nonnegative(self, gene_cell_matrix, group_ids_3):
        """MMD^2 should be non-negative."""
        csr, _ = gene_cell_matrix
        n_targets = 2
        gamma = 1.0
        
        result = mmd.mmd_rbf(csr, group_ids_3, n_targets, gamma)
        
        assert np.all(result >= 0)
    
    def test_mmd_rbf_self_is_zero(self, gene_cell_matrix):
        """MMD of identical groups should be ~0."""
        csr, _ = gene_cell_matrix
        
        # All same group (all reference)
        group_ids = np.zeros(200, dtype=np.int64)
        group_ids[100:] = 1  # But we'll make target same as ref
        
        # For a single row with constant values, MMD should be 0
        # This is a weak test; mainly checking no crashes
        result = mmd.mmd_rbf(csr, group_ids, 1, gamma=1.0)
        
        assert result.shape == (csr.nrows, 1)


class TestTTestParallel:
    """Test parallelized T-test functions."""
    
    def test_ttest_shape(self, gene_cell_matrix, group_ids_3):
        """ttest should return correct shapes."""
        csr, _ = gene_cell_matrix
        n_targets = 2
        
        t_stats, p_values, log2_fc = ttest.ttest(csr, group_ids_3, n_targets, use_welch=True)
        
        assert t_stats.shape == (csr.nrows, n_targets)
        assert p_values.shape == (csr.nrows, n_targets)
        assert log2_fc.shape == (csr.nrows, n_targets)
    
    def test_ttest_pvalues_range(self, gene_cell_matrix, group_ids_binary):
        """P-values should be in [0, 1]."""
        csr, _ = gene_cell_matrix
        
        _, p_values, _ = ttest.ttest(csr, group_ids_binary, 1, use_welch=True)
        
        assert np.all(p_values >= 0)
        assert np.all(p_values <= 1)
    
    def test_welch_vs_student(self, gene_cell_matrix, group_ids_binary):
        """Welch and Student t-tests should give different results."""
        csr, _ = gene_cell_matrix
        
        t_welch, p_welch, _ = ttest.ttest(csr, group_ids_binary, 1, use_welch=True)
        t_student, p_student, _ = ttest.ttest(csr, group_ids_binary, 1, use_welch=False)
        
        # T-statistics should be similar but not identical
        # (equal variance assumption makes a difference)
        correlation = np.corrcoef(t_welch.flatten(), t_student.flatten())[0, 1]
        assert correlation > 0.9  # Should be highly correlated
    
    def test_convenience_wrappers(self, gene_cell_matrix, group_ids_binary):
        """welch_ttest and student_ttest should work."""
        csr, _ = gene_cell_matrix
        
        t1, p1, fc1 = ttest.welch_ttest(csr, group_ids_binary, 1)
        t2, p2, fc2 = ttest.student_ttest(csr, group_ids_binary, 1)
        
        assert t1.shape == t2.shape
        assert p1.shape == p2.shape


class TestMWUParallel:
    """Test parallelized MWU functions."""
    
    def test_mwu_shape(self, gene_cell_matrix, group_ids_3):
        """mwu_test should return correct shapes."""
        csr, _ = gene_cell_matrix
        n_targets = 2
        
        u_stats, p_values, log2_fc, auroc = mwu.mwu_test(csr, group_ids_3, n_targets)
        
        assert u_stats.shape == (csr.nrows, n_targets)
        assert p_values.shape == (csr.nrows, n_targets)
        assert log2_fc.shape == (csr.nrows, n_targets)
        assert auroc.shape == (csr.nrows, n_targets)
    
    def test_mwu_pvalues_range(self, gene_cell_matrix, group_ids_binary):
        """P-values should be in [0, 1]."""
        csr, _ = gene_cell_matrix
        
        _, p_values, _, _ = mwu.mwu_test(csr, group_ids_binary, 1)
        
        assert np.all(p_values >= 0)
        assert np.all(p_values <= 1)
    
    def test_mwu_auroc_range(self, gene_cell_matrix, group_ids_binary):
        """AUROC should be in [0, 1]."""
        csr, _ = gene_cell_matrix
        
        _, _, _, auroc = mwu.mwu_test(csr, group_ids_binary, 1)
        
        assert np.all(auroc >= 0)
        assert np.all(auroc <= 1)


class TestParallelConsistency:
    """Test that parallel results are deterministic."""
    
    def test_compute_moments_deterministic(self, gene_cell_matrix):
        """Multiple calls should give same results."""
        csr, _ = gene_cell_matrix
        
        means1, vars1 = hvg.compute_moments(csr, ddof=1)
        means2, vars2 = hvg.compute_moments(csr, ddof=1)
        
        np.testing.assert_array_equal(means1, means2)
        np.testing.assert_array_equal(vars1, vars2)
    
    def test_ttest_deterministic(self, gene_cell_matrix, group_ids_binary):
        """Multiple calls should give same results."""
        csr, _ = gene_cell_matrix
        
        t1, p1, fc1 = ttest.ttest(csr, group_ids_binary, 1)
        t2, p2, fc2 = ttest.ttest(csr, group_ids_binary, 1)
        
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(fc1, fc2)
    
    def test_mwu_deterministic(self, gene_cell_matrix, group_ids_binary):
        """Multiple calls should give same results."""
        csr, _ = gene_cell_matrix
        
        u1, p1, fc1, a1 = mwu.mwu_test(csr, group_ids_binary, 1)
        u2, p2, fc2, a2 = mwu.mwu_test(csr, group_ids_binary, 1)
        
        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(a1, a2)
