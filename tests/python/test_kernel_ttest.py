"""Tests for kernel.ttest module - T-test."""

import pytest
import numpy as np

# Check dependencies
try:
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from _binding import CSRF64
    import _numba
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
    # 100 genes, 200 cells, ~10% density
    mat = sp.random(100, 200, density=0.1, format='csr', dtype=np.float64)
    mat.data = np.abs(mat.data) * 10  # Make values positive
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def group_ids_two():
    """Group IDs for two-group test (100 ref, 100 target)."""
    group_ids = np.zeros(200, dtype=np.int32)
    group_ids[100:] = 1
    return group_ids


@pytest.fixture
def group_ids_multi():
    """Group IDs for multi-group test (50 ref, 50 each for 3 targets)."""
    group_ids = np.zeros(200, dtype=np.int32)
    group_ids[50:100] = 1
    group_ids[100:150] = 2
    group_ids[150:200] = 3
    return group_ids


class TestWelchTTest:
    """Test Welch's t-test."""
    
    def test_welch_basic(self, gene_expression_matrix, group_ids_two):
        """Welch's t-test should produce valid results."""
        from kernel.ttest import welch_ttest
        
        csr, scipy_mat = gene_expression_matrix
        
        t_stats, p_values, log2_fc = welch_ttest(csr, group_ids_two, 1)
        
        # Check shapes
        assert t_stats.shape == (100, 1)
        assert p_values.shape == (100, 1)
        assert log2_fc.shape == (100, 1)
        
        # P-values should be in [0, 1]
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)
    
    def test_welch_vs_scipy(self, gene_expression_matrix, group_ids_two):
        """Welch's t-test should match scipy.stats.ttest_ind."""
        from kernel.ttest import welch_ttest
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        t_stats, p_values, log2_fc = welch_ttest(csr, group_ids_two, 1)
        
        # Test first 10 genes
        for gene_idx in range(10):
            row = dense[gene_idx, :]
            ref_vals = row[group_ids_two == 0]
            tar_vals = row[group_ids_two == 1]
            
            # scipy Welch's t-test
            scipy_result = scipy.stats.ttest_ind(
                tar_vals, ref_vals,  # target - ref for positive = upregulated
                equal_var=False  # Welch's
            )
            
            our_t = t_stats[gene_idx, 0]
            scipy_t = scipy_result.statistic
            
            our_p = p_values[gene_idx, 0]
            scipy_p = scipy_result.pvalue
            
            # T-statistics should be close
            np.testing.assert_allclose(our_t, scipy_t, rtol=0.05, atol=1e-6,
                err_msg=f"Gene {gene_idx}: t-stat mismatch")
            
            # P-values should be close
            np.testing.assert_allclose(our_p, scipy_p, rtol=0.1, atol=1e-6,
                err_msg=f"Gene {gene_idx}: p-value mismatch")
    
    def test_welch_log2fc(self, gene_expression_matrix, group_ids_two):
        """Log2FC should be computed correctly."""
        from kernel.ttest import welch_ttest
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        t_stats, p_values, log2_fc = welch_ttest(csr, group_ids_two, 1)
        
        EPS = 1e-9
        
        for gene_idx in range(10):
            row = dense[gene_idx, :]
            ref_vals = row[group_ids_two == 0]
            tar_vals = row[group_ids_two == 1]
            
            mean_ref = np.mean(ref_vals)
            mean_tar = np.mean(tar_vals)
            expected_fc = np.log2((mean_tar + EPS) / (mean_ref + EPS))
            
            np.testing.assert_allclose(log2_fc[gene_idx, 0], expected_fc, rtol=1e-10)


class TestStudentTTest:
    """Test Student's t-test (equal variance)."""
    
    def test_student_basic(self, gene_expression_matrix, group_ids_two):
        """Student's t-test should produce valid results."""
        from kernel.ttest import student_ttest
        
        csr, scipy_mat = gene_expression_matrix
        
        t_stats, p_values, log2_fc = student_ttest(csr, group_ids_two, 1)
        
        # Check shapes
        assert t_stats.shape == (100, 1)
        assert p_values.shape == (100, 1)
        assert log2_fc.shape == (100, 1)
        
        # P-values should be in [0, 1]
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)
    
    def test_student_vs_scipy(self, gene_expression_matrix, group_ids_two):
        """Student's t-test should match scipy.stats.ttest_ind with equal_var=True."""
        from kernel.ttest import student_ttest
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        t_stats, p_values, log2_fc = student_ttest(csr, group_ids_two, 1)
        
        # Test first 10 genes
        for gene_idx in range(10):
            row = dense[gene_idx, :]
            ref_vals = row[group_ids_two == 0]
            tar_vals = row[group_ids_two == 1]
            
            # scipy Student's t-test
            scipy_result = scipy.stats.ttest_ind(
                tar_vals, ref_vals,
                equal_var=True  # Student's
            )
            
            our_t = t_stats[gene_idx, 0]
            scipy_t = scipy_result.statistic
            
            our_p = p_values[gene_idx, 0]
            scipy_p = scipy_result.pvalue
            
            # T-statistics should be close
            np.testing.assert_allclose(our_t, scipy_t, rtol=0.05, atol=1e-6,
                err_msg=f"Gene {gene_idx}: t-stat mismatch")
            
            # P-values should be close
            np.testing.assert_allclose(our_p, scipy_p, rtol=0.1, atol=1e-6,
                err_msg=f"Gene {gene_idx}: p-value mismatch")


class TestTTestMultiGroup:
    """Test t-test with multiple target groups."""
    
    def test_welch_multi_basic(self, gene_expression_matrix, group_ids_multi):
        """Multi-target Welch's t-test should produce valid results."""
        from kernel.ttest import welch_ttest
        
        csr, scipy_mat = gene_expression_matrix
        
        t_stats, p_values, log2_fc = welch_ttest(csr, group_ids_multi, 3)
        
        # Check shapes: (n_genes, n_targets)
        assert t_stats.shape == (100, 3)
        assert p_values.shape == (100, 3)
        assert log2_fc.shape == (100, 3)
        
        # P-values should be in [0, 1]
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)
    
    def test_welch_multi_vs_scipy(self, gene_expression_matrix, group_ids_multi):
        """Multi-target Welch should match scipy for each target."""
        from kernel.ttest import welch_ttest
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        t_stats, p_values, log2_fc = welch_ttest(csr, group_ids_multi, 3)
        
        # Test first 5 genes, all 3 targets
        for gene_idx in range(5):
            row = dense[gene_idx, :]
            ref_vals = row[group_ids_multi == 0]
            
            for target_idx in range(3):
                tar_vals = row[group_ids_multi == (target_idx + 1)]
                
                scipy_result = scipy.stats.ttest_ind(
                    tar_vals, ref_vals,
                    equal_var=False
                )
                
                our_t = t_stats[gene_idx, target_idx]
                scipy_t = scipy_result.statistic
                
                our_p = p_values[gene_idx, target_idx]
                scipy_p = scipy_result.pvalue
                
                np.testing.assert_allclose(our_t, scipy_t, rtol=0.1, atol=1e-6,
                    err_msg=f"Gene {gene_idx}, Target {target_idx}: t-stat mismatch")
                
                np.testing.assert_allclose(our_p, scipy_p, rtol=0.15, atol=1e-6,
                    err_msg=f"Gene {gene_idx}, Target {target_idx}: p-value mismatch")


class TestTTestInterface:
    """Test ttest function interface."""
    
    def test_ttest_welch_flag(self, gene_expression_matrix, group_ids_two):
        """ttest with use_welch=True should match welch_ttest."""
        from kernel.ttest import ttest, welch_ttest
        
        csr, scipy_mat = gene_expression_matrix
        
        t1, p1, fc1 = ttest(csr, group_ids_two, 1, use_welch=True)
        t2, p2, fc2 = welch_ttest(csr, group_ids_two, 1)
        
        np.testing.assert_allclose(t1, t2)
        np.testing.assert_allclose(p1, p2)
        np.testing.assert_allclose(fc1, fc2)
    
    def test_ttest_student_flag(self, gene_expression_matrix, group_ids_two):
        """ttest with use_welch=False should match student_ttest."""
        from kernel.ttest import ttest, student_ttest
        
        csr, scipy_mat = gene_expression_matrix
        
        t1, p1, fc1 = ttest(csr, group_ids_two, 1, use_welch=False)
        t2, p2, fc2 = student_ttest(csr, group_ids_two, 1)
        
        np.testing.assert_allclose(t1, t2)
        np.testing.assert_allclose(p1, p2)
        np.testing.assert_allclose(fc1, fc2)
