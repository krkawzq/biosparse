"""Tests for kernel.mwu module - Mann-Whitney U test."""

import pytest
import numpy as np

# Check dependencies
try:
    import scipy.stats
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
    # 100 genes, 200 cells, ~10% density
    mat = sp.random(100, 200, density=0.1, format='csr', dtype=np.float64)
    mat.data = np.abs(mat.data) * 10  # Make values positive (expression-like)
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def group_ids_two():
    """Group IDs for two-group test (100 ref, 100 target)."""
    np.random.seed(42)
    group_ids = np.zeros(200, dtype=np.int32)
    group_ids[100:] = 1  # Second half is target
    return group_ids


@pytest.fixture
def group_ids_multi():
    """Group IDs for multi-group test (50 ref, 50 each for 3 targets)."""
    np.random.seed(42)
    group_ids = np.zeros(200, dtype=np.int32)
    group_ids[50:100] = 1   # Target 1
    group_ids[100:150] = 2  # Target 2
    group_ids[150:200] = 3  # Target 3
    return group_ids


class TestMWUTwoGroups:
    """Test MWU with two groups (ref vs single target)."""
    
    def test_mwu_basic(self, gene_expression_matrix, group_ids_two):
        """MWU should produce valid results."""
        from biosparse.kernel.mwu import mwu_test
        
        csr, scipy_mat = gene_expression_matrix
        
        u_stats, p_values, log2_fc, auroc = mwu_test(csr, group_ids_two, 1)
        
        # Check shapes
        assert u_stats.shape == (100, 1)
        assert p_values.shape == (100, 1)
        assert log2_fc.shape == (100, 1)
        assert auroc.shape == (100, 1)
        
        # P-values should be in [0, 1]
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)
        
        # AUROC should be in [0, 1]
        assert np.all(auroc >= 0.0)
        assert np.all(auroc <= 1.0)
    
    def test_mwu_vs_scipy(self, gene_expression_matrix, group_ids_two):
        """MWU results should match scipy.stats.mannwhitneyu."""
        from biosparse.kernel.mwu import mwu_test
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        u_stats, p_values, log2_fc, auroc = mwu_test(csr, group_ids_two, 1)
        
        # Test first 10 genes for speed
        for gene_idx in range(10):
            row = dense[gene_idx, :]
            ref_vals = row[group_ids_two == 0]
            tar_vals = row[group_ids_two == 1]
            
            # scipy MWU
            scipy_result = scipy.stats.mannwhitneyu(
                ref_vals, tar_vals, 
                alternative='two-sided',
                use_continuity=True
            )
            
            # Compare U-statistics (scipy returns min(U1, U2) or U1 depending on version)
            # Our implementation returns U1 (rank sum of ref - correction)
            our_u = u_stats[gene_idx, 0]
            
            # For p-values, check relative tolerance
            our_p = p_values[gene_idx, 0]
            scipy_p = scipy_result.pvalue
            
            # Allow some tolerance due to different tie correction methods
            np.testing.assert_allclose(our_p, scipy_p, rtol=0.1, atol=1e-6,
                err_msg=f"Gene {gene_idx}: p-value mismatch")
    
    def test_mwu_log2fc(self, gene_expression_matrix, group_ids_two):
        """Log2FC should be computed correctly."""
        from biosparse.kernel.mwu import mwu_test
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        u_stats, p_values, log2_fc, auroc = mwu_test(csr, group_ids_two, 1)
        
        EPS = 1e-9
        
        for gene_idx in range(10):
            row = dense[gene_idx, :]
            ref_vals = row[group_ids_two == 0]
            tar_vals = row[group_ids_two == 1]
            
            mean_ref = np.mean(ref_vals)
            mean_tar = np.mean(tar_vals)
            expected_fc = np.log2((mean_tar + EPS) / (mean_ref + EPS))
            
            np.testing.assert_allclose(log2_fc[gene_idx, 0], expected_fc, rtol=1e-10)


class TestMWUMultiGroup:
    """Test MWU with multiple target groups."""
    
    def test_mwu_multi_basic(self, gene_expression_matrix, group_ids_multi):
        """MWU with multiple targets should produce valid results."""
        from biosparse.kernel.mwu import mwu_test
        
        csr, scipy_mat = gene_expression_matrix
        
        u_stats, p_values, log2_fc, auroc = mwu_test(csr, group_ids_multi, 3)
        
        # Check shapes: (n_genes, n_targets)
        assert u_stats.shape == (100, 3)
        assert p_values.shape == (100, 3)
        assert log2_fc.shape == (100, 3)
        assert auroc.shape == (100, 3)
        
        # P-values should be in [0, 1]
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)
    
    def test_mwu_multi_vs_scipy(self, gene_expression_matrix, group_ids_multi):
        """Multi-target MWU should match scipy for each target."""
        from biosparse.kernel.mwu import mwu_test
        
        csr, scipy_mat = gene_expression_matrix
        dense = scipy_mat.toarray()
        
        u_stats, p_values, log2_fc, auroc = mwu_test(csr, group_ids_multi, 3)
        
        # Test first 5 genes, all 3 targets
        for gene_idx in range(5):
            row = dense[gene_idx, :]
            ref_vals = row[group_ids_multi == 0]
            
            for target_idx in range(3):
                tar_vals = row[group_ids_multi == (target_idx + 1)]
                
                scipy_result = scipy.stats.mannwhitneyu(
                    ref_vals, tar_vals,
                    alternative='two-sided',
                    use_continuity=True
                )
                
                our_p = p_values[gene_idx, target_idx]
                scipy_p = scipy_result.pvalue
                
                np.testing.assert_allclose(our_p, scipy_p, rtol=0.15, atol=1e-6,
                    err_msg=f"Gene {gene_idx}, Target {target_idx}: p-value mismatch")


class TestCountGroups:
    """Test count_groups function."""
    
    def test_count_groups_basic(self):
        """count_groups should count correctly."""
        from biosparse.kernel.mwu import count_groups
        
        group_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        counts = count_groups(group_ids, 3)
        
        np.testing.assert_array_equal(counts, [2, 3, 4])
    
    def test_count_groups_single(self):
        """count_groups with single group."""
        from biosparse.kernel.mwu import count_groups
        
        group_ids = np.zeros(100, dtype=np.int32)
        counts = count_groups(group_ids, 1)
        
        np.testing.assert_array_equal(counts, [100])
