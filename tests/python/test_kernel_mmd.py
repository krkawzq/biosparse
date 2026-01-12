"""Tests for kernel.mmd module - Maximum Mean Discrepancy."""

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


def compute_mmd_rbf_dense(x_vals, y_vals, gamma):
    """Reference implementation of MMD^2 with RBF kernel on dense arrays."""
    n_x = len(x_vals)
    n_y = len(y_vals)
    
    # Self kernel X
    sum_xx = 0.0
    for i in range(n_x):
        for j in range(n_x):
            diff = x_vals[i] - x_vals[j]
            sum_xx += np.exp(-gamma * diff * diff)
    
    # Self kernel Y
    sum_yy = 0.0
    for i in range(n_y):
        for j in range(n_y):
            diff = y_vals[i] - y_vals[j]
            sum_yy += np.exp(-gamma * diff * diff)
    
    # Cross kernel
    sum_xy = 0.0
    for i in range(n_x):
        for j in range(n_y):
            diff = x_vals[i] - y_vals[j]
            sum_xy += np.exp(-gamma * diff * diff)
    
    mmd2 = sum_xx / (n_x * n_x) + sum_yy / (n_y * n_y) - 2.0 * sum_xy / (n_x * n_y)
    return max(0.0, mmd2)


@pytest.fixture
def feature_matrix():
    """Create a feature CSR matrix (features x samples)."""
    import scipy.sparse as sp
    np.random.seed(42)
    # 50 features, 100 samples, ~20% density
    mat = sp.random(50, 100, density=0.2, format='csr', dtype=np.float64)
    mat.data = mat.data * 5  # Scale values
    return CSRF64.from_scipy(mat), mat


@pytest.fixture
def group_ids_two():
    """Group IDs for two-group test (50 ref, 50 target)."""
    group_ids = np.zeros(100, dtype=np.int32)
    group_ids[50:] = 1
    return group_ids


@pytest.fixture
def group_ids_multi():
    """Group IDs for multi-group test."""
    group_ids = np.zeros(100, dtype=np.int32)
    group_ids[25:50] = 1
    group_ids[50:75] = 2
    group_ids[75:100] = 3
    return group_ids


class TestMMDRBF:
    """Test MMD with RBF kernel."""
    
    def test_mmd_basic(self, feature_matrix, group_ids_two):
        """MMD should produce valid results."""
        from biosparse.kernel.mmd import mmd_rbf
        
        csr, scipy_mat = feature_matrix
        
        gamma = 1.0
        mmd2 = mmd_rbf(csr, group_ids_two, 1, gamma)
        
        # Check shape
        assert mmd2.shape == (50, 1)
        
        # MMD^2 should be non-negative
        assert np.all(mmd2 >= 0.0)
    
    def test_mmd_vs_reference(self, feature_matrix, group_ids_two):
        """MMD should match reference implementation."""
        from biosparse.kernel.mmd import mmd_rbf
        
        csr, scipy_mat = feature_matrix
        dense = scipy_mat.toarray()
        
        gamma = 0.5
        mmd2 = mmd_rbf(csr, group_ids_two, 1, gamma)
        
        # Test first 10 features
        for feat_idx in range(10):
            row = dense[feat_idx, :]
            ref_vals = row[group_ids_two == 0]
            tar_vals = row[group_ids_two == 1]
            
            expected_mmd2 = compute_mmd_rbf_dense(ref_vals, tar_vals, gamma)
            
            np.testing.assert_allclose(mmd2[feat_idx, 0], expected_mmd2, rtol=1e-10,
                err_msg=f"Feature {feat_idx}: MMD^2 mismatch")
    
    def test_mmd_same_distribution(self):
        """MMD between identical distributions should be ~0."""
        from biosparse.kernel.mmd import mmd_rbf
        
        import scipy.sparse as sp
        
        np.random.seed(42)
        # Create matrix where all samples come from same distribution
        data = np.random.randn(1, 100)  # Same distribution
        mat = sp.csr_matrix(data)
        csr = CSRF64.from_scipy(mat)
        
        group_ids = np.zeros(100, dtype=np.int32)
        group_ids[50:] = 1
        
        gamma = 1.0
        mmd2 = mmd_rbf(csr, group_ids, 1, gamma)
        
        # MMD^2 should be small (samples from same distribution)
        # Not exactly 0 due to finite sample variance
        assert mmd2[0, 0] < 0.5  # Loose bound
    
    def test_mmd_different_distributions(self):
        """MMD between different distributions should be large."""
        from biosparse.kernel.mmd import mmd_rbf
        
        import scipy.sparse as sp
        
        np.random.seed(42)
        # Create matrix with very different groups
        data = np.zeros((1, 100))
        data[0, :50] = 0.0  # Group 0: all zeros
        data[0, 50:] = 10.0  # Group 1: all tens
        
        mat = sp.csr_matrix(data)
        csr = CSRF64.from_scipy(mat)
        
        group_ids = np.zeros(100, dtype=np.int32)
        group_ids[50:] = 1
        
        gamma = 0.1
        mmd2 = mmd_rbf(csr, group_ids, 1, gamma)
        
        # MMD^2 should be large
        assert mmd2[0, 0] > 0.5


class TestMMDMultiGroup:
    """Test MMD with multiple target groups."""
    
    def test_mmd_multi_basic(self, feature_matrix, group_ids_multi):
        """Multi-target MMD should produce valid results."""
        from biosparse.kernel.mmd import mmd_rbf
        
        csr, scipy_mat = feature_matrix
        
        gamma = 1.0
        mmd2 = mmd_rbf(csr, group_ids_multi, 3, gamma)
        
        # Check shape: (n_features, n_targets)
        assert mmd2.shape == (50, 3)
        
        # MMD^2 should be non-negative
        assert np.all(mmd2 >= 0.0)
    
    def test_mmd_multi_vs_reference(self, feature_matrix, group_ids_multi):
        """Multi-target MMD should match reference for each target."""
        from biosparse.kernel.mmd import mmd_rbf
        
        csr, scipy_mat = feature_matrix
        dense = scipy_mat.toarray()
        
        gamma = 0.5
        mmd2 = mmd_rbf(csr, group_ids_multi, 3, gamma)
        
        # Test first 5 features, all 3 targets
        for feat_idx in range(5):
            row = dense[feat_idx, :]
            ref_vals = row[group_ids_multi == 0]
            
            for target_idx in range(3):
                tar_vals = row[group_ids_multi == (target_idx + 1)]
                
                expected_mmd2 = compute_mmd_rbf_dense(ref_vals, tar_vals, gamma)
                
                np.testing.assert_allclose(mmd2[feat_idx, target_idx], expected_mmd2, rtol=1e-10,
                    err_msg=f"Feature {feat_idx}, Target {target_idx}: MMD^2 mismatch")


class TestMMDGammaParameter:
    """Test MMD with different gamma values."""
    
    def test_mmd_gamma_effect(self, feature_matrix, group_ids_two):
        """Different gamma values should produce different results."""
        from biosparse.kernel.mmd import mmd_rbf
        
        csr, scipy_mat = feature_matrix
        
        mmd_low_gamma = mmd_rbf(csr, group_ids_two, 1, 0.01)
        mmd_high_gamma = mmd_rbf(csr, group_ids_two, 1, 10.0)
        
        # Results should differ
        assert not np.allclose(mmd_low_gamma, mmd_high_gamma)
    
    def test_mmd_gamma_zero_like(self, feature_matrix, group_ids_two):
        """Very small gamma should make all kernels ~1, MMD ~0."""
        from biosparse.kernel.mmd import mmd_rbf
        
        csr, scipy_mat = feature_matrix
        
        # Very small gamma -> exp(-gamma * d^2) ≈ 1 for all d
        mmd2 = mmd_rbf(csr, group_ids_two, 1, 1e-10)
        
        # MMD should be very small
        assert np.all(mmd2 < 1e-5)
