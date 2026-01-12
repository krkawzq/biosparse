"""Pytest configuration for biosparse tests."""

import sys
import os

# Add src to path so biosparse package can be imported
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if _src not in sys.path:
    sys.path.insert(0, _src)

import pytest
import numpy as np

# =============================================================================
# Check available components
# =============================================================================

BINDING_AVAILABLE = False
OPTIM_AVAILABLE = False
NUMBA_AVAILABLE = False

# Try to import numba
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Try to import bindings (requires Rust library)
try:
    from biosparse._binding import CSRF64, CSCF64, SpanF64, lib
    BINDING_AVAILABLE = lib is not None
except Exception:
    BINDING_AVAILABLE = False

# Try to import optim (requires numba)
try:
    from biosparse.optim import disable_logging
    disable_logging()
    OPTIM_AVAILABLE = True
except Exception:
    OPTIM_AVAILABLE = False

# Check if _numba extension is available (requires both binding and numba)
NUMBA_EXT_AVAILABLE = BINDING_AVAILABLE and NUMBA_AVAILABLE
if NUMBA_EXT_AVAILABLE:
    try:
        import biosparse._numba
    except Exception:
        NUMBA_EXT_AVAILABLE = False


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "binding: tests requiring Rust FFI bindings")
    config.addinivalue_line("markers", "numba: tests requiring Numba")
    config.addinivalue_line("markers", "slow: slow tests")


# =============================================================================
# Fixtures - Arrays
# =============================================================================

@pytest.fixture
def arr():
    """Random float64 array (1000 elements)."""
    np.random.seed(42)
    return np.random.rand(1000)


@pytest.fixture
def small_arr():
    """Small array for quick tests (100 elements)."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture
def aligned_arr():
    """Array with size divisible by 8 (1024 elements)."""
    np.random.seed(42)
    return np.random.rand(1024)


# =============================================================================
# Fixtures - Sparse Matrices (only if bindings available)
# =============================================================================

@pytest.fixture
def scipy_csr():
    """Create a scipy CSR matrix."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    return sp.random(100, 50, density=0.1, format='csr', dtype=np.float64)


@pytest.fixture
def scipy_csc():
    """Create a scipy CSC matrix."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    return sp.random(100, 50, density=0.1, format='csc', dtype=np.float64)


@pytest.fixture
def csr_matrix(scipy_csr):
    """Create a biosparse CSRF64 from scipy."""
    if not BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    from biosparse._binding import CSRF64
    return CSRF64.from_scipy(scipy_csr)


@pytest.fixture
def csc_matrix(scipy_csc):
    """Create a biosparse CSCF64 from scipy."""
    if not BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    from biosparse._binding import CSCF64
    return CSCF64.from_scipy(scipy_csc)


# =============================================================================
# Fixtures - Larger Sparse Matrices for Performance Tests
# =============================================================================

@pytest.fixture
def large_scipy_csr():
    """Create a larger scipy CSR matrix for performance tests."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    return sp.random(1000, 500, density=0.05, format='csr', dtype=np.float64)


@pytest.fixture
def large_csr_matrix(large_scipy_csr):
    """Create a larger biosparse CSRF64 from scipy."""
    if not BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    from biosparse._binding import CSRF64
    return CSRF64.from_scipy(large_scipy_csr)


@pytest.fixture
def scipy_csr_f32():
    """Create a scipy CSR matrix with float32."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    np.random.seed(42)
    return sp.random(100, 50, density=0.1, format='csr', dtype=np.float32)


@pytest.fixture
def csr_matrix_f32(scipy_csr_f32):
    """Create a biosparse CSRF32 from scipy."""
    if not BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    from biosparse._binding import CSRF32
    return CSRF32.from_scipy(scipy_csr_f32)


# =============================================================================
# Fixtures - Group Labels for Statistical Tests
# =============================================================================

@pytest.fixture
def group_labels():
    """Group labels for statistical tests (3 groups)."""
    np.random.seed(42)
    return np.random.randint(0, 3, size=100).astype(np.int64)


@pytest.fixture
def binary_group_labels():
    """Binary group labels (0 = reference, 1 = target)."""
    np.random.seed(42)
    labels = np.zeros(100, dtype=np.int64)
    labels[50:] = 1
    return labels


# =============================================================================
# Skip Decorators
# =============================================================================

requires_binding = pytest.mark.skipif(
    not BINDING_AVAILABLE,
    reason="Rust FFI bindings not available"
)

requires_numba = pytest.mark.skipif(
    not NUMBA_AVAILABLE,
    reason="Numba not available"
)

requires_numba_ext = pytest.mark.skipif(
    not NUMBA_EXT_AVAILABLE,
    reason="Numba extension not available"
)

requires_optim = pytest.mark.skipif(
    not OPTIM_AVAILABLE,
    reason="Optim module not available"
)
