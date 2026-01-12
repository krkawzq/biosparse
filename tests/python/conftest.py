"""Pytest configuration for biosparse tests."""

import sys
import os

# Add src/python to path as biosparse package
_src_python = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'python'))
if _src_python not in sys.path:
    sys.path.insert(0, _src_python)

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
    from _binding import CSRF64, CSCF64, SpanF64, lib
    BINDING_AVAILABLE = lib is not None
except Exception:
    BINDING_AVAILABLE = False

# Try to import optim (requires numba)
try:
    from optim import disable_logging
    disable_logging()
    OPTIM_AVAILABLE = True
except Exception:
    OPTIM_AVAILABLE = False

# Check if _numba extension is available (requires both binding and numba)
NUMBA_EXT_AVAILABLE = BINDING_AVAILABLE and NUMBA_AVAILABLE
if NUMBA_EXT_AVAILABLE:
    try:
        import _numba
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
    from _binding import CSRF64
    return CSRF64.from_scipy(scipy_csr)


@pytest.fixture
def csc_matrix(scipy_csc):
    """Create a biosparse CSCF64 from scipy."""
    if not BINDING_AVAILABLE:
        pytest.skip("Rust FFI bindings not available")
    from _binding import CSCF64
    return CSCF64.from_scipy(scipy_csc)
