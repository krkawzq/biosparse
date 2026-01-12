"""Numba type definitions for CSR/CSC sparse matrices.

This module defines the complete type system for Numba JIT compilation of
sparse matrix operations.
"""

from numba import types
from numba.extending import typeof_impl

# Import the Python classes for type registration
# Use absolute import to work with flat path setup in tests
try:
    from .._binding._sparse import CSR, CSC, CSRF32, CSRF64, CSCF32, CSCF64
except ImportError:
    from _binding._sparse import CSR, CSC, CSRF32, CSRF64, CSCF32, CSCF64


# =============================================================================
# Forward declaration of iterator types (defined at end to avoid circular import)
# =============================================================================

class CSRIteratorType(types.SimpleIteratorType):
    """Iterator type for CSR row iteration.
    
    This iterator yields (values, indices) tuples for each row.
    """
    pass


class CSCIteratorType(types.SimpleIteratorType):
    """Iterator type for CSC column iteration.
    
    This iterator yields (values, indices) tuples for each column.
    """
    pass


# =============================================================================
# CSR Type
# =============================================================================

class CSRType(types.IterableType):
    """Numba type for CSR sparse matrix.
    
    This type represents a CSR (Compressed Sparse Row) matrix in Numba's
    type system. It supports iteration over rows.
    
    Attributes:
        dtype: The data type of matrix values (float32 or float64)
    """
    
    def __init__(self, dtype):
        """Initialize CSR type.
        
        Args:
            dtype: Numba type for matrix values (types.float32 or types.float64)
        """
        self.dtype = dtype
        super().__init__(name=f'CSR[{dtype}]')
    
    @property
    def key(self):
        """Unique key for type identity."""
        return ('CSRType', self.dtype)
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        if isinstance(other, CSRType):
            return self.dtype == other.dtype
        return False
    
    @property
    def iterator_type(self):
        """Return the iterator type for this CSR type."""
        return CSRIteratorType(self)
    
    @property
    def ffi_prefix(self):
        """Return the FFI function prefix for this type.
        
        Returns:
            'csr_f32' for float32, 'csr_f64' for float64
        """
        if self.dtype == types.float32:
            return 'csr_f32'
        elif self.dtype == types.float64:
            return 'csr_f64'
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")


class CSCType(types.IterableType):
    """Numba type for CSC sparse matrix.
    
    This type represents a CSC (Compressed Sparse Column) matrix in Numba's
    type system. It supports iteration over columns.
    
    Attributes:
        dtype: The data type of matrix values (float32 or float64)
    """
    
    def __init__(self, dtype):
        """Initialize CSC type.
        
        Args:
            dtype: Numba type for matrix values (types.float32 or types.float64)
        """
        self.dtype = dtype
        super().__init__(name=f'CSC[{dtype}]')
    
    @property
    def key(self):
        """Unique key for type identity."""
        return ('CSCType', self.dtype)
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        if isinstance(other, CSCType):
            return self.dtype == other.dtype
        return False
    
    @property
    def iterator_type(self):
        """Return the iterator type for this CSC type."""
        return CSCIteratorType(self)
    
    @property
    def ffi_prefix(self):
        """Return the FFI function prefix for this type.
        
        Returns:
            'csc_f32' for float32, 'csc_f64' for float64
        """
        if self.dtype == types.float32:
            return 'csc_f32'
        elif self.dtype == types.float64:
            return 'csc_f64'
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")


# =============================================================================
# Concrete type instances
# =============================================================================

CSRFloat32Type = CSRType(types.float32)
CSRFloat64Type = CSRType(types.float64)
CSCFloat32Type = CSCType(types.float32)
CSCFloat64Type = CSCType(types.float64)


# =============================================================================
# Type Inference: Python object -> Numba type
# =============================================================================

@typeof_impl.register(CSRF32)
def typeof_csrf32(val, c):
    """Infer Numba type for CSRF32 Python object."""
    return CSRFloat32Type


@typeof_impl.register(CSRF64)
def typeof_csrf64(val, c):
    """Infer Numba type for CSRF64 Python object."""
    return CSRFloat64Type


@typeof_impl.register(CSCF32)
def typeof_cscf32(val, c):
    """Infer Numba type for CSCF32 Python object."""
    return CSCFloat32Type


@typeof_impl.register(CSCF64)
def typeof_cscf64(val, c):
    """Infer Numba type for CSCF64 Python object."""
    return CSCFloat64Type


# =============================================================================
# Iterator Type Implementation (complete definition)
# =============================================================================

def _init_csr_iterator_type(self, csr_type):
    """Initialize CSR iterator type."""
    self.csr_type = csr_type
    # Each iteration yields (values_array, indices_array)
    yield_type = types.Tuple([
        types.Array(csr_type.dtype, 1, 'C'),
        types.Array(types.int64, 1, 'C'),
    ])
    types.SimpleIteratorType.__init__(
        self, 
        name=f'CSRIterator[{csr_type.dtype}]',
        yield_type=yield_type
    )


def _init_csc_iterator_type(self, csc_type):
    """Initialize CSC iterator type."""
    self.csc_type = csc_type
    # Each iteration yields (values_array, indices_array)
    yield_type = types.Tuple([
        types.Array(csc_type.dtype, 1, 'C'),
        types.Array(types.int64, 1, 'C'),
    ])
    types.SimpleIteratorType.__init__(
        self,
        name=f'CSCIterator[{csc_type.dtype}]',
        yield_type=yield_type
    )


# Apply the initializers
CSRIteratorType.__init__ = _init_csr_iterator_type
CSCIteratorType.__init__ = _init_csc_iterator_type
