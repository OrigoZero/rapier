//! ComponentMul trait for element-wise vector operations.

use crate::math::Vector;
#[cfg(not(target_arch = "spirv"))]
use crate::utils::SimdRealCopy;
#[cfg(not(target_arch = "spirv"))]
use na::{Vector2, Vector3};

/// Extension trait for element-wise vector operations
pub trait ComponentMul: Sized {
    /// Element-wise multiplication
    fn component_mul(&self, other: &Self) -> Self;
    /// One where the component is nonzero and zero where it is zero.
    fn component_nonzero(&self) -> Self;
}

impl ComponentMul for Vector {
    #[inline]
    fn component_mul(&self, other: &Self) -> Self {
        *self * *other
    }

    #[inline]
    fn component_nonzero(&self) -> Self {
        Vector::select(self.cmpne(Vector::ZERO), Vector::ONE, Vector::ZERO)
    }
}

// Nalgebra vector implementations for SIMD support
#[cfg(not(target_arch = "spirv"))]
impl<N: SimdRealCopy> ComponentMul for Vector2<N> {
    #[inline]
    fn component_mul(&self, other: &Self) -> Self {
        nalgebra::Vector2::component_mul(self, other)
    }

    #[inline]
    fn component_nonzero(&self) -> Self {
        self.map(|x| N::one().select(x.simd_ne(N::zero()), N::zero()))
    }
}

#[cfg(not(target_arch = "spirv"))]
impl<N: SimdRealCopy> ComponentMul for Vector3<N> {
    #[inline]
    fn component_mul(&self, other: &Self) -> Self {
        nalgebra::Vector3::component_mul(self, other)
    }

    #[inline]
    fn component_nonzero(&self) -> Self {
        self.map(|x| N::one().select(x.simd_ne(N::zero()), N::zero()))
    }
}
