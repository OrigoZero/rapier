//! The scalar N-particle constraint shared by the soft-body element kinds (distance,
//! dihedral bending, cell volume), and the block constraint of the corotational elastic cells.

#[cfg(feature = "dim3")]
mod soft_elastic_batch;
mod soft_elastic_constraint;
mod soft_element_linalg;
mod soft_neo_hookean;
mod soft_scalar_constraint;
#[cfg(test)]
mod test;

#[cfg(feature = "dim3")]
pub(crate) use self::soft_elastic_batch::*;
pub(crate) use self::soft_elastic_constraint::*;
pub(crate) use self::soft_element_linalg::*;
pub(crate) use self::soft_neo_hookean::*;
pub(crate) use self::soft_scalar_constraint::*;
