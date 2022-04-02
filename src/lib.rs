//! This crate allows to evaluate anything that implements the [`network::NetworkLike`] trait.
//!
//! See [`network::net`] for an examplatory implementation.
//!
//! Networks accept any value that implements the [`network::NetworkIO`] trait.
//!
//! The feature `ndarray` implements `NetworkIO` from `ndarray::Array1` when enabled.

pub mod matrix;
pub mod neat_original;
pub mod network;
pub mod sparse_matrix;

type Matrix = Vec<Vec<f64>>;
type Transformations = Vec<fn(f64) -> f64>;
