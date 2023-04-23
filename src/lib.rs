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

pub use matrix::{
    feedforward::{evaluator::MatrixFeedforwardEvaluator, fabricator::MatrixFeedforwardFabricator},
    recurrent::{evaluator::MatrixRecurrentEvaluator, fabricator::MatrixRecurrentFabricator},
};

pub use sparse_matrix::{
    feedforward::{
        evaluator::SparseMatrixFeedforwardEvaluator, fabricator::SparseMatrixFeedforwardFabricator,
    },
    recurrent::{
        evaluator::SparseMatrixRecurrentEvaluator, fabricator::SparseMatrixRecurrentFabricator,
    },
};

pub use network::{Evaluator, Fabricator, StatefulEvaluator, StatefulFabricator};

type Matrix = Vec<Vec<f64>>;
type Transformations = Vec<fn(f64) -> f64>;
