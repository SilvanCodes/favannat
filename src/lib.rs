pub mod matrix;
pub mod neat_original;
pub mod network;
pub mod sparse_matrix;

type Matrix = Vec<Vec<f64>>;
type Transformations = Vec<fn(f64) -> f64>;
