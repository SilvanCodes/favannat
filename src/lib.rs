pub mod looping;
pub mod matrix;
pub mod neat_original;
pub mod network;

type Matrix = Vec<Vec<f64>>;
type Transformations = Vec<fn(f64) -> f64>;
