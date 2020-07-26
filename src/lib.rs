#[macro_use]
pub mod network;
pub mod matrix;

type Matrix = Vec<Vec<f64>>;
type Transformations = Vec<fn(f64)->f64>;
