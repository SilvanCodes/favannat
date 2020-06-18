use std::fmt;
use ndarray::{Array, Ix1};

pub mod activations {
    pub const LINEAR: fn(f64) -> f64 = |val| val;
    pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-1.0 * val).exp());
    pub const TANH: fn(f64) -> f64 = |val| 2.0 * SIGMOID(2.0 * val) - 1.0;
    pub const GAUSSIAN: fn(f64) -> f64 = |val| (val * val / -2.0).exp(); // a = 1, b = 0, c = 1
}

pub trait NodeLike {
    fn id(&self) -> usize;
    fn activation(&self) -> fn(f64) -> f64;
}

pub trait EdgeLike {
    fn start(&self) -> usize;
    fn end(&self) -> usize;
    fn weight(&self) -> f64;
}

pub trait NetLike<N: NodeLike + fmt::Debug, E: EdgeLike + fmt::Debug>: fmt::Debug {
    fn nodes(&self) -> Vec<&N>;
    fn edges(&self) -> Vec<&E>;
    fn inputs(&self) -> Vec<&N>;
    fn outputs(&self) -> Vec<&N>;
}

pub trait Evaluator<D=Ix1> {
    fn evaluate(&self, input: Array<f64, D>) -> Array<f64, D>;
}

pub trait Fabricator<N: NodeLike + fmt::Debug, E: EdgeLike + fmt::Debug> {
    type Output: Evaluator;

    fn fabricate(net: &impl NetLike<N, E>) -> Result<Self::Output, &'static str>;
}
