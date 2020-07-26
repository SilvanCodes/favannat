use std::fmt;
use ndarray::{Array, Ix1};


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

pub trait Recurrent<N: NodeLike + fmt::Debug, E: EdgeLike + fmt::Debug> {
    type Net: NetLike<N,E>;
    fn unroll(&self) -> Self::Net;
    fn memory(&self) -> usize;
}

pub trait Evaluator<D=Ix1> {
    fn evaluate(&self, input: Array<f64, D>) -> Array<f64, D>;
}

pub trait StatefulEvaluator<D=Ix1> {
    fn evaluate(&mut self, input: Array<f64, D>) -> Array<f64, D>;
}

pub trait Fabricator<N: NodeLike + fmt::Debug, E: EdgeLike + fmt::Debug> {
    type Output: Evaluator;

    fn fabricate(net: &impl NetLike<N, E>) -> Result<Self::Output, &'static str>;
}

pub trait StatefulFabricator<N: NodeLike + fmt::Debug, E: EdgeLike + fmt::Debug> {
    type Output: StatefulEvaluator;

    fn fabricate(net: &impl Recurrent<N, E>) -> Result<Self::Output, &'static str>;
}
