use ndarray::Array1;

pub trait NodeLike: Ord {
    fn id(&self) -> usize;
    fn activation(&self) -> fn(f64) -> f64;
}

pub trait EdgeLike {
    fn start(&self) -> usize;
    fn end(&self) -> usize;
    fn weight(&self) -> f64;
}

pub trait NetLike<N: NodeLike, E: EdgeLike> {
    fn nodes(&self) -> Vec<&N>;
    fn edges(&self) -> Vec<&E>;
    fn inputs(&self) -> Vec<&N>;
    fn outputs(&self) -> Vec<&N>;
}

pub trait Recurrent<N: NodeLike, E: EdgeLike>: NetLike<N, E> {
    type Net: NetLike<N, E>;
    fn unroll(&self) -> Self::Net;
    fn recurrent_edges(&self) -> Vec<&E>;
}

pub trait Evaluator {
    fn evaluate(&self, input: Array1<f64>) -> Array1<f64>;
}

pub trait StatefulEvaluator {
    fn evaluate(&mut self, input: Array1<f64>) -> Array1<f64>;
    fn reset_internal_state(&mut self);
}

pub trait Fabricator<N: NodeLike, E: EdgeLike> {
    type Output: Evaluator;

    fn fabricate(net: &impl NetLike<N, E>) -> Result<Self::Output, &'static str>;
}

pub trait StatefulFabricator<N: NodeLike, E: EdgeLike> {
    type Output: StatefulEvaluator;

    fn fabricate(net: &impl Recurrent<N, E>) -> Result<Self::Output, &'static str>;
}
