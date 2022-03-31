use nalgebra::DMatrix;

use crate::network::{Evaluator, NetworkIO};

#[derive(Debug)]
pub struct MatrixFeedforwardEvaluator {
    pub stages: Vec<DMatrix<f64>>,
    pub transformations: Vec<crate::Transformations>,
}

impl Evaluator for MatrixFeedforwardEvaluator {
    fn evaluate<T: NetworkIO>(&self, state: T) -> T {
        let mut state = NetworkIO::input(state);
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (stage_matrix, transformations) in self.stages.iter().zip(&self.transformations) {
            state *= stage_matrix;
            for (value, activation) in state.iter_mut().zip(transformations) {
                *value = activation(*value);
            }
        }
        NetworkIO::output(state)
    }
}
