use nalgebra::DMatrix;

use crate::network::Evaluator;

#[derive(Debug)]
pub struct MatrixFeedforwardEvaluator {
    pub stages: Vec<DMatrix<f64>>,
    pub transformations: Vec<crate::Transformations>,
}

impl Evaluator for MatrixFeedforwardEvaluator {
    fn evaluate(&self, mut state: DMatrix<f64>) -> DMatrix<f64> {
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (stage_matrix, transformations) in self.stages.iter().zip(&self.transformations) {
            state *= stage_matrix;
            for (value, activation) in state.iter_mut().zip(transformations) {
                *value = activation(*value);
            }
        }
        state
    }
}
