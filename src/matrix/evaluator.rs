use nalgebra::DMatrix;

use crate::network::{Evaluator, StatefulEvaluator};

#[derive(Debug)]
pub struct MatrixEvaluator {
    pub stages: Vec<DMatrix<f64>>,
    pub transformations: Vec<crate::Transformations>,
}

impl Evaluator for MatrixEvaluator {
    fn evaluate(&self, mut state: DMatrix<f64>) -> DMatrix<f64> {
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (stage_matrix, transformations) in self.stages.iter().zip(&self.transformations) {
            state = state * stage_matrix;
            for (value, activation) in state.iter_mut().zip(transformations) {
                *value = activation(*value);
            }
        }
        state.into()
    }
}

#[derive(Debug)]
pub struct RecurrentMatrixEvaluator {
    pub internal: DMatrix<f64>,
    pub evaluator: MatrixEvaluator,
    pub outputs: usize,
}

impl StatefulEvaluator for RecurrentMatrixEvaluator {
    fn evaluate(&mut self, mut input: DMatrix<f64>) -> DMatrix<f64> {
        input = DMatrix::from_iterator(
            1,
            input.len() + self.internal.len(),
            input.iter().chain(self.internal.iter()).cloned(),
        );

        self.internal = self.evaluator.evaluate(input);

        DMatrix::from_iterator(
            1,
            self.outputs,
            self.internal
                .slice((0, 0), (1, self.outputs))
                .iter()
                .cloned(),
        )
    }

    fn reset_internal_state(&mut self) {
        self.internal = DMatrix::from_element(1, self.internal.len(), 0.0);
    }
}
