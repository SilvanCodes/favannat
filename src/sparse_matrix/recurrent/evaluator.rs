use nalgebra::DMatrix;

use crate::{
    network::{Evaluator, NetworkIO, StatefulEvaluator},
    sparse_matrix::feedforward::evaluator::SparseMatrixFeedforwardEvaluator,
};

#[derive(Debug)]
pub struct SparseMatrixRecurrentEvaluator {
    pub internal: DMatrix<f64>,
    pub evaluator: SparseMatrixFeedforwardEvaluator,
    pub outputs: usize,
}

impl StatefulEvaluator for SparseMatrixRecurrentEvaluator {
    fn evaluate<T: NetworkIO>(&mut self, input: T) -> T {
        let mut input = NetworkIO::input(input);
        input = DMatrix::from_iterator(
            1,
            input.len() + self.internal.len(),
            input.iter().chain(self.internal.iter()).cloned(),
        );

        self.internal = self.evaluator.evaluate(input);

        NetworkIO::output(DMatrix::from_iterator(
            1,
            self.outputs,
            self.internal
                .view((0, 0), (1, self.outputs))
                .iter()
                .cloned(),
        ))
    }

    fn reset_internal_state(&mut self) {
        self.internal = DMatrix::from_element(1, self.internal.len(), 0.0);
    }
}
