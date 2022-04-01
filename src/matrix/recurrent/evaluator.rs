use nalgebra::DMatrix;

use crate::{
    matrix::feedforward::evaluator::MatrixFeedforwardEvaluator,
    network::{Evaluator, NetworkIO, StatefulEvaluator},
};

#[derive(Debug)]
pub struct MatrixRecurrentEvaluator {
    pub internal: DMatrix<f64>,
    pub evaluator: MatrixFeedforwardEvaluator,
    pub outputs: usize,
}

impl StatefulEvaluator for MatrixRecurrentEvaluator {
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
                .slice((0, 0), (1, self.outputs))
                .iter()
                .cloned(),
        ))
    }

    fn reset_internal_state(&mut self) {
        self.internal = DMatrix::from_element(1, self.internal.len(), 0.0);
    }
}
