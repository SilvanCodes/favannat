use nalgebra::DMatrix;
use nalgebra_sparse::{CscMatrix, SparseEntry, SparseEntryMut};

use crate::network::{Evaluator, StatefulEvaluator};

#[derive(Debug)]
pub struct SparseMatrixEvaluator {
    pub stages: Vec<CscMatrix<f64>>,
    pub transformations: Vec<crate::Transformations>,
}

impl Evaluator for SparseMatrixEvaluator {
    fn evaluate(&self, state: DMatrix<f64>) -> DMatrix<f64> {
        let mut len = 0;
        let mut state: CscMatrix<f64> = (&state).into();
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (stage_matrix, transformations) in self.stages.iter().zip(&self.transformations) {
            len = transformations.len();
            state = state * stage_matrix;
            for (index, activation) in transformations.iter().enumerate() {
                if let SparseEntryMut::NonZero(value) = state.index_entry_mut(0, index) {
                    *value = activation(*value);
                }
            }
        }
        DMatrix::from_iterator(
            1,
            len,
            (0..len).map(|index| {
                if let SparseEntry::NonZero(value) = state.index_entry(0, index) {
                    *value
                } else {
                    0.0
                }
            }),
        )
    }
}

#[derive(Debug)]
pub struct RecurrentMatrixEvaluator {
    pub internal: DMatrix<f64>,
    pub evaluator: SparseMatrixEvaluator,
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
