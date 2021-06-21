use crate::network::{Evaluator, StatefulEvaluator};
use ndarray::{s, Array, Array1, Axis};
use sprs::CsMat;

#[derive(Debug)]
pub struct SparseMatrixEvaluator {
    pub stages: Vec<CsMat<f64>>,
    pub transformations: Vec<crate::Transformations>,
}

impl Evaluator for SparseMatrixEvaluator {
    fn evaluate(&self, state: Array1<f64>) -> Array1<f64> {
        let mut state = state.insert_axis(Axis(0));
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (stage_matrix, transformations) in self.stages.iter().zip(&self.transformations) {
            state = (state).dot(stage_matrix);
            for (value, activation) in state.iter_mut().zip(transformations) {
                *value = activation(*value);
            }
        }
        state.remove_axis(Axis(0))
    }
}

#[derive(Debug)]
pub struct RecurrentMatrixEvaluator {
    pub internal: Array1<f64>,
    pub evaluator: SparseMatrixEvaluator,
    pub outputs: usize,
}

impl StatefulEvaluator for RecurrentMatrixEvaluator {
    fn evaluate(&mut self, mut input: Array1<f64>) -> Array1<f64> {
        input = input.iter().chain(self.internal.iter()).cloned().collect();
        self.internal = self.evaluator.evaluate(input);
        self.internal.slice(s![0..self.outputs]).to_owned()
    }

    fn reset_internal_state(&mut self) {
        self.internal = Array::zeros(self.internal.len());
    }
}
