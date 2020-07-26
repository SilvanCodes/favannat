use ndarray::{Array1, Array2, Axis, stack};
use crate::network::{Evaluator, StatefulEvaluator};

#[derive(Debug)]
pub struct MatrixEvaluator {
    pub stages: Vec<Array2<f64>>,
    pub transformations: Vec<crate::Transformations>
}

impl Evaluator for MatrixEvaluator {
    fn evaluate(&self, mut state: Array1<f64>) -> Array1<f64> {
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (matrix, activations) in self.stages.iter().zip(self.transformations.iter()) {
            state = state.dot(matrix);
            for (value, activation) in state.iter_mut().zip(activations.iter()) {
                *value = activation(*value);
            }
        }
        state
    }
}

#[derive(Debug)]
pub struct StatefulMatrixEvaluator {
    pub internal: Array1<f64>,
    pub evaluator: MatrixEvaluator,
}

impl StatefulEvaluator for StatefulMatrixEvaluator {
    fn evaluate(&mut self, mut input: Array1<f64>) -> Array1<f64> {
        input = stack!(Axis(0), input, self.internal);
        let output = self.evaluator.evaluate(input);
        let (output, internal) = output.view().split_at(Axis(0), output.len() - self.internal.len());
        self.internal = internal.to_owned();
        output.to_owned()
    }
}
