use ndarray::Array2;
use ndarray::Array1;

#[derive(Debug)]
pub struct MatrixEvaluator {
    pub stages: Vec<Array2<f64>>,
    pub transformations: Vec<crate::Transformations>
}

impl crate::network::Evaluator for MatrixEvaluator {
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