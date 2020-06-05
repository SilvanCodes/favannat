use ndarray::Array2;
use ndarray::Array;

#[derive(Debug)]
pub struct MatrixEvaluator {
    pub stages: Vec<Array2<f64>>,
    pub transformations: Vec<crate::Transformations>
}

impl crate::network::Evaluator for MatrixEvaluator {
    fn evaluate(&self, input: Vec<f64>) -> Vec<f64> {
        let mut state = Array::from_shape_vec(input.len(), input).unwrap();
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (matrix, activations) in self.stages.iter().zip(self.transformations.iter()) {
            state = state.dot(matrix);
            for (value, activation) in state.iter_mut().zip(activations.iter()) {
                *value = activation(*value);
            }
        }

        Vec::from(state.as_slice().unwrap())
    }
}