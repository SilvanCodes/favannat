use std::iter::once;

use crate::network::{Evaluator, StatefulEvaluator};
use ndarray::{stack, Array, Array1, Array2, Axis};

#[derive(Debug)]
pub struct MatrixEvaluator {
    pub stages: Vec<Array2<f64>>,
    pub transformations: Vec<crate::Transformations>,
}

impl Evaluator for MatrixEvaluator {
    fn evaluate(&self, mut state: Array1<f64>) -> Array1<f64> {
        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (stage_matrix, transformations) in self.stages.iter().zip(&self.transformations) {
            state = state.dot(stage_matrix);
            for (value, activation) in state.iter_mut().zip(transformations) {
                *value = activation(*value);
            }
        }
        state
    }
}

#[derive(Debug)]
pub struct RecurrentMatrixEvaluator {
    pub internal: Array1<f64>,
    pub evaluator: MatrixEvaluator,
}

impl StatefulEvaluator for RecurrentMatrixEvaluator {
    fn evaluate(&mut self, mut input: Array1<f64>) -> Array1<f64> {
        input = stack!(Axis(0), input, self.internal);
        let output = self.evaluator.evaluate(input);
        let (output, internal) = output
            .view()
            .split_at(Axis(0), output.len() - self.internal.len());
        self.internal = internal.to_owned();
        output.to_owned()
    }

    fn reset_internal_state(&mut self) {
        self.internal = Array::zeros(self.internal.len());
    }
}

#[derive(Debug)]
pub struct SelfNormalizingMatrixEvaluator {
    pub stages: Vec<Array2<f64>>,
    pub transformations: Vec<crate::Transformations>,
    pub cumulative_average: Vec<Array1<f64>>,
    pub cumulative_square_average: Vec<Array1<f64>>,
    pub evaluation_counter: usize,
}

impl From<MatrixEvaluator> for SelfNormalizingMatrixEvaluator {
    fn from(simple: MatrixEvaluator) -> Self {
        let zeroes = once(simple.stages[0].shape()[0])
            .chain(
                simple
                    .transformations
                    .iter()
                    .take(simple.transformations.len() - 1)
                    .map(|t| t.len()),
            )
            // simple.transformations.iter().map(|t| t.len())
            .map(Array::zeros)
            .collect::<Vec<Array1<f64>>>();
        Self {
            cumulative_average: zeroes.clone(),
            cumulative_square_average: zeroes,
            evaluation_counter: 0,
            stages: simple.stages,
            transformations: simple.transformations,
        }
    }
}

impl StatefulEvaluator for SelfNormalizingMatrixEvaluator {
    fn evaluate(&mut self, mut state: Array1<f64>) -> Array1<f64> {
        // increment evaluation counter to build cumulative averages
        self.evaluation_counter += 1;

        // performs evaluation by sequentially matrix multiplying and transforming the state with every stage
        for (index, stage_matrix) in self.stages.iter().enumerate() {
            self.cumulative_average[index] = &self.cumulative_average[index]
                + &((&state - &self.cumulative_average[index]) / self.evaluation_counter as f64);

            self.cumulative_square_average[index] = &self.cumulative_square_average[index]
                + &((&state * &state - &self.cumulative_square_average[index])
                    / self.evaluation_counter as f64);

            let variance = &self.cumulative_square_average[index]
                - &(&self.cumulative_average[index] * &self.cumulative_average[index])
                + f64::EPSILON;
            let std_dev = variance.mapv_into(|x| x.sqrt());

            // apply z-score to state values
            state -= &self.cumulative_average[index];
            state /= &std_dev;

            // apply layer
            state = state.dot(stage_matrix);

            // apply activation
            for (value, activation) in state.iter_mut().zip(self.transformations[index].iter()) {
                *value = activation(*value);
            }
        }
        state
    }

    fn reset_internal_state(&mut self) {
        for stage_average in &mut self.cumulative_average {
            stage_average.fill(0.0);
        }
        for stage_average in &mut self.cumulative_square_average {
            stage_average.fill(0.0);
        }
        self.evaluation_counter = 0;
    }
}

#[derive(Debug)]
pub struct SelfNormalizingRecurrentMatrixEvaluator {
    pub internal: Array1<f64>,
    pub evaluator: SelfNormalizingMatrixEvaluator,
}

impl StatefulEvaluator for SelfNormalizingRecurrentMatrixEvaluator {
    fn evaluate(&mut self, mut input: Array1<f64>) -> Array1<f64> {
        input = stack!(Axis(0), input, self.internal);
        let output = self.evaluator.evaluate(input);
        let (output, internal) = output
            .view()
            .split_at(Axis(0), output.len() - self.internal.len());
        self.internal = internal.to_owned();
        output.to_owned()
    }

    fn reset_internal_state(&mut self) {
        self.evaluator.reset_internal_state();
        self.internal = Array::zeros(self.internal.len());
    }
}

impl From<RecurrentMatrixEvaluator> for SelfNormalizingRecurrentMatrixEvaluator {
    fn from(simple: RecurrentMatrixEvaluator) -> Self {
        Self {
            internal: simple.internal,
            evaluator: simple.evaluator.into(),
        }
    }
}
