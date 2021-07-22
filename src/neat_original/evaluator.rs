use nalgebra::DMatrix;

use crate::network::StatefulEvaluator;

#[derive(Debug)]
pub struct DependentNode {
    pub activation_function: fn(f64) -> f64,
    pub inputs: Vec<(usize, f64, bool)>,
    pub is_active: bool,
}

#[derive(Debug)]
pub struct NeatOriginalEvaluator {
    pub input_ids: Vec<usize>,
    pub output_ids: Vec<usize>,
    pub nodes: Vec<DependentNode>,
    pub node_input_sum: Vec<f64>,
    // [0] is current output, [1] it output before that
    pub node_active_output: Vec<[f64; 2]>,
}

impl NeatOriginalEvaluator {
    fn outputs_off(&self) -> bool {
        for &id in self.output_ids.iter() {
            if !self.nodes[id].is_active {
                return true;
            }
        }
        false
    }
}

impl StatefulEvaluator for NeatOriginalEvaluator {
    fn evaluate(&mut self, input: DMatrix<f64>) -> DMatrix<f64> {
        for (&id, &value) in self.input_ids.iter().zip(input.iter()) {
            self.node_active_output[id][0] = value;
            self.nodes[id].is_active = true;
        }

        let mut onetime = false;

        while self.outputs_off() || !onetime {
            for id in 0..self.nodes.len() {
                if !self.input_ids.contains(&id) {
                    self.node_input_sum[id] = 0.0;
                    self.nodes[id].is_active = false;

                    let inputs = self.nodes[id].inputs.clone();
                    for &(dep_id, weight, recurrent) in inputs.iter() {
                        if !recurrent {
                            if self.nodes[dep_id].is_active {
                                self.nodes[id].is_active = true;
                            }
                            self.node_input_sum[id] += self.node_active_output[dep_id][0] * weight;
                        } else {
                            self.node_input_sum[id] += self.node_active_output[dep_id][1] * weight;
                        }
                    }
                }
            }

            for id in 0..self.nodes.len() {
                if !self.input_ids.contains(&id) && self.nodes[id].is_active {
                    // shift last output in time
                    self.node_active_output[id][1] = self.node_active_output[id][0];
                    // compute new output when possible
                    self.node_active_output[id][0] =
                        (self.nodes[id].activation_function)(self.node_input_sum[id]);
                }
            }

            onetime = true;
        }

        DMatrix::from_iterator(
            1,
            self.output_ids.len(),
            self.output_ids
                .iter()
                .map(|&id| self.node_active_output[id][0]), // .collect::<Vec<_>>(),
        )
    }

    fn reset_internal_state(&mut self) {
        for value in self.node_input_sum.iter_mut() {
            *value = 0.0;
        }
        for value in self.node_active_output.iter_mut() {
            *value = [0.0; 2];
        }
    }
}
