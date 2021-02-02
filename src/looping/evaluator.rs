use crate::network::StatefulEvaluator;

#[derive(Debug)]
pub struct DependentNode {
    pub activation_function: fn(f64) -> f64,
    pub inputs: Vec<(usize, f64, bool)>,
    // flag[0] is 'should compute', flag[1] is 'should propagate'
    pub flags: [bool; 2],
}

#[derive(Debug)]
pub struct LoopingEvaluator {
    pub input_ids: Vec<usize>,
    pub output_ids: Vec<usize>,
    pub nodes: Vec<DependentNode>,
    pub node_input_sum: Vec<f64>,
    // [0] is current output, [1] it output before that
    pub node_active_output: Vec<[f64; 2]>,
}

impl LoopingEvaluator {
    fn outputs_off(&self) -> bool {
        for &id in self.output_ids.iter() {
            if !self.nodes[id].flags[1] {
                return true;
            }
        }
        false
    }
}

impl StatefulEvaluator for LoopingEvaluator {
    fn evaluate(&mut self, input: ndarray::Array1<f64>) -> ndarray::Array1<f64> {
        for (&id, &value) in self.input_ids.iter().zip(input.iter()) {
            self.node_active_output[id][0] = value;
            self.nodes[id].flags[1] = true;
        }

        while self.outputs_off() {
            for id in 0..self.nodes.len() {
                if !self.input_ids.contains(&id) {
                    self.node_input_sum[id] = 0.0;

                    let inputs = self.nodes[id].inputs.clone();
                    for &(dep_id, weight, recurrent) in inputs.iter() {
                        if self.nodes[dep_id].flags[1] {
                            self.nodes[id].flags[0] = true;
                            if recurrent {
                                self.node_input_sum[id] +=
                                    self.node_active_output[dep_id][1] * weight;
                            } else {
                                self.node_input_sum[id] +=
                                    self.node_active_output[dep_id][0] * weight;
                            }
                        }
                    }
                }
            }

            for id in 0..self.nodes.len() {
                // shift last output in time
                self.node_active_output[id][1] = self.node_active_output[id][0];
                // compute new output when possible
                if self.nodes[id].flags[0] {
                    self.node_active_output[id][0] =
                        (self.nodes[id].activation_function)(self.node_input_sum[id]);
                    self.nodes[id].flags[0] = false;
                    self.nodes[id].flags[1] = true;
                }
                // or write zero
                else {
                    self.node_active_output[id][0] = 0.0;
                }
            }
        }

        let mut output = Vec::new();

        for &id in self.output_ids.iter() {
            self.nodes[id].flags[1] = false;
            output.push(self.node_active_output[id][0]);
        }

        output.into()
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
