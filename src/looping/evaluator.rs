use std::{
    collections::HashMap,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use ndarray::Array1;

use crate::network::StatefulEvaluator;

#[derive(Debug)]

pub struct DependentNode {
    pub activation_function: fn(f64) -> f64,
    pub inputs: Vec<(usize, f64, bool)>,
    pub active_flag: bool,
}

#[derive(Debug)]

pub struct LoopingEvaluator {
    pub input_ids: Vec<usize>,
    pub output_ids: Vec<usize>,
    pub nodes: HashMap<usize, DependentNode>,
    pub node_active_sum_map: ValueMap<f64>,
    pub node_active_out_map: ValueMap<(f64, f64)>,
}

// TODO: rewrite ValueMap as Vec with id as

#[derive(Debug, Default)]
pub struct ValueMap<T>(HashMap<usize, T>);

impl<T> Index<&usize> for ValueMap<T> {
    type Output = T;

    fn index(&self, index: &usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<&usize> for ValueMap<T> {
    fn index_mut(&mut self, index: &usize) -> &mut Self::Output {
        self.0.get_mut(index).unwrap()
    }
}

impl<T> Deref for ValueMap<T> {
    type Target = HashMap<usize, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for ValueMap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl LoopingEvaluator {
    fn outputs_off(&self) -> bool {
        for id in self.output_ids.iter() {
            if !self.nodes[id].active_flag {
                return true;
            }
        }
        false
    }
}

impl StatefulEvaluator for LoopingEvaluator {
    fn evaluate(&mut self, input: ndarray::Array1<f64>) -> ndarray::Array1<f64> {
        for (id, &value) in self.input_ids.iter().zip(input.iter()) {
            self.node_active_out_map[id].1 = self.node_active_out_map[id].0;
            self.node_active_out_map[id].0 = value;
        }

        let mut onetime = false;

        while self.outputs_off() || !onetime {
            for (id, node) in self.nodes.iter_mut() {
                if !self.input_ids.contains(id) {
                    self.node_active_sum_map[id] = 0.0;
                    node.active_flag = false;

                    for (dep_id, weight, recurrent) in node.inputs.iter() {
                        node.active_flag = true;
                        if *recurrent {
                            self.node_active_sum_map[id] +=
                                self.node_active_out_map[dep_id].1 * weight;
                        } else {
                            self.node_active_sum_map[id] +=
                                self.node_active_out_map[dep_id].0 * weight;
                        }
                    }
                }
            }

            for (id, node) in self.nodes.iter().filter(|(_, n)| n.active_flag) {
                self.node_active_out_map[id].1 = self.node_active_out_map[id].0;
                self.node_active_out_map[id].0 =
                    (node.activation_function)(self.node_active_sum_map[id]);
            }

            onetime = true;
        }

        self.output_ids
            .iter()
            .map(|id| self.node_active_out_map[id].0)
            .collect::<Array1<f64>>()
    }

    fn reset_internal_state(&mut self) {
        for value in self.node_active_sum_map.values_mut() {
            *value = 0.0;
        }
        for value in self.node_active_out_map.values_mut() {
            *value = (0.0, 0.0);
        }
    }
}
