use std::collections::HashMap;

use crate::network::{EdgeLike, NodeLike, StatefulFabricator};

use super::evaluator::{DependentNode, NeatOriginalEvaluator};

#[derive(Debug)]
pub struct NeatOriginalFabricator {}

impl<N, E> StatefulFabricator<N, E> for NeatOriginalFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::NeatOriginalEvaluator;

    fn fabricate(net: &impl crate::network::Recurrent<N, E>) -> Result<Self::Output, &'static str> {
        let mut nodes: Vec<DependentNode> = Vec::new();

        let node_input_sum: Vec<f64> = vec![0.0; net.nodes().len()];
        let node_active_output: Vec<[f64; 2]> = vec![[0.0; 2]; net.nodes().len()];

        let mut id_gen = 0_usize..;
        let mut id_map: HashMap<usize, usize> = HashMap::new();

        for node in net.nodes() {
            id_map.insert(node.id(), id_gen.next().unwrap());

            nodes.push(DependentNode {
                activation_function: node.activation(),
                inputs: Vec::new(),
                is_active: false,
            });
        }

        for edge in net.edges() {
            nodes[*id_map.get(&edge.end()).unwrap()].inputs.push((
                *id_map.get(&edge.start()).unwrap(),
                edge.weight(),
                false,
            ))
        }

        for edge in net.recurrent_edges() {
            nodes[*id_map.get(&edge.end()).unwrap()].inputs.push((
                *id_map.get(&edge.start()).unwrap(),
                edge.weight(),
                true,
            ))
        }

        Ok(NeatOriginalEvaluator {
            input_ids: net
                .inputs()
                .iter()
                .map(|i| *id_map.get(&i.id()).unwrap())
                .collect(),
            output_ids: net
                .outputs()
                .iter()
                .map(|i| *id_map.get(&i.id()).unwrap())
                .collect(),
            nodes,
            node_input_sum,
            node_active_output,
        })
    }
}
