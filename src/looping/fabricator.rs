use std::collections::HashMap;

use crate::network::{EdgeLike, NodeLike, StatefulFabricator};

use super::evaluator::{DependentNode, LoopingEvaluator};

#[deprecated(
    since = "0.1.1",
    note = "Please use NeatOriginalFabricator, which this LoopingFabricator was intended to be."
)]
#[derive(Debug)]
pub struct LoopingFabricator {}

impl<N, E> StatefulFabricator<N, E> for LoopingFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::LoopingEvaluator;

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
                flags: [false; 2],
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

        Ok(LoopingEvaluator {
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

#[cfg(test)]
mod tests {
    use super::LoopingFabricator;
    use crate::network::{
        EdgeLike, NetLike, NodeLike, Recurrent, StatefulEvaluator, StatefulFabricator,
    };
    use ndarray::array;

    pub mod activations {
        pub const LINEAR: fn(f64) -> f64 = |val| val;
    }

    #[derive(Debug, Clone)]
    pub struct Node {
        id: usize,
        activation: fn(f64) -> f64,
    }

    impl NodeLike for Node {
        fn id(&self) -> usize {
            self.id
        }
        fn activation(&self) -> fn(f64) -> f64 {
            self.activation
        }
    }

    impl PartialEq for Node {
        fn eq(&self, other: &Self) -> bool {
            self.id() == other.id()
        }
    }

    impl Eq for Node {}

    impl PartialOrd for Node {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Node {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.id().cmp(&other.id())
        }
    }

    #[derive(Debug, Clone)]
    pub struct Edge {
        start: usize,
        end: usize,
        weight: f64,
    }

    impl EdgeLike for Edge {
        fn start(&self) -> usize {
            self.start
        }
        fn end(&self) -> usize {
            self.end
        }
        fn weight(&self) -> f64 {
            self.weight
        }
    }

    #[derive(Debug, Clone)]
    pub struct Net {
        inputs: usize,
        outputs: usize,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        recurrent_edges: Vec<Edge>,
    }

    impl NetLike<Node, Edge> for Net {
        fn nodes(&self) -> Vec<&Node> {
            self.nodes.iter().collect()
        }
        fn edges(&self) -> Vec<&Edge> {
            self.edges.iter().collect()
        }
        fn inputs(&self) -> Vec<&Node> {
            self.nodes.iter().take(self.inputs).collect()
        }
        fn outputs(&self) -> Vec<&Node> {
            self.nodes
                .iter()
                .skip(self.nodes().len() - self.outputs)
                .collect()
        }
    }

    impl Recurrent<Node, Edge> for Net {
        type Net = Self;

        fn unroll(&self) -> Self::Net {
            self.clone()
        }

        fn recurrent_edges(&self) -> Vec<&Edge> {
            self.recurrent_edges.iter().collect()
        }
    }

    impl Net {
        pub fn new(inputs: usize, outputs: usize, nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
            Net {
                inputs,
                outputs,
                nodes,
                edges,
                recurrent_edges: Vec::new(),
            }
        }
    }

    macro_rules! edges {
        ( $( $start:literal -- $weight:literal -> $end:literal ),* ) => {
            {
            let mut edges = Vec::new();

            $(
                edges.push(
                    Edge { start: $start, end: $end, weight: $weight }
                );
            )*

            edges
            }
        };
    }

    macro_rules! nodes {
        ( $( $activation:literal ),* ) => {
            {
            let mut nodes = Vec::new();

            $(
                nodes.push(
                    Node { id: nodes.len(), activation: match $activation {
                        _ => activations::LINEAR }
                    }
                );
            )*

            nodes
            }
        };
    }

    // tests construction and evaluation of simplest network
    #[test]
    fn simple_net_evaluator_0() {
        let some_net = Net::new(1, 1, nodes!('l', 'l'), edges!(0--0.5->1));

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![2.5]);
    }

    // tests input dimension > 1
    #[test]
    fn simple_net_evaluator_1() {
        let some_net = Net::new(
            2,
            1,
            nodes!('l', 'l', 'l'),
            edges!(
                0--0.5->2,
                1--0.5->2
            ),
        );

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0, 5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![5.0]);
    }

    // test linear chaining of edges
    #[test]
    fn simple_net_evaluator_2() {
        let some_net = Net::new(
            1,
            1,
            nodes!('l', 'l', 'l'),
            edges!(
                0--0.5->1,
                1--0.5->2
            ),
        );

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![1.25]);
    }

    // test construction of carry for later needs
    #[test]
    fn simple_net_evaluator_3() {
        let some_net = Net::new(
            1,
            1,
            nodes!('l', 'l', 'l'),
            edges!(
                0--0.5->1,
                1--0.5->2,
                0--0.5->2
            ),
        );

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![2.5]);
    }

    // test construction of carry for early result flipped order
    #[test]
    fn simple_net_evaluator_4() {
        let some_net = Net::new(
            1,
            2,
            nodes!('l', 'l', 'l', 'l'),
            edges!(
                0--0.5->1,
                1--0.5->3,
                0--0.5->2
            ),
        );

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![0.0, 1.25]);
    }

    #[test]
    fn simple_net_evaluator_5() {
        let some_net = Net::new(
            2,
            1,
            nodes!('l', 'l', 'l'),
            edges!(
                0--0.5->2,
                1--0.0->2
            ),
        );

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0, 5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![2.5]);
    }

    #[test]
    fn stateful_net_evaluator_0() {
        let mut some_net = Net::new(
            2,
            2,
            nodes!('l', 'l', 'l', 'l'),
            edges!(
            // standard feed-forward
            0--1.0->2,
            1--1.0->3),
        );

        some_net.recurrent_edges = edges!(
            0--1.0->2,
            1--1.0->3
        );

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0, 0.0]);
        assert_eq!(result, array![5.0, 0.0]);

        let result = evaluator.evaluate(array![5.0, 5.0]);
        assert_eq!(result, array![10.0, 5.0]);

        let result = evaluator.evaluate(array![0.0, 5.0]);
        assert_eq!(result, array![5.0, 10.0]);

        let result = evaluator.evaluate(array![0.0, 0.0]);
        assert_eq!(result, array![0.0, 5.0]);
    }
}
