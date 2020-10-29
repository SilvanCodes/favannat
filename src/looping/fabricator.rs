use std::collections::HashMap;

use crate::network::{EdgeLike, NodeLike, StatefulFabricator};

use super::evaluator::{DependentNode, LoopingEvaluator, ValueMap};

#[derive(Debug)]
pub struct LoopingFabricator {}

impl<N, E> StatefulFabricator<N, E> for LoopingFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::LoopingEvaluator;

    fn fabricate(net: &impl crate::network::Recurrent<N, E>) -> Result<Self::Output, &'static str> {
        // TODO: map every node to custom id starting from zero too use them as

        let mut nodes: HashMap<usize, DependentNode> = HashMap::new();
        let mut node_active_sum_map: ValueMap<f64> = Default::default();
        let mut node_active_out_map: ValueMap<(f64, f64)> = Default::default();

        for node in net.nodes() {
            nodes.insert(
                node.id(),
                DependentNode {
                    activation_function: node.activation(),
                    inputs: Vec::new(),
                    active_flag: false,
                },
            );
            node_active_sum_map.insert(node.id(), 0.0);
            node_active_out_map.insert(node.id(), (0.0, 0.0));
        }

        for edge in net.edges() {
            if let Some(node) = nodes.get_mut(&edge.end()) {
                node.inputs.push((edge.start(), edge.weight(), false))
            }
        }

        for edge in net.recurrent_edges() {
            if let Some(node) = nodes.get_mut(&edge.end()) {
                node.inputs.push((edge.start(), edge.weight(), true))
            }
        }

        Ok(LoopingEvaluator {
            input_ids: net.inputs().iter().map(|i| i.id()).collect(),
            output_ids: net.outputs().iter().map(|i| i.id()).collect(),
            nodes,
            node_active_sum_map,
            node_active_out_map,
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
        // pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-1.0 * val).exp());
        pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-4.9 * val).exp());
        pub const TANH: fn(f64) -> f64 = |val| 2.0 * SIGMOID(2.0 * val) - 1.0;
        // a = 1, b = 0, c = 1
        pub const GAUSSIAN: fn(f64) -> f64 = |val| (val * val / -2.0).exp();
        // pub const STEP: fn(f64) -> f64 = |val| if val > 0.0 { 1.0 } else { 0.0 };
        // pub const SINE: fn(f64) -> f64 = |val| (val * std::f64::consts::PI).sin();
        // pub const COSINE: fn(f64) -> f64 = |val| (val * std::f64::consts::PI).cos();
        pub const INVERSE: fn(f64) -> f64 = |val| -val;
        // pub const ABSOLUTE: fn(f64) -> f64 = |val| val.abs();
        pub const RELU: fn(f64) -> f64 = |val| 0f64.max(val);
        pub const SQUARED: fn(f64) -> f64 = |val| val * val;
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
                        'l' => activations::LINEAR,
                        's' => activations::SIGMOID,
                        't' => activations::TANH,
                        'g' => activations::GAUSSIAN,
                        'r' => activations::RELU,
                        'q' => activations::SQUARED,
                        'i' => activations::INVERSE,
                        _ => activations::SIGMOID }
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

        assert_eq!(result, array![3.75]);
    }

    // test construction of carry for early result with dedup carry
    #[test]
    fn simple_net_evaluator_4() {
        let some_net = Net::new(
            1,
            2,
            nodes!('l', 'l', 'l', 'l'),
            edges!(
                0--0.5->1,
                1--0.5->2,
                0--0.5->3,
                0--0.5->2
            ),
        );

        let mut evaluator = LoopingFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![3.75, 2.5]);
    }

    // test construction of carry for early result flipped order
    #[test]
    fn simple_net_evaluator_5() {
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

        assert_eq!(result, array![2.5, 1.25]);
    }

    // test unconnected net
    #[test]
    fn simple_net_evaluator_6() {
        let some_net = Net::new(1, 1, nodes!('l', 'l'), Vec::new());

        if let Err(message) = LoopingFabricator::fabricate(&some_net) {
            assert_eq!(message, "no edges present, net invalid");
        } else {
            unreachable!();
        }
    }

    // test uncomputable output
    #[test]
    fn simple_net_evaluator_7() {
        let some_net = Net::new(1, 1, nodes!('l', 'l', 'l'), edges!(0--0.5->1));

        if let Err(message) = LoopingFabricator::fabricate(&some_net) {
            assert_eq!(
                message,
                "dependencies resolved but not all outputs computable, net invalid"
            );
        } else {
            unreachable!();
        }
    }

    // test uncomputable output
    #[test]
    fn simple_net_evaluator_8() {
        let some_net = Net::new(1, 1, nodes!('l', 'l', 'l'), edges!(1--0.5->2));

        if let Err(message) = LoopingFabricator::fabricate(&some_net) {
            assert_eq!(message, "can't resolve dependencies, net invalid");
        } else {
            unreachable!();
        }
    }

    #[test]
    fn simple_net_evaluator_9() {
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

        // recurrent unrolled
        // 0--1.0->6,
        // 2--1.0->4,
        //
        // 1--1.0->7,
        // 3--1.0->5

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
