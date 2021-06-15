use ndarray::Array1;

pub trait NodeLike: Ord {
    fn id(&self) -> usize;
    fn activation(&self) -> fn(f64) -> f64;
}

pub trait EdgeLike {
    fn start(&self) -> usize;
    fn end(&self) -> usize;
    fn weight(&self) -> f64;
}

pub trait NetLike<N: NodeLike, E: EdgeLike> {
    fn edges(&self) -> Vec<&E>;
    fn inputs(&self) -> Vec<&N>;
    fn hidden(&self) -> Vec<&N>;
    fn outputs(&self) -> Vec<&N>;

    fn nodes(&self) -> Vec<&N> {
        self.inputs()
            .into_iter()
            .chain(self.hidden().into_iter())
            .chain(self.outputs().into_iter())
            .collect()
    }
}

pub trait Recurrent<N: NodeLike, E: EdgeLike>: NetLike<N, E> {
    fn recurrent_edges(&self) -> Vec<&E>;
}

pub trait Evaluator {
    fn evaluate(&self, input: Array1<f64>) -> Array1<f64>;
}

pub trait StatefulEvaluator {
    fn evaluate(&mut self, input: Array1<f64>) -> Array1<f64>;
    fn reset_internal_state(&mut self);
}

pub trait Fabricator<N: NodeLike, E: EdgeLike> {
    type Output: Evaluator;

    fn fabricate(net: &impl NetLike<N, E>) -> Result<Self::Output, &'static str>;
}

pub trait StatefulFabricator<N: NodeLike, E: EdgeLike> {
    type Output: StatefulEvaluator;

    fn fabricate(net: &impl Recurrent<N, E>) -> Result<Self::Output, &'static str>;
}

pub mod net {
    use std::{collections::HashMap, ops::Shr};

    use super::{EdgeLike, NetLike, NodeLike, Recurrent};

    #[derive(Debug)]
    pub struct Node {
        id: usize,
        activation: fn(f64) -> f64,
    }

    impl Node {
        pub fn new(id: usize, activation: fn(f64) -> f64) -> Self {
            Self { id, activation }
        }
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

    #[derive(Debug)]
    pub struct Edge {
        start: usize,
        end: usize,
        weight: f64,
    }

    impl Edge {
        pub fn new(start: usize, end: usize, weight: f64) -> Self {
            Self { start, end, weight }
        }
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

    #[derive(Debug)]
    pub struct Net {
        inputs: usize,
        outputs: usize,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        recurrent_edges: Vec<Edge>,
    }

    impl NetLike<Node, Edge> for Net {
        fn edges(&self) -> Vec<&Edge> {
            self.edges.iter().collect()
        }
        fn inputs(&self) -> Vec<&Node> {
            self.nodes.iter().take(self.inputs).collect()
        }
        fn hidden(&self) -> Vec<&Node> {
            self.nodes
                .iter()
                .skip(self.inputs)
                .take(self.nodes.len() - self.inputs - self.outputs)
                .collect()
        }

        fn outputs(&self) -> Vec<&Node> {
            self.nodes
                .iter()
                .skip(self.nodes().len() - self.outputs)
                .collect()
        }

        fn nodes(&self) -> Vec<&Node> {
            self.nodes.iter().collect()
        }
    }

    impl Recurrent<Node, Edge> for Net {
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
        pub fn set_recurrent_edges(&mut self, edges: Vec<Edge>) {
            self.recurrent_edges = edges
        }
    }

    pub fn unroll<R: Recurrent<N, E>, N: NodeLike, E: EdgeLike>(recurrent: &R) -> Net {
        let mut known_inputs = recurrent
            .inputs()
            .iter()
            .map(|n| Node {
                id: n.id(),
                activation: n.activation(),
            })
            .collect::<Vec<_>>();

        let mut known_outputs = recurrent
            .outputs()
            .iter()
            .map(|n| Node {
                id: n.id(),
                activation: n.activation(),
            })
            .collect::<Vec<_>>();

        let mut known_edges = recurrent
            .edges()
            .iter()
            .map(|e| Edge {
                start: e.start(),
                end: e.end(),
                weight: e.weight(),
            })
            .collect::<Vec<_>>();

        let mut unroll_map: HashMap<usize, usize> = HashMap::new();
        // WARN: upper half of usize is used for wrappping node ids
        let mut tmp_ids = usize::MAX.shr(1)..usize::MAX;

        // create wrapping input for all original outputs, regardless of if they are used
        // this is to simplify the state transfer inside the stateful matrix evaluator
        for output in recurrent.outputs() {
            let wrapper_input_id = tmp_ids.next().unwrap();

            let wrapper_input_node = Node {
                id: wrapper_input_id,
                activation: |val| val,
            };

            known_inputs.push(wrapper_input_node);

            unroll_map.insert(output.id(), wrapper_input_id);
        }

        // create all wrapping nodes and egdes for recurent connections
        for recurrent_edge in recurrent.recurrent_edges() {
            let recurrent_input = unroll_map.entry(recurrent_edge.start()).or_insert_with(|| {
                let wrapper_input_id = tmp_ids.next().unwrap();

                let wrapper_input_node = Node {
                    id: wrapper_input_id,
                    activation: |val| val,
                };
                let wrapper_output_node = Node {
                    id: tmp_ids.next().unwrap(),
                    activation: |val| val,
                };

                // used to carry value into next evaluation
                let outward_wrapping_edge = Edge {
                    start: recurrent_edge.start(),
                    weight: 1.0,
                    end: wrapper_output_node.id(),
                };

                // add nodes for wrapping
                known_inputs.push(wrapper_input_node);
                known_outputs.push(wrapper_output_node);

                // add outward wrapping connection
                known_edges.push(outward_wrapping_edge);

                wrapper_input_id
            });

            let inward_wrapping_connection = Edge {
                start: *recurrent_input,
                end: recurrent_edge.end(),
                weight: recurrent_edge.weight(),
            };

            known_edges.push(inward_wrapping_connection);
        }

        let inputs_count = known_inputs.len();
        let outputs_count = known_outputs.len();
        let nodes = known_inputs
            .into_iter()
            .chain(recurrent.hidden().iter().map(|n| Node {
                id: n.id(),
                activation: n.activation(),
            }))
            .chain(known_outputs.into_iter())
            .collect::<Vec<_>>();
        let edges = known_edges;

        Net::new(inputs_count, outputs_count, nodes, edges)
    }

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

    #[macro_export]
    macro_rules! edges {
        ( $( $start:literal -- $weight:literal -> $end:literal ),* ) => {
            {
                vec![
                    $(
                        crate::network::net::Edge::new($start, $end, $weight),
                    )*
                ]
            }
        };
    }

    #[macro_export]
    macro_rules! nodes {
        ( $( $activation:literal ),* ) => {
            {
            let mut nodes = Vec::new();

            $(
                nodes.push(
                    crate::network::net::Node::new(nodes.len(), match $activation {
                        'l' => crate::network::net::activations::LINEAR,
                        's' => crate::network::net::activations::SIGMOID,
                        't' => crate::network::net::activations::TANH,
                        'g' => crate::network::net::activations::GAUSSIAN,
                        'r' => crate::network::net::activations::RELU,
                        'q' => crate::network::net::activations::SQUARED,
                        'i' => crate::network::net::activations::INVERSE,
                        _ => crate::network::net::activations::SIGMOID }
                    )
                );
            )*

            nodes
            }
        };
    }
}
