use std::ops::Deref;

pub mod activations {

    #[derive(Debug)]
    pub struct ActivationFunction(pub fn(f64) -> f64);

    impl Default for ActivationFunction {
        fn default() -> Self { ActivationFunction(SIGMOID) }
    }

    impl ActivationFunction {
        pub fn linear() -> Self { ActivationFunction(LINEAR) }
        pub fn sigmoid() -> Self { ActivationFunction(SIGMOID) }
        pub fn tanh() -> Self { ActivationFunction(TANH) }
        pub fn gaussian() -> Self { ActivationFunction(GAUSSIAN) }
    }

    pub const LINEAR: fn(f64) -> f64 = |val| val;
    pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-1.0 * val).exp());
    pub const TANH: fn(f64) -> f64 = |val| 2.0 * SIGMOID(2.0 * val) - 1.0;
    pub const GAUSSIAN: fn(f64) -> f64 = |val| (val * val / -2.0).exp(); // a = 1, b = 0, c = 1
}

use activations::ActivationFunction;

macro_rules! transparent {
    ( $name:ident, $type:ty ) => {
        impl Deref for $name {
            type Target = $type;
        
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct NodeId(pub usize);

#[derive(Debug)]
pub struct Node (
    NodeId,
    ActivationFunction
);
transparent!(Node, NodeId);

impl Node {
    pub fn new(id: usize, activation: Option<ActivationFunction>) -> Self {
        Node (
            NodeId(id),
            activation.unwrap_or_default()
        )
    }

    pub fn activation(&self) -> fn(f64) -> f64 {
        (self.1).0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeStart(pub NodeId);
transparent!(EdgeStart, NodeId);

#[derive(Clone, Copy, Debug)]
pub struct EdgeEnd(pub NodeId);
transparent!(EdgeEnd, NodeId);

#[derive(Clone, Copy, Debug)]
pub struct Weight(pub f64);
transparent!(Weight, f64);

#[derive(Clone, Copy, Debug)]
pub struct Edge (
    pub EdgeStart,
    pub EdgeEnd,
    pub Weight,
);

impl Edge {
    pub fn new(from: NodeId, to: NodeId, weight: f64) -> Self {
        Edge (
            EdgeStart(from),
            EdgeEnd(to),
            Weight(weight)
        )
    }
}

#[derive(Debug)]
pub struct IoDim(usize, usize);

#[derive(Debug)]
pub struct Nodes(Vec<Node>);
transparent!(Nodes, Vec<Node>);

impl Nodes {
    pub fn select(&self, id: NodeId) -> Option<&Node> {
        self.0.iter().find(|node| ***node == id)
    }
}

#[derive(Debug)]
pub struct Inputs(Vec<NodeId>);
transparent!(Inputs, Vec<NodeId>);

#[derive(Debug)]
pub struct Outputs(Vec<NodeId>);
transparent!(Outputs, Vec<NodeId>);

#[derive(Debug)]
pub struct Edges(Vec<Edge>);
transparent!(Edges, Vec<Edge>);

macro_rules! edges {
    ( $( $from:literal -- $w:literal -> $to:literal ),* ) => {
        {
        let mut edges = Vec::new();

        $(
            edges.push(
                Edge::new(NodeId($from), NodeId($to), $w)
            );
        )*

        edges
        }
    };
}

macro_rules! nodes {
    ( $( $id:literal $activation:literal),* ) => {
        {
        let mut nodes = Vec::new();

        $(
            nodes.push(
                Node::new($id, match $activation {
                    'l' => Some(ActivationFunction::linear()),
                    's' => Some(ActivationFunction::sigmoid()),
                    't' => Some(ActivationFunction::tanh()),
                    'g' => Some(ActivationFunction::gaussian()),
                    _ => Some(ActivationFunction::sigmoid())
                })
            );
        )*

        nodes
        }
    };
}



#[derive(Debug)]
pub struct Net (
    IoDim,
    Nodes,
    Edges
);

pub trait Evaluator {
    fn evaluate(&self, input: Vec<f64>) -> Vec<f64>;
}

pub trait Fabricator {
    type Output: Evaluator;

    fn fabricate(net: Net) -> Result<Self::Output, &'static str>;
}

impl Net {
    pub fn new(input: usize, output: usize, nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
        Net (
            IoDim(input, output),
            Nodes(nodes),
            Edges(edges)
        )
    }

    pub fn nodes(&self) -> &Nodes {
        &self.1
    }

    pub fn edges(&self) -> &Edges {
        &self.2
    }

    pub fn inputs(&self) -> Inputs {
        Inputs(self.1.iter().take((self.0).0).map(|node| **node).collect())
    }

    pub fn outputs(&self) -> Vec<NodeId> {
        self.1.iter().rev().take((self.0).1).rev().map(|node| **node).collect()
    }
}
