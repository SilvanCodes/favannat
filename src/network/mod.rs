//! Defines vocabulary and interfaces for this crate.

pub use self::io::NetworkIO;

mod io;

use std::collections::HashMap;

/// Declares a structure to have [`NodeLike`] properties.
///
/// [`NodeLike`] provides the plumbing to accept user-defined structures and use them as nodes in this crates context.
/// The implementation of [`NodeLike::id`] needs to provide a unique identifier per node.
pub trait NodeLike: Ord {
    fn id(&self) -> usize;
    fn activation(&self) -> fn(f64) -> f64;
}

/// Declares a structure to have [`EdgeLike`] properties.
///
/// [`EdgeLike`] provides the plumbing to accept user-defined structures and use them as edges in this crates context.
pub trait EdgeLike {
    fn start(&self) -> usize;
    fn end(&self) -> usize;
    fn weight(&self) -> f64;
}

/// Declares a structure to have network-like properties.
///
/// `NetworkLike` sits at the core of this crate.
/// Together with [`NodeLike`] and [`EdgeLike`] it provides the interface to start using this crate.
/// Structures that are `NetworkLike` can be fabricated and evaluated by the different implementations of the
/// [`Fabricator`], [`Evaluator`], [`StatefulFabricator`] and [`StatefulEvaluator`] traits.
pub trait NetworkLike<N: NodeLike, E: EdgeLike> {
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

/// Declares a [`NetworkLike`] structure to have recurrent edges.
///
/// Recurrent edges act like memory cells in a network.
/// They imply that internal state has to be preserved.
pub trait Recurrent<N: NodeLike, E: EdgeLike>: NetworkLike<N, E> {
    fn recurrent_edges(&self) -> Vec<&E>;
}

/// A facade behind which evaluation of a fabricated [`NetworkLike`] structure is implemented.
pub trait Evaluator {
    fn evaluate<T: NetworkIO>(&self, input: T) -> T;
}

/// A facade behind which evaluation of a fabricated [`Recurrent`] [`NetworkLike`] structure is implemented.
///
/// Due to its statefulness it needs mutable access and provides a way to reset the internal state.
pub trait StatefulEvaluator {
    fn evaluate<T: NetworkIO>(&mut self, input: T) -> T;
    fn reset_internal_state(&mut self);
}

/// A facade behind which the fabrication of a [`NetworkLike`] structure is implemented.
///
/// Fabrication means transforming a description of a network, the [`NetworkLike`] structure, into an executable form of its encoded function, an [`Evaluator`].
pub trait Fabricator<N: NodeLike, E: EdgeLike> {
    type Output: Evaluator;

    fn fabricate(net: &impl NetworkLike<N, E>) -> Result<Self::Output, &'static str>;
}

/// A facade behind which the fabrication of a [`Recurrent`] [`NetworkLike`] structure is implemented.
///
/// Fabrication means transforming a description of a network, the [`Recurrent`] [`NetworkLike`] structure, into an executable form of its encoded function, a [`StatefulEvaluator`].
pub trait StatefulFabricator<N: NodeLike, E: EdgeLike> {
    type Output: StatefulEvaluator;

    fn fabricate(net: &impl Recurrent<N, E>) -> Result<Self::Output, &'static str>;
}

/// Abstracts over how a computation stage matrix is built from triplet (COO) data.
///
/// Implementing this trait allows a matrix type to be used as the backend for [`fabricate_stages`].
pub trait StageMatrix: Sized {
    fn from_stage_data(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Self;
}

/// Core fabrication logic shared by all matrix-backed fabricators.
///
/// Resolves the computation stages for a [`NetworkLike`] structure using topological dependency
/// resolution, carry propagation, and output reordering. The resulting stage matrices are built
/// via [`StageMatrix::from_stage_data`], allowing both dense and sparse backends to reuse this
/// single implementation.
pub fn fabricate_stages<N, E, M>(
    net: &impl NetworkLike<N, E>,
) -> Result<(Vec<M>, Vec<crate::Transformations>), &'static str>
where
    N: NodeLike,
    E: EdgeLike,
    M: StageMatrix,
{
    let mut dependency_graph: HashMap<usize, Vec<&E>> = HashMap::new();
    for edge in net.edges() {
        dependency_graph
            .entry(edge.end())
            .and_modify(|deps| deps.push(edge))
            .or_insert_with(|| vec![edge]);
    }
    if dependency_graph.is_empty() {
        return Err("no edges present, net invalid");
    }

    let mut dependency_count = dependency_graph.len();
    let mut compute_stages: Vec<(usize, usize, Vec<usize>, Vec<usize>, Vec<f64>)> = Vec::new();
    let mut stage_transformations: Vec<crate::Transformations> = Vec::new();

    let mut available_nodes = net.inputs();
    available_nodes.sort_unstable();
    let mut available_nodes: Vec<usize> = available_nodes.iter().map(|n| n.id()).collect();

    let mut wanted_nodes = net.outputs();
    wanted_nodes.sort_unstable();
    let wanted_nodes: Vec<usize> = wanted_nodes.iter().map(|n| n.id()).collect();

    while !dependency_graph.is_empty() {
        let mut transformations: crate::Transformations = Vec::new();
        let mut next_available_nodes: Vec<usize> = Vec::new();
        let mut column_index = 0usize;
        let mut stage_col_indices: Vec<usize> = Vec::new();
        let mut stage_row_indices: Vec<usize> = Vec::new();
        let mut stage_data: Vec<f64> = Vec::new();

        for (&dependent_node, dependencies) in dependency_graph.iter() {
            let mut node_col_indices = Vec::new();
            let mut node_row_indices = Vec::new();
            let mut node_data = Vec::new();
            let mut computable = true;

            for &dependency in dependencies {
                let mut found = false;
                for (row_index, &id) in available_nodes.iter().enumerate() {
                    if dependency.start() == id {
                        node_col_indices.push(column_index);
                        node_row_indices.push(row_index);
                        node_data.push(dependency.weight());
                        found = true;
                    }
                }
                if !found {
                    computable = false;
                }
            }

            if computable {
                stage_col_indices.extend(node_col_indices);
                stage_row_indices.extend(node_row_indices);
                stage_data.extend(node_data);
                transformations.push(
                    net.nodes()
                        .iter()
                        .find(|&node| node.id() == dependent_node)
                        .unwrap()
                        .activation(),
                );
                column_index += 1;
                next_available_nodes.push(dependent_node);
            } else {
                // carry partial dependencies forward
                for row_index in node_row_indices {
                    if !next_available_nodes.contains(&available_nodes[row_index]) {
                        stage_row_indices.push(row_index);
                        stage_col_indices.push(column_index);
                        stage_data.push(1.0);
                        column_index += 1;
                        transformations.push(|val| val);
                        next_available_nodes.push(available_nodes[row_index]);
                    }
                }
            }
        }

        // carry wanted output nodes if already available
        for wanted_node in wanted_nodes.iter() {
            for (row_index, available_node) in available_nodes.iter().enumerate() {
                if available_node == wanted_node && !next_available_nodes.contains(available_node) {
                    stage_col_indices.push(column_index);
                    stage_row_indices.push(row_index);
                    stage_data.push(1.0);
                    column_index += 1;
                    transformations.push(|val| val);
                    next_available_nodes.push(*available_node);
                }
            }
        }

        for node in next_available_nodes.iter() {
            dependency_graph.remove(node);
        }

        if dependency_graph.len() == dependency_count {
            return Err("can't resolve dependencies, net invalid");
        } else {
            dependency_count = dependency_graph.len();
        }

        // reorder last stage to match wanted output order
        if dependency_graph.is_empty() {
            let mut reordered_col_indices = vec![usize::MAX; stage_col_indices.len()];
            let mut reordered_transformations = transformations.clone();
            let mut matched_wanted_count = 0;

            for (old_col, available_node) in next_available_nodes.iter().enumerate() {
                for (new_col, wanted_node) in wanted_nodes.iter().enumerate() {
                    if available_node == wanted_node {
                        for (reordered, &old) in
                            reordered_col_indices.iter_mut().zip(stage_col_indices.iter())
                        {
                            if old == old_col {
                                *reordered = new_col;
                            }
                        }
                        reordered_transformations[new_col] = transformations[old_col];
                        matched_wanted_count += 1;
                        break;
                    }
                }
            }

            if matched_wanted_count < wanted_nodes.len() {
                return Err(
                    "dependencies resolved but not all outputs computable, net invalid",
                );
            }

            stage_col_indices = reordered_col_indices;
            transformations = reordered_transformations;
        }

        compute_stages.push((
            available_nodes.len(),
            column_index,
            stage_row_indices,
            stage_col_indices,
            stage_data,
        ));
        stage_transformations.push(transformations);
        available_nodes = next_available_nodes;
    }

    Ok((
        compute_stages
            .into_iter()
            .map(|(rows, cols, ri, ci, data)| M::from_stage_data(rows, cols, ri, ci, data))
            .collect(),
        stage_transformations,
    ))
}

/// Contains an example of a [`Recurrent`] [`NetworkLike`] structure.
pub mod net {
    use std::collections::HashMap;

    use super::{EdgeLike, NetworkLike, NodeLike, Recurrent};

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

    /// [`Net`] is an example of a [`Recurrent`] [`NetworkLike`] structure and also used as an intermediate representation to perform the [`unroll`] operation on [`Recurrent`] [`NetworkLike`] structures.
    #[derive(Debug)]
    pub struct Net {
        inputs: usize,
        outputs: usize,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        recurrent_edges: Vec<Edge>,
    }

    impl NetworkLike<Node, Edge> for Net {
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

    /// unroll is an essential operation in order to evaluate [`Recurrent`] [`NetworkLike`] structures.
    ///
    /// It restructures the edges and nodes to be evaluatable in a feedforward manner.
    /// The evaluation further depends on the implementations in [`crate::matrix::recurrent::evaluator`] and [`crate::sparse_matrix::recurrent::evaluator`] which handle the internal state.
    pub fn unroll<R: Recurrent<N, E>, N: NodeLike, E: EdgeLike>(recurrent: &R) -> Net {
        // remember known ids as they can not be reused as otherwise
        // during rewriting edge inputs/outputs stuff would be confused
        let known_ids = recurrent
            .nodes()
            .iter()
            .map(|node| node.id())
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

        let mut known_recurrent_edges = recurrent
            .recurrent_edges()
            .iter()
            .map(|e| Edge {
                start: e.start(),
                end: e.end(),
                weight: e.weight(),
            })
            .collect::<Vec<_>>();

        let mut new_low_ids = (usize::MIN..usize::MAX).filter(|tmp_id| !known_ids.contains(tmp_id));

        // give static input nodes the lowest possible ids to not fuck up output order by sorting in feedforward fabricator
        let mut known_inputs = recurrent.inputs();
        known_inputs.sort_unstable();
        let mut known_inputs = known_inputs
            .iter()
            .map(|n| {
                let new_id = new_low_ids.next().unwrap();

                // patch all edges to new id
                for edge in &mut known_edges {
                    if edge.start == n.id() {
                        edge.start = new_id
                    }
                    if edge.end == n.id() {
                        edge.end = new_id
                    }
                }

                // patch all recurrent edges to new id
                for edge in &mut known_recurrent_edges {
                    if edge.start == n.id() {
                        edge.start = new_id
                    }
                    if edge.end == n.id() {
                        edge.end = new_id
                    }
                }

                Node {
                    id: new_id,
                    activation: n.activation(),
                }
            })
            .collect::<Vec<_>>();

        // give static output nodes the lowest possible ids to not fuck up output order by sorting in feedforward fabricator
        let mut known_outputs = recurrent.outputs();
        known_outputs.sort_unstable();
        let mut known_outputs = known_outputs
            .iter()
            .map(|n| {
                let new_id = new_low_ids.next().unwrap();

                // patch all edges to new id
                for edge in &mut known_edges {
                    if edge.start == n.id() {
                        edge.start = new_id
                    }
                    if edge.end == n.id() {
                        edge.end = new_id
                    }
                }

                // patch all recurrent edges to new id
                for edge in &mut known_recurrent_edges {
                    if edge.start == n.id() {
                        edge.start = new_id
                    }
                    if edge.end == n.id() {
                        edge.end = new_id
                    }
                }

                Node {
                    id: new_id,
                    activation: n.activation(),
                }
            })
            .collect::<Vec<_>>();

        let mut unroll_map: HashMap<usize, usize> = HashMap::new();

        // create wrapping input for all original outputs, regardless of if they are used
        // this is to simplify the state transfer inside the stateful matrix evaluator
        for output in &known_outputs {
            let wrapper_input_id = new_low_ids.next().unwrap();

            let wrapper_input_node = Node {
                id: wrapper_input_id,
                activation: |val| val,
            };

            known_inputs.push(wrapper_input_node);

            unroll_map.insert(output.id(), wrapper_input_id);
        }

        // create all wrapping nodes and egdes for recurrent connections with patched ids
        for recurrent_edge in known_recurrent_edges {
            let recurrent_input = unroll_map.entry(recurrent_edge.start()).or_insert_with(|| {
                let wrapper_input_id = new_low_ids.next().unwrap();

                let wrapper_input_node = Node {
                    id: wrapper_input_id,
                    activation: |val| val,
                };
                let wrapper_output_node = Node {
                    id: new_low_ids.next().unwrap(),
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
