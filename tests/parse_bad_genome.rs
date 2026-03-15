use std::cmp::Ordering;

use favannat::{
    network::{EdgeLike, Fabricator, NetworkLike, NodeLike},
    MatrixFeedforwardFabricator,
};
use serde::Deserialize;

// ── RON-compatible genome types ──────────────────────────────────────────────

/// Newtype wrapping u64, matching the `(u64)` tuple-struct syntax in RON.
#[derive(Debug, Deserialize)]
struct NodeId(u64);

/// All activation variants that appear in bad_genome.ron.
/// The crate's built-ins cover Linear/Sigmoid/Tanh/Gaussian/Relu/Squared/Inverse;
/// Cosine/Sine/Step/Absolute are implemented inline below.
#[derive(Debug, Deserialize)]
enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Gaussian,
    Relu,
    Squared,
    Inverse,
    Cosine,
    Sine,
    Step,
    Absolute,
}

#[derive(Debug, Deserialize)]
struct GenomeNode {
    id: NodeId,
    #[allow(dead_code)]
    order: u32,
    activation: Activation,
    #[allow(dead_code)]
    id_counter: u32,
}

#[derive(Debug, Deserialize)]
struct GenomeEdge {
    input: NodeId,
    output: NodeId,
    weight: f64,
    #[allow(dead_code)]
    id_counter: u32,
}

/// Newtype for `([Node, ...])` RON syntax — a tuple wrapping an array.
#[derive(Debug, Deserialize)]
struct NodeList(Vec<GenomeNode>);

/// Newtype for `([Edge, ...])` RON syntax — a tuple wrapping an array.
#[derive(Debug, Deserialize)]
struct EdgeList(Vec<GenomeEdge>);

#[derive(Debug, Deserialize)]
struct Genome {
    inputs: NodeList,
    hidden: NodeList,
    outputs: NodeList,
    feed_forward: EdgeList,
}

/// Top-level structure of bad_genome.ron:
///   parents: (genome1, genome2)  – the two parent genomes
///   child:   genome3             – the crossover result
#[derive(Debug, Deserialize)]
struct BadGenomeFile {
    #[allow(dead_code)]
    parents: (Genome, Genome),
    child: Genome,
}

// ── Trait implementations ─────────────────────────────────────────────────────

impl PartialEq for GenomeNode {
    fn eq(&self, other: &Self) -> bool {
        self.id().eq(&other.id())
    }
}

impl Eq for GenomeNode {}

impl PartialOrd for GenomeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GenomeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id().cmp(&other.id())
    }
}

impl NodeLike for GenomeNode {
    fn id(&self) -> usize {
        // Safe on 64-bit platforms (usize == u64).
        self.id.0 as usize
    }

    fn activation(&self) -> fn(f64) -> f64 {
        use favannat::network::net::activations::{
            GAUSSIAN, INVERSE, LINEAR, RELU, SIGMOID, SQUARED, TANH,
        };
        match self.activation {
            Activation::Linear   => LINEAR,
            Activation::Sigmoid  => SIGMOID,
            Activation::Tanh     => TANH,
            Activation::Gaussian => GAUSSIAN,
            Activation::Relu     => RELU,
            Activation::Squared  => SQUARED,
            Activation::Inverse  => INVERSE,
            Activation::Cosine   => |val: f64| val.cos(),
            Activation::Sine     => |val: f64| val.sin(),
            Activation::Step     => |val: f64| if val > 0.0 { 1.0 } else { 0.0 },
            Activation::Absolute => |val: f64| val.abs(),
        }
    }
}

impl EdgeLike for GenomeEdge {
    fn start(&self) -> usize {
        self.input.0 as usize
    }

    fn end(&self) -> usize {
        self.output.0 as usize
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

impl NetworkLike<GenomeNode, GenomeEdge> for Genome {
    fn edges(&self) -> Vec<&GenomeEdge> {
        self.feed_forward.0.iter().collect()
    }

    fn inputs(&self) -> Vec<&GenomeNode> {
        self.inputs.0.iter().collect()
    }

    fn hidden(&self) -> Vec<&GenomeNode> {
        self.hidden.0.iter().collect()
    }

    fn outputs(&self) -> Vec<&GenomeNode> {
        self.outputs.0.iter().collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_bad_genome_and_fabricate() {
    let ron_str = include_str!("../src/bad_genome.ron");

    // Step 1: RON parsing must succeed — the file is syntactically valid.
    let parsed: BadGenomeFile =
        ron::from_str(ron_str).expect("RON parsing of bad_genome.ron should succeed");

    // Step 2: Attempt to fabricate the child genome (crossover result).
    let result = MatrixFeedforwardFabricator::fabricate(&parsed.child);

    // The child genome is the result of NEAT crossover of the two parents.
    // Analysis reveals an 11-node cycle in its feed_forward connections
    // (10 hidden nodes + 1 output node), making it invalid for feedforward fabrication.
    //
    // Cycle: H(14211212388494148245) → H(3826633468684569139) → H(17077754657716914943)
    //      → H(11797454570146984742) → H(15996739864361207706) → H(8672086569406954279)
    //      → O(10763953748989339248) → H(17387659509746847695) → H(9986850205867386787)
    //      → H(10477926725837074344) → H(11602570492947847242) → (back to start)
    //
    // The fabricator correctly returns "can't resolve dependencies, net invalid".
    assert_eq!(
        result.unwrap_err(),
        "can't resolve dependencies, net invalid"
    );
}
