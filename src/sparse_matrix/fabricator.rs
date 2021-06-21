// external imports
use ndarray::Array;
use ndarray::Array2;
use sprs::CsMat;
use sprs::TriMat;
// crate imports
use crate::network::{
    net::unroll, EdgeLike, Fabricator, NetLike, NodeLike, Recurrent, StatefulFabricator,
};
// std imports
use std::collections::HashMap;

pub struct FeedForwardSparseMatrixFabricator;

pub struct RecurrentSparseMatrixFabricator;

impl<N, E> StatefulFabricator<N, E> for RecurrentSparseMatrixFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::RecurrentMatrixEvaluator;

    fn fabricate(net: &impl Recurrent<N, E>) -> Result<Self::Output, &'static str> {
        let unrolled = unroll(net);
        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&unrolled)?;
        let memory = unrolled.outputs().len();

        assert!(unrolled.inputs().len() - net.inputs().len() == memory);

        Ok(super::evaluator::RecurrentMatrixEvaluator {
            internal: Array::zeros(memory),
            evaluator,
            outputs: net.outputs().len(),
        })
    }
}

impl FeedForwardSparseMatrixFabricator {
    fn get_sparse((col_inds, row_inds, data): (Vec<usize>, Vec<usize>, Vec<f64>)) -> CsMat<f64> {
        let colums = col_inds.iter().max().unwrap() + 1;
        let rows = row_inds.iter().max().unwrap() + 1;

        TriMat::from_triplets((rows, colums), row_inds, col_inds, data).to_csr()
    }
}

impl<N, E> Fabricator<N, E> for FeedForwardSparseMatrixFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::SparseMatrixEvaluator;

    fn fabricate(net: &impl NetLike<N, E>) -> Result<Self::Output, &'static str> {
        // build dependency graph by collecting incoming edges per node
        let mut dependency_graph: HashMap<usize, Vec<&E>> = HashMap::new();

        for edge in net.edges() {
            dependency_graph
                .entry(edge.end())
                .and_modify(|dependencies| dependencies.push(&edge))
                .or_insert_with(|| vec![&edge]);
        }

        if dependency_graph.is_empty() {
            return Err("no edges present, net invalid");
        }

        // keep track of dependencies present
        let mut dependency_count = dependency_graph.len();

        // println!("initial dependency_graph {:#?}", dependency_graph);

        // contains list of matrices (stages) that form the computable net
        let mut compute_stages: Vec<(Vec<usize>, Vec<usize>, Vec<f64>)> = Vec::new();
        // contains activation functions corresponding to each stage
        let mut stage_transformations: Vec<crate::Transformations> = Vec::new();
        // set available nodes a.k.a net input
        let mut available_nodes: Vec<usize> = net.inputs().iter().map(|n| n.id()).collect();
        // sort to guarantee each input will be processed by the same node every time
        available_nodes.sort_unstable();

        // println!("available_nodes {:?}", available_nodes);

        // set wanted nodes a.k.a net output
        let mut wanted_nodes: Vec<usize> = net.outputs().iter().map(|n| n.id()).collect();
        // sort to guarantee each output will appear in the same order every time
        wanted_nodes.sort_unstable();
        let wanted_nodes = wanted_nodes;

        // println!("wanted_nodes {:?}", wanted_nodes);

        // gather compute stages by finding computable nodes and required carries until all dependencies are resolved
        while !dependency_graph.is_empty() {
            // setup new transformations
            let mut transformations: crate::Transformations = Vec::new();
            // list of nodes becoming available by compute stage
            let mut next_available_nodes: Vec<usize> = Vec::new();

            let mut column_index = 0;
            let mut stage_column_indices: Vec<usize> = Vec::new();
            let mut stage_row_indices = Vec::new();
            let mut stage_data = Vec::new();

            for (&dependent_node, dependencies) in dependency_graph.iter() {
                let mut node_column_indices = Vec::new();
                let mut node_row_indices = Vec::new();
                let mut node_data = Vec::new();
                // marker if all dependencies are available
                let mut computable = true;
                // check every dependency
                for &dependency in dependencies {
                    let mut found = false;
                    for (row_index, &id) in available_nodes.iter().enumerate() {
                        // index here is row index
                        if dependency.start() == id {
                            node_column_indices.push(column_index);
                            node_row_indices.push(row_index);
                            node_data.push(dependency.weight());
                            found = true;
                        }
                    }
                    // if any dependency is not found the node is not computable yet
                    if !found {
                        computable = false;
                    }
                }
                if computable {
                    stage_column_indices = [stage_column_indices, node_column_indices].concat();
                    stage_row_indices = [stage_row_indices, node_row_indices].concat();
                    stage_data = [stage_data, node_data].concat();
                    // add activation function to stage transformations
                    transformations.push(
                        net.nodes()
                            .iter()
                            .find(|&node| node.id() == dependent_node)
                            .unwrap()
                            .activation(),
                    );
                    column_index += 1;
                    // mark node as available in next iteration
                    next_available_nodes.push(dependent_node);
                } else {
                    let mut carry_column_indices = Vec::new();
                    let mut carry_row_indices = Vec::new();
                    let mut carry_data = Vec::new();
                    for row_index in node_row_indices {
                        if !next_available_nodes.contains(&available_nodes[row_index]) {
                            carry_row_indices.push(row_index);
                            carry_column_indices.push(column_index);
                            column_index += 1;
                            carry_data.push(1.0);
                            transformations.push(|val| val);
                            next_available_nodes.push(available_nodes[row_index]);
                        }
                    }
                    stage_column_indices = [stage_column_indices, carry_column_indices].concat();
                    stage_row_indices = [stage_row_indices, carry_row_indices].concat();
                    stage_data = [stage_data, carry_data].concat();
                }
            }

            // keep any wanted notes if available (output)
            for wanted_node in wanted_nodes.iter() {
                for (row_index, available_node) in available_nodes.iter().enumerate() {
                    if available_node == wanted_node {
                        // carry only if not carried already
                        if !next_available_nodes.contains(&available_node) {
                            stage_column_indices.push(column_index);
                            column_index += 1;
                            stage_row_indices.push(row_index);
                            stage_data.push(1.0);

                            // add identity function for carried vector
                            transformations.push(|val| val);
                            // add node as available
                            next_available_nodes.push(*available_node);
                        }
                    }
                }
            }

            // remove resolved dependencies from dependency graph
            for node in next_available_nodes.iter() {
                dependency_graph.remove(node);
            }

            // if no dependency was removed no progess was made
            if dependency_graph.len() == dependency_count {
                return Err("can't resolve dependencies, net invalid");
            } else {
                dependency_count = dependency_graph.len();
            }

            // println!("next_available_nodes {:?}", next_available_nodes);

            // reorder last stage according to net output order (invalidates next_available_nodes order which wont be used after this point)
            if dependency_graph.is_empty() {
                // println!("stage_matrix {:?}", stage_matrix);

                // let mut reordered_matrix = stage_matrix.clone();
                let mut reordered_stage_column_indices =
                    vec![usize::MAX; stage_column_indices.len()];
                let mut reordered_transformations = transformations.clone();

                let mut matched_wanted_count = 0;

                for (old_column_index, available_node) in next_available_nodes.iter().enumerate() {
                    for (new_column_index, wanted_node) in wanted_nodes.iter().enumerate() {
                        if available_node == wanted_node {
                            for (reordered_index, &old_index) in reordered_stage_column_indices
                                .iter_mut()
                                .zip(stage_column_indices.iter())
                            {
                                if old_index == old_column_index {
                                    *reordered_index = new_column_index;
                                }
                            }

                            reordered_transformations[new_column_index] =
                                transformations[old_column_index];
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

                // println!("reordered_matrix {:?}", reordered_matrix);

                stage_column_indices = reordered_stage_column_indices;
                transformations = reordered_transformations;
            }

            // add resolved dependencies and transformations to compute stages
            compute_stages.push((stage_column_indices, stage_row_indices, stage_data));
            stage_transformations.push(transformations);

            // set available nodes for next iteration
            available_nodes = next_available_nodes;
        }
        // compute_stages
        // .into_iter()
        // .map(FeedForwardSparseMatrixFabricator::get_sparse)
        // .collect()
        Ok(super::evaluator::SparseMatrixEvaluator {
            stages: compute_stages
                .into_iter()
                .map(FeedForwardSparseMatrixFabricator::get_sparse)
                .collect(),
            transformations: stage_transformations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{FeedForwardSparseMatrixFabricator, RecurrentSparseMatrixFabricator};
    use crate::{
        edges,
        network::{net::Net, Evaluator, Fabricator, StatefulEvaluator, StatefulFabricator},
        nodes,
    };
    use ndarray::array;

    // tests construction and evaluation of simplest network
    #[test]
    fn simple_net_evaluator_0() {
        let some_net = Net::new(1, 1, nodes!('l', 'l'), edges!(0--0.5->1));

        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(array![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, array![2.5, 1.25]);
    }

    // test unconnected net
    #[test]
    fn simple_net_evaluator_6() {
        let some_net = Net::new(1, 1, nodes!('l', 'l'), Vec::new());

        if let Err(message) = FeedForwardSparseMatrixFabricator::fabricate(&some_net) {
            assert_eq!(message, "no edges present, net invalid");
        } else {
            unreachable!();
        }
    }

    // test uncomputable output
    #[test]
    fn simple_net_evaluator_7() {
        let some_net = Net::new(1, 1, nodes!('l', 'l', 'l'), edges!(0--0.5->1));

        if let Err(message) = FeedForwardSparseMatrixFabricator::fabricate(&some_net) {
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

        if let Err(message) = FeedForwardSparseMatrixFabricator::fabricate(&some_net) {
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

        let evaluator = FeedForwardSparseMatrixFabricator::fabricate(&some_net).unwrap();
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
                0--1.0->2,
                1--1.0->3
            ),
        );

        some_net.set_recurrent_edges(edges!(
            0--1.0->2,
            1--1.0->3
        ));
        let mut evaluator = RecurrentSparseMatrixFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator);

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
