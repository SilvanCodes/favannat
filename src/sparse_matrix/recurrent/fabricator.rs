use nalgebra::DMatrix;

use crate::{
    network::{
        net::unroll, EdgeLike, Fabricator, NetworkLike, NodeLike, Recurrent, StatefulFabricator,
    },
    sparse_matrix::feedforward::fabricator::SparseMatrixFeedforwardFabricator,
};

pub struct SparseMatrixRecurrentFabricator;

impl<N, E> StatefulFabricator<N, E> for SparseMatrixRecurrentFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::SparseMatrixRecurrentEvaluator;

    fn fabricate(net: &impl Recurrent<N, E>) -> Result<Self::Output, &'static str> {
        let unrolled = unroll(net);
        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&unrolled)?;
        let memory = unrolled.outputs().len();

        assert!(unrolled.inputs().len() - net.inputs().len() == memory);

        Ok(super::evaluator::SparseMatrixRecurrentEvaluator {
            internal: DMatrix::from_element(1, memory, 0.0),
            evaluator,
            outputs: net.outputs().len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use crate::{
        edges,
        network::{net::Net, StatefulEvaluator, StatefulFabricator},
        nodes,
        sparse_matrix::recurrent::fabricator::SparseMatrixRecurrentFabricator,
    };

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
        let mut evaluator = SparseMatrixRecurrentFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator);

        let result = evaluator.evaluate(dmatrix![5.0, 0.0]);
        assert_eq!(result, dmatrix![5.0, 0.0]);

        let result = evaluator.evaluate(dmatrix![5.0, 5.0]);
        assert_eq!(result, dmatrix![10.0, 5.0]);

        let result = evaluator.evaluate(dmatrix![0.0, 5.0]);
        assert_eq!(result, dmatrix![5.0, 10.0]);

        let result = evaluator.evaluate(dmatrix![0.0, 0.0]);
        assert_eq!(result, dmatrix![0.0, 5.0]);
    }

    #[test]
    fn stateful_net_evaluator_with_hidden_node() {
        let mut some_net = Net::new(
            1,
            1,
            nodes!('l', 'l', 'l'),
            edges!(
                0--1.0->1,
                1--1.0->2
            ),
        );

        some_net.set_recurrent_edges(edges!(
            1--1.0->1
        ));
        let mut evaluator = SparseMatrixRecurrentFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator);

        let result = evaluator.evaluate(dmatrix![1.0]);
        assert_eq!(result, dmatrix![1.0]);

        let result = evaluator.evaluate(dmatrix![1.0]);
        assert_eq!(result, dmatrix![2.0]);

        let result = evaluator.evaluate(dmatrix![1.0]);
        assert_eq!(result, dmatrix![3.0]);
    }

    #[test]
    fn stateful_net_evaluator_not_all_inputs_connected() {
        let mut some_net = Net::new(
            2,
            1,
            nodes!('l', 'l', 'l'),
            edges!(
                0--1.0->2
            ),
        );

        some_net.set_recurrent_edges(edges!(
            0--1.0->2
        ));
        let mut evaluator = SparseMatrixRecurrentFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator);

        let result = evaluator.evaluate(dmatrix![1.0, 1.0]);
        assert_eq!(result, dmatrix![1.0]);

        let result = evaluator.evaluate(dmatrix![1.0, 1.0]);
        assert_eq!(result, dmatrix![2.0]);

        let result = evaluator.evaluate(dmatrix![1.0, 1.0]);
        assert_eq!(result, dmatrix![2.0]);
    }

    #[test]
    fn stateful_net_evaluator_not_all_inputs_connected_with_hidden_node() {
        let mut some_net = Net::new(
            2,
            1,
            nodes!('l', 'l', 'l', 'l'),
            edges!(
                0--1.0->2,
                2--1.0->3
            ),
        );

        some_net.set_recurrent_edges(edges!(
            0--1.0->2,
            2--1.0->2
        ));
        let mut evaluator = SparseMatrixRecurrentFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator);

        let result = evaluator.evaluate(dmatrix![1.0, 1.0]);
        assert_eq!(result, dmatrix![1.0]);

        let result = evaluator.evaluate(dmatrix![1.0, 1.0]);
        assert_eq!(result, dmatrix![3.0]);

        let result = evaluator.evaluate(dmatrix![1.0, 1.0]);
        assert_eq!(result, dmatrix![5.0]);
    }

    #[test]
    fn stateful_net_evaluator_two_hidden_nodes() {
        let mut some_net = Net::new(
            1,
            1,
            nodes!('l', 'l', 'l', 'l'),
            edges!(
                0--1.0->1,
                0--1.0->2,
                1--1.0->3,
                2--1.0->3
            ),
        );

        some_net.set_recurrent_edges(edges!(
            3--1.0->1,
            3--1.0->2
        ));
        let mut evaluator = SparseMatrixRecurrentFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator);

        let result = evaluator.evaluate(dmatrix![1.0]);
        assert_eq!(result, dmatrix![2.0]);

        let result = evaluator.evaluate(dmatrix![1.0]);
        assert_eq!(result, dmatrix![6.0]);

        let result = evaluator.evaluate(dmatrix![1.0]);
        assert_eq!(result, dmatrix![14.0]);
    }
}
