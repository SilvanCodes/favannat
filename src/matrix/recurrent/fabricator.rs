use nalgebra::DMatrix;

use crate::{
    matrix::feedforward::fabricator::MatrixFeedforwardFabricator,
    network::{
        net::unroll, EdgeLike, Fabricator, NetworkLike, NodeLike, Recurrent, StatefulFabricator,
    },
};

use super::evaluator::MatrixRecurrentEvaluator;

pub struct MatrixRecurrentFabricator;

impl<N, E> StatefulFabricator<N, E> for MatrixRecurrentFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = MatrixRecurrentEvaluator;

    fn fabricate(net: &impl Recurrent<N, E>) -> Result<Self::Output, &'static str> {
        let unrolled = unroll(net);
        let evaluator = MatrixFeedforwardFabricator::fabricate(&unrolled)?;
        let memory = unrolled.outputs().len();

        assert!(unrolled.inputs().len() - net.inputs().len() == memory);

        Ok(MatrixRecurrentEvaluator {
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
        matrix::recurrent::fabricator::MatrixRecurrentFabricator,
        network::{net::Net, StatefulEvaluator, StatefulFabricator},
        nodes,
    };

    #[test]
    fn computes_without_recurrent_edges() {
        let some_net = Net::new(
            1,
            1,
            nodes!('l', 'l'),
            edges!(
                0--1.0->1
            ),
        );

        let mut evaluator = MatrixRecurrentFabricator::fabricate(&some_net).unwrap();
        println!("stages {:?}", evaluator);

        let result = evaluator.evaluate(dmatrix![5.0]);
        assert_eq!(result, dmatrix![5.0]);
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
        let mut evaluator = MatrixRecurrentFabricator::fabricate(&some_net).unwrap();

        let result = evaluator.evaluate(dmatrix![5.0, 0.0]);
        assert_eq!(result, dmatrix![5.0, 0.0]);

        let result = evaluator.evaluate(dmatrix![5.0, 5.0]);
        assert_eq!(result, dmatrix![10.0, 5.0]);

        let result = evaluator.evaluate(dmatrix![0.0, 5.0]);
        assert_eq!(result, dmatrix![5.0, 10.0]);

        let result = evaluator.evaluate(dmatrix![0.0, 0.0]);
        assert_eq!(result, dmatrix![0.0, 5.0]);
    }
}
