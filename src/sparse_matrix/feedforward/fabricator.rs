use crate::network::{fabricate_stages, EdgeLike, Fabricator, NetworkLike, NodeLike, StageMatrix};
use nalgebra_sparse::{CooMatrix, CscMatrix};

pub struct SparseMatrixFeedforwardFabricator;

impl StageMatrix for CscMatrix<f64> {
    fn from_stage_data(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Self {
        CscMatrix::from(
            &CooMatrix::try_from_triplets(rows, cols, row_indices, col_indices, data).unwrap(),
        )
    }
}

impl<N, E> Fabricator<N, E> for SparseMatrixFeedforwardFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::SparseMatrixFeedforwardEvaluator;

    fn fabricate(net: &impl NetworkLike<N, E>) -> Result<Self::Output, &'static str> {
        let (stages, transformations) = fabricate_stages::<N, E, CscMatrix<f64>>(net)?;
        Ok(super::evaluator::SparseMatrixFeedforwardEvaluator {
            stages,
            transformations,
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::SparseMatrixFeedforwardFabricator;
    use crate::{
        edges,
        network::{net::Net, Evaluator, Fabricator},
        nodes,
    };

    // tests construction and evaluation of simplest network
    #[test]
    fn simple_net_evaluator_0() {
        let some_net = Net::new(1, 1, nodes!('l', 'l'), edges!(0--0.5->1));

        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![2.5]);
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

        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0, 5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![5.0]);
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

        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![1.25]);
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

        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![3.75]);
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

        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![3.75, 2.5]);
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

        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();

        let result = evaluator.evaluate(dmatrix![5.0]);

        assert_eq!(result, dmatrix![2.5, 1.25]);
    }

    // test unconnected net
    #[test]
    fn simple_net_evaluator_6() {
        let some_net = Net::new(1, 1, nodes!('l', 'l'), Vec::new());

        if let Err(message) = SparseMatrixFeedforwardFabricator::fabricate(&some_net) {
            assert_eq!(message, "no edges present, net invalid");
        } else {
            unreachable!();
        }
    }

    // test uncomputable output
    #[test]
    fn simple_net_evaluator_7() {
        let some_net = Net::new(1, 1, nodes!('l', 'l', 'l'), edges!(0--0.5->1));

        if let Err(message) = SparseMatrixFeedforwardFabricator::fabricate(&some_net) {
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

        if let Err(message) = SparseMatrixFeedforwardFabricator::fabricate(&some_net) {
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

        let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0, 5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![2.5]);
    }

    // This test fails as currently it is necessary to run connections into all outputs.
    //
    // #[test]
    // fn simple_net_evaluator_not_fully_connected_outputs() {
    //     let some_net = Net::new(
    //         1,
    //         2,
    //         nodes!('l', 'l', 'l'),
    //         edges!(
    //             0--1.0->1
    //         ),
    //     );

    //     let evaluator = SparseMatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
    //     // println!("stages {:?}", evaluator.stages);

    //     let result = evaluator.evaluate(dmatrix![5.0]);
    //     // println!("result {:?}", result);

    //     assert_eq!(result, dmatrix![5.0, 0.0]);
    // }
}
