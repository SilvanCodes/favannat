use crate::network::{fabricate_stages, EdgeLike, Fabricator, NetworkLike, NodeLike, StageMatrix};
use nalgebra::DMatrix;

pub struct MatrixFeedforwardFabricator;

impl StageMatrix for DMatrix<f64> {
    fn from_stage_data(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(
            row_indices.len(),
            col_indices.len(),
            "from_stage_data: row_indices and col_indices must have the same length"
        );
        debug_assert_eq!(
            row_indices.len(),
            data.len(),
            "from_stage_data: row_indices and data must have the same length"
        );
        let mut m = DMatrix::zeros(rows, cols);
        for ((&c, &r), &v) in col_indices.iter().zip(row_indices.iter()).zip(data.iter()) {
            m[(r, c)] = v;
        }
        m
    }
}

impl<N, E> Fabricator<N, E> for MatrixFeedforwardFabricator
where
    N: NodeLike,
    E: EdgeLike,
{
    type Output = super::evaluator::MatrixFeedforwardEvaluator;

    fn fabricate(net: &impl NetworkLike<N, E>) -> Result<Self::Output, &'static str> {
        let (stages, transformations) = fabricate_stages::<N, E, DMatrix<f64>>(net)?;
        Ok(super::evaluator::MatrixFeedforwardEvaluator {
            stages,
            transformations,
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::MatrixFeedforwardFabricator;
    use crate::{
        edges,
        network::{net::Net, Evaluator, Fabricator},
        nodes,
    };

    #[test]
    fn reports_error_on_empty_edges() {
        let net = Net::new(1, 1, nodes!('l', 'l'), Vec::new());

        assert_eq!(
            MatrixFeedforwardFabricator::fabricate(&net).err(),
            Some("no edges present, net invalid")
        );
    }

    #[test]
    fn reports_error_on_missing_edges_to_output() {
        let net = Net::new(1, 2, nodes!('l', 'l', 'l'), edges!(0--1.0->1));

        assert_eq!(
            MatrixFeedforwardFabricator::fabricate(&net).err(),
            Some("dependencies resolved but not all outputs computable, net invalid")
        );
    }

    // test uncomputable output
    #[test]
    fn reports_error_on_unresolvable_dependency() {
        let net = Net::new(1, 1, nodes!('l', 'l', 'l'), edges!(1--0.5->2));

        assert_eq!(
            MatrixFeedforwardFabricator::fabricate(&net).err(),
            Some("can't resolve dependencies, net invalid")
        );
    }

    // tests construction and evaluation of simplest network
    #[test]
    fn simple_net_evaluator_0() {
        let some_net = Net::new(1, 1, nodes!('l', 'l'), edges!(0--0.5->1));

        let evaluator = MatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = MatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = MatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = MatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
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

        let evaluator = MatrixFeedforwardFabricator::fabricate(&some_net).unwrap();

        let result = evaluator.evaluate(dmatrix![5.0]);

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

        let evaluator = MatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![2.5, 1.25]);
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

        let evaluator = MatrixFeedforwardFabricator::fabricate(&some_net).unwrap();
        // println!("stages {:?}", evaluator.stages);

        let result = evaluator.evaluate(dmatrix![5.0, 5.0]);
        // println!("result {:?}", result);

        assert_eq!(result, dmatrix![2.5]);
    }

    // Regression test: hidden node H2 is computed in the same final stage as output O.
    // H1 feeds both H2 and O; H2 has no outgoing connection to O (dead-end hidden node).
    // Bug: reorder assigns usize::MAX for H2's column → panic in matrix construction.
    #[test]
    fn hidden_node_computed_in_final_stage_with_output() {
        // topology: I(0) → H1(1) → H2(2)
        //                        ↘ O(3)
        //           H2 is a dead-end (not connected to O)
        let some_net = Net::new(
            1, // inputs
            1, // outputs
            nodes!('l', 'l', 'l', 'l'), // I, H1, H2, O
            edges!(
                0--1.0->1,  // I → H1
                1--1.0->2,  // H1 → H2  (H2 is a dead-end)
                1--1.0->3   // H1 → O
            ),
        );
        // This should succeed without panic
        let result = MatrixFeedforwardFabricator::fabricate(&some_net);
        assert!(result.is_ok(), "Expected fabrication to succeed, got: {:?}", result.err());
        let evaluator = result.unwrap();
        let out = evaluator.evaluate(dmatrix![2.0]);
        assert_eq!(out, dmatrix![2.0]);
    }
}
