use nalgebra::{DMatrix, DVector};

/// Data structures implementing this trait can be used as input and output of networks.
pub trait NetworkIO {
    fn input(input: Self) -> DMatrix<f64>;
    fn output(output: DMatrix<f64>) -> Self;
}

impl NetworkIO for DMatrix<f64> {
    fn input(input: Self) -> DMatrix<f64> {
        input
    }
    fn output(output: DMatrix<f64>) -> Self {
        output
    }
}

impl NetworkIO for DVector<f64> {
    fn input(input: Self) -> DMatrix<f64> {
        DMatrix::from_iterator(1, input.len(), input.into_iter().cloned())
    }
    fn output(output: DMatrix<f64>) -> Self {
        DVector::from(output.into_iter().cloned().collect::<Vec<f64>>())
    }
}

impl NetworkIO for Vec<f64> {
    fn input(input: Self) -> DMatrix<f64> {
        DMatrix::from_iterator(1, input.len(), input.into_iter())
    }
    fn output(output: DMatrix<f64>) -> Self {
        output.into_iter().cloned().collect::<Vec<f64>>()
    }
}

#[cfg(feature = "ndarray")]
use ndarray::Array1;

#[cfg(feature = "ndarray")]
impl NetworkIO for Array1<f64> {
    fn input(input: Self) -> DMatrix<f64> {
        DMatrix::from_iterator(1, input.len(), input.into_iter)
    }
    fn output(output: DMatrix<f64>) -> Self {
        Array1::from_iter(output.into_iter().cloned())
    }
}
