use ndarray_linalg::{EighInplace, UPLO};
use numpy::ndarray::linalg::general_mat_mul;
use numpy::ndarray::prelude::*;

pub fn pca_transform(
    xs: &ArrayViewD<f64>, // (n_samples, n_features)
    n_components: usize,
) -> ArrayD<f64> {
    let n_features = xs.shape()[1];

    let xs: ArrayView2<f64> = xs.clone().into_dimensionality().expect("must be 2d array");
    let mean = xs.mean_axis(Axis(0)).unwrap();

    let xs = &xs - &mean;

    let mut c = Array2::<f64>::zeros((xs.ncols(), xs.ncols()));
    general_mat_mul(1.0, &xs.t(), &xs, 0.0, &mut c);

    let (_eigenvalues, eigenvectors) = c.eigh_inplace(UPLO::Upper).expect("eigh failed");

    if n_components > n_features {
        panic!("n_components must be less than or equal to n_features");
    }

    let components = eigenvectors.slice(s![.., -(n_components as isize)..]);
    let components = components.t().to_owned();

    let xs = xs.dot(&components.t());

    xs.into_dyn()
}
