use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use numpy::ndarray::prelude::*;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};

mod pca;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Som {
    dim_latent: usize,
    dim_input: usize,
    latent: ArrayD<f64>,
}

impl Som {
    fn new(dim_latent: usize, dim_input: usize) -> Self {
        Self {
            dim_latent,
            dim_input,
            latent: Default::default(),
        }
    }

    fn fit_internal(
        &self,
        input: ArrayViewD<f64>,
        grid: ArrayViewD<f64>,
        max_iter: usize,
        tau: f64,
    ) -> Result<ArrayD<f64>, String> {
        let input_size: usize;
        let grid_size: usize;

        if let &[input_size_in, dim] = input.shape() {
            input_size = input_size_in;

            if dim != self.dim_input {
                return Err(format!(
                    "input dimension mismatch: expected {}, got {}",
                    dim, self.dim_input
                ));
            }
        } else {
            return Err("input must be 2d array".into());
        }

        if let &[grid_size_in, dim] = grid.shape() {
            grid_size = grid_size_in;

            if dim != self.dim_latent {
                return Err(format!(
                    "grid dimension mismatch: expected {}, got {}",
                    dim, self.dim_latent
                ));
            }
        } else {
            return Err("grid must be 2d array".into());
        }

        let mut y: ArrayD<f64> = ArrayD::zeros(vec![grid_size, self.dim_input]);
        let mut z: ArrayD<f64> = pca::pca_transform(&input.view(), self.dim_latent);

        // TODO: イテレーション回数を設定できるようにする
        for i in 0..max_iter {
            if i != 0 {
                for i in 0..input_size {
                    let mut min_dist = std::f64::MAX;
                    let mut min_index = 0;

                    for j in 0..grid_size {
                        let dist = (&input.slice(s![i, ..]) - &y.slice(s![j, ..]))
                            .mapv(|x| x.powi(2))
                            .sum();

                        if dist < min_dist {
                            min_dist = dist;
                            min_index = j;
                        }
                    }

                    z.slice_mut(s![i, ..])
                        .assign(&grid.slice(s![min_index, ..]));
                }
            }

            // TODO: learning_rateを設定できるようにする
            let l: f64 = 1.0;
            let sigma: f64 = f64::max(l / 10.0, l * (-(i as f64) / tau).exp());
            let inv_sigma_sq = 1. / sigma.powi(2);

            let mut r: ArrayD<f64> = ArrayD::zeros(vec![input_size, grid_size]);

            for i in 0..input_size {
                for j in 0..grid_size {
                    let z_n = &z.slice(s![i, ..]);
                    let g_k = &grid.slice(s![j, ..]);
                    let dist = (z_n - g_k).mapv(|x| x.powi(2)).sum();
                    let r_nk = -0.5 * inv_sigma_sq * dist;

                    r[[i, j]] = r_nk.exp();
                }
            }

            let r_sum = r.sum_axis(Axis(0));

            for j in 0..grid_size {
                let rnk_xn = &r.slice(s![.., j]).insert_axis(Axis(1)) * &input;
                let rnk_xn_sum = rnk_xn.sum_axis(Axis(0));

                y.slice_mut(s![j, ..])
                    .assign(&(&rnk_xn_sum / &r_sum.slice(s![j])));
            }
        }

        Ok(z)
    }
}

#[pymethods]
impl Som {
    #[new]
    pub fn __new__(dim_latent: usize, dim_input: usize) -> Self {
        Self::new(dim_latent, dim_input)
    }

    #[getter]
    pub fn latent<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<f64>> {
        if let &[_, dim] = self.latent.shape() {
            if dim != self.dim_latent {
                return Err(PyValueError::new_err("latent dimension mismatch"));
            }
        } else {
            return Err(PyValueError::new_err("run fit() first"));
        }

        Ok(self.latent.to_pyarray(py))
    }

    #[pyo3(signature=(input, grid, **kwargs))]
    pub fn fit(
        &mut self,
        input: PyReadonlyArrayDyn<f64>,
        grid: PyReadonlyArrayDyn<f64>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<()> {
        let input = input.as_array();
        let grid = grid.as_array();

        let tau = if let Some(kwargs) = kwargs {
            if let Some(tau) = kwargs.get_item("tau")? {
                tau.extract::<f64>()?
            } else {
                100.0
            }
        } else {
            100.0
        };

        let max_iter = if let Some(kwargs) = kwargs {
            if let Some(max_iter) = kwargs.get_item("max_iter")? {
                max_iter.extract::<usize>()?
            } else {
                100
            }
        } else {
            100
        };

        let res = self
            .fit_internal(input, grid, max_iter, tau)
            .map_err(|e| PyValueError::new_err(e))?;

        self.latent = res;

        Ok(())
    }
}
