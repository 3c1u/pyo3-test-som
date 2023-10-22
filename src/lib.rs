use pyo3::prelude::*;

pub mod som;

#[pymodule]
fn ddkdk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<som::Som>()?;
    Ok(())
}
