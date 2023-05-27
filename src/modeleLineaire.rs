use rusty_machine::learning::lin_reg::LinRegressor as LinReg;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::SupModel;

pub struct LinRegressor {
    inner: LinReg,
}

impl LinRegressor {
    pub fn default() -> Self {
        Self {
            inner: LinReg::default(),
        }
    }

    pub fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<f64>) {
        self.inner.train(inputs, targets).unwrap();
    }

    pub fn predict(&self, inputs: &Matrix<f64>) -> Vector<f64> {
        self.inner.predict(inputs).unwrap()
    }
}