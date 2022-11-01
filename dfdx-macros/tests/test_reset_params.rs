// Test macro expansion with `cargo expand --test test_reset_params`
use dfdx::tensor::*;
use rand::prelude::*;

use dfdx::prelude::*;
use dfdx::prelude::ResetParams;
use dfdx_macros::ResetParams;
use dfdx::devices::{Cpu, FillElements};

#[test]
fn test_named_fields() {
    #[derive(ResetParams, Default)]
    #[reset_params(initializer="ones")]
    pub struct Linear<const I: usize, const O: usize> {
        // Transposed weight matrix, shape (O, I)
        pub weight: Tensor2D<O, I>,

        // Bias vector, shape (O, )
        pub bias: Tensor1D<O>,
    }
    let mut model: Linear<5, 2> = Linear::default();
    let mut rng = StdRng::seed_from_u64(0);
    model.reset_params(&mut rng);
}

/*#[test]
fn test_unnamed_fields() {
    #[derive(ResetParams, Default)]
    pub struct Residual<F>(pub F);

    let mut model: Residual<Linear<2, 5>> = Default::default();
    let mut rng = StdRng::seed_from_u64(0);
    model.reset_params(&mut rng);
}*/
