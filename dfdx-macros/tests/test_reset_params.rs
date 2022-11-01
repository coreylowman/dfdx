// Test macro expansion with `cargo expand --test test_reset_params`
use dfdx::tensor::*;
use rand::prelude::*;

use dfdx::prelude::*;
use dfdx_macros::ResetParams;
use dfdx::devices::{Cpu, FillElements};
use dfdx::arrays::AllAxes;


#[test]
fn test_named_fields_default_attribute() {
    use rand_distr::StandardNormal;
    #[derive(ResetParams, Default)]
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

#[test]
fn test_named_fields_with_attribute() {
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
    assert_eq!(model.weight.sum::<_, AllAxes>().data(), &10.0)
}

#[test]
fn test_unnamed_fields() {
    #[derive(ResetParams, Default)]
    pub struct Residual<F>(pub F);

    let mut model: Residual<Linear<2, 5>> = Default::default();
    let mut rng = StdRng::seed_from_u64(0);
    model.reset_params(&mut rng);
}
