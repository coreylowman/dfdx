use dfdx::shapes::Const;
use dfdx_nn::*;

#[derive(Default, Clone, Sequential)]
#[built(MixedMlp)]
pub struct MixedMlpConfig {
    pub l1: LinearConfig<Const<3>, usize>,
    pub act1: ReLU,
    pub l2: LinearConfig<usize, Const<10>>,
    pub act2: ReLU,
}

fn main() {
    use dfdx::prelude::*;

    let dev: Cpu = Default::default();

    let structure = MixedMlpConfig {
        l1: dfdx_nn::LinearConfig::new(Const, 5),
        act1: Default::default(),
        l2: dfdx_nn::LinearConfig::new(5, Const),
        act2: Default::default(),
    };
    let module: MixedMlp<f32, Cpu> = dev.build_module_ext::<f32>(structure);
    let x: Tensor<(Const<10>, Const<3>), f32, _> = dev.sample_normal();
    let _: Tensor<(Const<10>, Const<10>), f32, _> = module.forward(x);
}
