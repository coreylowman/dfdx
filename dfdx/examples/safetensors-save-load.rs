//! Demonstrates how to save and load arrays with safetensors

#[cfg(feature = "safetensors")]
fn main() {
    use dfdx::prelude::*;

    let dev: Cpu = Default::default();

    type Model = (LinearConstConfig<5, 10>, LinearConstConfig<10, 5>);

    let model = dev.build_module::<f32>(Model::default());
    model
        .save_safetensors("model.safetensors")
        .expect("Failed to save model");

    let mut model2 = dev.build_module::<f32>(Model::default());
    model2
        .load_safetensors("model.safetensors")
        .expect("Failed to load model");

    assert_eq!(model.0.weight.array(), model2.0.weight.array());
}

#[cfg(not(feature = "safetensors"))]
fn main() {
    panic!("Run this example with `-F safetensors`")
}
