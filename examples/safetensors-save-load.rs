//! Demonstrates how to save and load arrays with safetensors

#[cfg(feature = "safetensors")]
fn main() {
    use ::safetensors::SafeTensors;
    use dfdx::{
        prelude::*,
        tensor::{AsArray, AutoDevice, Cpu},
    };
    use memmap2::MmapOptions;
    let dev: Cpu = Default::default();

    type Model = (Linear<5, 10>, Linear<10, 5>);
    let model = dev.build_module::<Model, f32>();
    model
        .save_safetensors("model.safetensors")
        .expect("Failed to save model");

    let mut model2 = dev.build_module::<Model, f32>();
    model2
        .load_safetensors("model.safetensors")
        .expect("Failed to load model");

    assert_eq!(model.0.weight.array(), model2.0.weight.array());

    //  ADVANCED USAGE to load pre-existing models

    // wget -O gpt2.safetensors https://huggingface.co/gpt2/resolve/main/model.safetensors

    let mut gpt2 = dev.build_module::<Linear<768, 50257>, f32>();
    let filename = "gpt2.safetensors";
    let f = std::fs::File::open(filename).expect("Couldn't read file, have you downloaded gpt2 ? `wget -O gpt2.safetensors https://huggingface.co/gpt2/resolve/main/model.safetensors`");
    let buffer = unsafe { MmapOptions::new().map(&f).expect("Could not mmap") };
    let tensors = SafeTensors::deserialize(&buffer).expect("Couldn't read safetensors file");

    gpt2.weight
        .load_safetensor(&tensors, "wte.weight")
        .expect("Could not load tensor");
}

#[cfg(not(feature = "safetensors"))]
fn main() {
    panic!("Use the 'safetensors' feature to run this example");
}
