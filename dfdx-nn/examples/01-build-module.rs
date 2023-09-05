use dfdx_nn::dfdx::prelude::*;

fn main() {
    // The first part of building a network is to specify the architecture.
    // This specifies the shape of the network, but does not allocate on a device
    // Here, we are building a linear layer with 2 inputs and 5 outputs.
    // NOTE: LinearConstConfig lets use use compile time sizes.
    let arch: dfdx_nn::LinearConstConfig<2, 5> = Default::default();

    // The second part is to use the BuildModuleExt trait on device to actually allocate the module.
    // NOTE: the type of module is `dfdx_nn::Linear<Const<2>, Const<5>, f32, AutoDevice>`
    use dfdx_nn::BuildModuleExt;
    let dev = AutoDevice::default();
    let _module = dev.build_module::<f32>(arch);

    // There are also methods to mix and match compile time and runtime sizes.
    // Here is the same architecture, but using a runtime output size, instead of compile time.
    let arch = dfdx_nn::LinearConfig::new(Const::<2>, 5);

    // We build it the same way, but now the type is `dfdx_nn::Linear<Const<2>, usize, f32, AutoDevice>`
    let _module = dev.build_module::<f32>(arch);
}
