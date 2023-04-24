use super::*;
use crate::shapes::*;

thread_local! {
    static GLOBAL_CPU: Cpu = Cpu::default();
}

pub fn thread_cpu() -> Cpu {
    GLOBAL_CPU.with(|cpu| cpu.clone())
}

pub fn tensor<S: ConstShape, E: Dtype, Array>(array: Array) -> Tensor<S, E, Cpu>
where
    Cpu: TensorFrom<Array, S, E>,
{
    thread_cpu().tensor(array)
}

pub fn zeros<S: ConstShape, E: Dtype>() -> Tensor<S, E, Cpu> {
    thread_cpu().zeros()
}

pub fn ones<S: ConstShape, E: Dtype>() -> Tensor<S, E, Cpu> {
    thread_cpu().ones()
}

pub fn sample_uniform<S: ConstShape, E: Dtype>() -> Tensor<S, E, Cpu>
where
    Cpu: SampleTensor<E>,
    rand_distr::Standard: rand_distr::Distribution<E>,
{
    thread_cpu().sample_uniform()
}

pub fn sample_normal<S: ConstShape, E: Dtype>() -> Tensor<S, E, Cpu>
where
    Cpu: SampleTensor<E>,
    rand_distr::StandardNormal: rand_distr::Distribution<E>,
{
    thread_cpu().sample_normal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let q = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let r = sample_normal::<Rank2<4, 5>, f32>();
    }
}
