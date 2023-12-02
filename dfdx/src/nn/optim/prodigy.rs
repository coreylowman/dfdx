use std::marker::PhantomData;

use crate::{
    shapes::{Dtype, Shape},
    tensor::{Error, Gradients, Storage, Tensor, Tensorlike, UniqueId},
    tensor_ops::{Device, ProdigyConfig},
};

/// An implementation of the Prodigy optimizer from
/// [Prodigy: An Expeditiously Adaptive Parameter-Free Learner](https://arxiv.org/abs/2306.06101),
/// specifically _Algorithm 4, Adam version_, based on the researchers' [implementation](https://github.com/konstmish/prodigy).
///
/// # Example Usage
/// ```rust
/// # use dfdx::prelude::*;
/// # type Model = Tensor<Rank0, f32, Cpu>;
/// # let dev: Cpu = Default::default();
/// # let model: Model = dev.zeros();
/// let mut opt: Prodigy<Model, f32, Cpu> = optim::Prodigy::new(&model, ProdigyConfig {
///     lr: 1.0,
///     betas: [0.5, 0.25],
///     eps: 1e-6,
///     weight_decay: Some(WeightDecay::Decoupled(1e-2)),
///     ..Default::default()
/// });
/// ```
///
/// See module level documentation at [crate::nn::optim] for examples of how to actually use an optimizer.
#[derive(Debug, Clone)]
pub struct Prodigy<M, E: Dtype, D: Storage<E>> {
    /// Hyperparameter configuration
    pub cfg: ProdigyConfig,

    /// Timestep.
    k: i32,

    // d-values that change across step updates
    d: f64,
    d_max: f64,
    d_numerator: f64,

    s: Gradients<E, D>,

    /// Initial value for a given parameter.
    ///
    /// For a given parameter, the initial value is observed at it's first optimizer update.
    p0: Gradients<E, D>,

    /// Helper data to identify whether `p0` has been initialized for a given parameter.
    ///
    /// - `E::zero()` indicates the `p0` value for the parameter hasn't been initialized.
    /// - `E::one()` indicates the `p0` value for the parameter has been initialized.
    //
    // Note: this is currently expensive since a single bool per parameter would be sufficient.
    p0b: Gradients<E, D>,

    moment1: Gradients<E, D>,
    moment2: Gradients<E, D>,

    marker: PhantomData<*const M>,
}

impl<M, E: Dtype + num_traits::Zero, D: Storage<E>> Prodigy<M, E, D> {
    /// Constructs using hyperparameters from `cfg`.
    pub fn new(_model: &M, cfg: ProdigyConfig) -> Self {
        Self {
            cfg,
            k: 0,
            d: cfg.d0,
            d_max: cfg.d0,
            d_numerator: 0.0,
            s: Gradients::leaky(),
            p0: Gradients::leaky(),
            p0b: Gradients::leaky(),
            moment1: Gradients::leaky(),
            moment2: Gradients::leaky(),
            marker: PhantomData,
        }
    }
}

impl<M, E: Dtype, D: Device<E>> crate::nn::Optimizer<M, E, D> for Prodigy<M, E, D> {
    fn update_tensor<S: Shape>(
        &mut self,
        t: &mut Tensor<S, E, D>,
        gradients: &Gradients<E, D>,
        missing_params: &mut Vec<UniqueId>,
    ) -> Result<(), crate::tensor::Error> {
        let g = gradients.get_ref_checked(t);
        match g {
            None => missing_params.push(t.id()),
            Some(g) => {
                let s_t = self.s.get_or_alloc_mut(t)?;
                let p0_t = self.p0.get_or_alloc_mut(t)?;
                let p0b_t = self.p0b.get_or_alloc_mut(t)?;
                let m_t = self.moment1.get_or_alloc_mut(t)?;
                let v_t = self.moment2.get_or_alloc_mut(t)?;
                self.cfg.try_update(
                    self.k,
                    &mut self.d,
                    &mut self.d_max,
                    &mut self.d_numerator,
                    t,
                    s_t,
                    p0_t,
                    p0b_t,
                    m_t,
                    v_t,
                    g,
                )?;
            }
        }
        Ok(())
    }

    fn update(&mut self, module: &mut M, gradients: &Gradients<E, D>) -> Result<(), Error>
    where
        M: crate::nn::UpdateParams<E, D>,
    {
        self.k = self.k.checked_add(1).unwrap();

        // NOTE: the rest of this is identical to default implementation of update.
        let mut missing_tensors = Vec::new();
        module.try_update_params(self, gradients, &mut missing_tensors)?;
        if missing_tensors.is_empty() {
            Ok(())
        } else {
            Err(Error::UnusedTensors(missing_tensors))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prelude::*, tests::*};

    type X = Tensor<Rank2<1, 2>, TestDtype, TestDevice>;
    type Y = [[TestDtype; 2]; 1];
    type M = MatMul<Const<2>, Const<2>, TestDtype, TestDevice>;
    fn init() -> (TestDevice, X, Y, M) {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([[0.1, 0.2]]);
        let y: [[TestDtype; 2]; 1] = [[7e2, 8e2]];
        let w: Tensor<_, TestDtype, _> = dev.tensor([[3., 4.], [5., 6.]]);
        let mut m = dev.build_module::<TestDtype>(MatMulConstConfig::<2, 2>::default());
        m.weight = w;
        (dev, x, y, m)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_against(
        dev: &TestDevice,
        x: X,
        y: Y,
        mut m: M,
        mut opt: Prodigy<M, TestDtype, TestDevice>,
        expected_prediction: [[[f64; 2]; 1]; 10],
        expected_grads: [[[f64; 2]; 2]; 10],
        expected_updates: [[[f64; 2]; 2]; 10],
    ) {
        let mut grads = m.alloc_grads();
        for ((ey, eg), eu) in expected_prediction
            .iter()
            .zip(expected_grads)
            .zip(expected_updates)
        {
            let prediction = m.forward_mut(x.trace(grads));

            #[cfg(feature = "test-f64")]
            assert_close_to_literal!(prediction, ey, 7e-5);
            #[cfg(not(feature = "test-f64"))]
            assert_close_to_literal!(prediction, ey);

            let loss = crate::losses::mse_loss(prediction, dev.tensor(y));
            grads = loss.backward();

            #[cfg(feature = "test-f64")]
            assert_close_to_literal!(grads.get(&m.weight), eg, 3e-5);
            #[cfg(not(feature = "test-f64"))]
            assert_close_to_literal!(grads.get(&m.weight), eg);

            opt.update(&mut m, &grads).expect("");

            #[cfg(feature = "test-f64")]
            assert_close_to_literal!(m.weight, eu, 5e-4);
            #[cfg(not(feature = "test-f64"))]
            assert_close_to_literal!(m.weight, eu);

            m.zero_grads(&mut grads);
        }
    }

    #[test]
    fn test_default_prodigy_params() {
        let (dev, x, y, m) = init();
        let opt = Prodigy::new(&m, Default::default());
        #[rustfmt::skip]
        let expected_prediction: [[[f64; 2]; 1]; 10] = [
            [[1.100000023841858, 1.7000000476837158]], [[1.1000009775161743, 1.7000010013580322]],
            [[1.1000022888183594, 1.7000024318695068]], [[1.1000046730041504, 1.7000048160552979]],
            [[1.1000117063522339, 1.7000117301940918]], [[1.1000306606292725, 1.7000305652618408]],
            [[1.1000787019729614, 1.7000787258148193]], [[1.1002024412155151, 1.700202465057373]],
            [[1.1005204916000366, 1.7005205154418945]], [[1.101338505744934, 1.701338529586792]],
        ];

        #[rustfmt::skip]
        let expected_grads: [[[f64; 2]; 2]; 10] = [
            [[-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938]],
            [[-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938]],
            [[-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938]],
            [[-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938]],
            [[-69.88999938964844, -139.77999877929688], [-79.83000183105469, -159.66000366210938]],
            [[-69.88999938964844, -139.77999877929688], [-79.83000183105469, -159.66000366210938]],
            [[-69.8899917602539, -139.7799835205078], [-79.82999420166016, -159.6599884033203]],
            [[-69.88997650146484, -139.7799530029297], [-79.8299789428711, -159.6599578857422]],
            [[-69.88994598388672, -139.77989196777344], [-79.82994842529297, -159.65989685058594]],
            [[-69.8898696899414, -139.7797393798828], [-79.82986450195312, -159.65972900390625]],
        ];

        #[rustfmt::skip]
        let expected_updates: [[[f64; 2]; 2]; 10] = [
            [[3.0000030994415283, 4.000003337860107], [5.000003337860107, 6.000003337860107]],
            [[3.000007390975952, 4.000007629394531], [5.000007629394531, 6.000007629394531]],
            [[3.0000154972076416, 4.000015735626221], [5.000015735626221, 6.000015735626221]],
            [[3.0000391006469727, 4.000039100646973], [5.000039100646973, 6.000039100646973]],
            [[3.0001020431518555, 4.0001020431518555], [5.0001020431518555, 6.0001020431518555]],
            [[3.0002622604370117, 4.000262260437012], [5.000262260437012, 6.000262260437012]],
            [[3.0006744861602783, 4.000674724578857], [5.000674724578857, 6.000674724578857]],
            [[3.001734733581543, 4.001734733581543], [5.001734733581543, 6.001734733581543]],
            [[3.0044617652893066, 4.004461765289307], [5.004461765289307, 6.004461765289307]],
            [[3.011474609375, 4.011474609375], [5.011474609375, 6.011474609375]],
        ];

        #[rustfmt::skip]
        check_against(&dev, x, y, m, opt, expected_prediction, expected_grads, expected_updates);
    }

    #[test]
    fn test_custom_prodigy_params() {
        let (dev, x, y, m) = init();
        let opt = Prodigy::new(
            &m,
            ProdigyConfig {
                lr: 2e1,
                betas: [0.5, 0.25],
                beta3: Some(0.4),
                eps: 1e-8,
                weight_decay: None,
                use_bias_correction: true,
                safeguard_warmup: true,
                d0: 1e-5,
                d_coef: 0.5,
                growth_rate: 1.02,
            },
        );

        #[rustfmt::skip]
        let expected_prediction: [[[f64; 2]; 1]; 10] = [
            [[1.100000023841858, 1.7000000476837158]], [[1.100059986114502, 1.7000598907470703]],
            [[1.100119948387146, 1.700119972229004]], [[1.1073875427246094, 1.7073874473571777]],
            [[1.1166749000549316, 1.716675043106079]], [[1.1270885467529297, 1.727088451385498]],
            [[1.1381947994232178, 1.7381949424743652]], [[1.149769902229309, 1.749769926071167]],
            [[1.1617008447647095, 1.7617008686065674]], [[1.173932433128357, 1.7739324569702148]],
        ];

        #[rustfmt::skip]
        let expected_grads: [[[f64; 2]; 2]; 10] = [
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.88999938964844, -139.77999877929688], [-79.82999420166016, -159.6599884033203], ],
            [ [-69.8899917602539, -139.7799835205078], [-79.82998657226562, -159.65997314453125], ],
            [ [-69.88926696777344, -139.77853393554688], [-79.82926177978516, -159.6585235595703], ],
            [ [-69.8883285522461, -139.7766571044922], [-79.82833099365234, -159.6566619873047], ],
            [ [-69.88729095458984, -139.7745819091797], [-79.8272933959961, -159.6545867919922], ],
            [ [-69.88618469238281, -139.77236938476562], [-79.82617950439453, -159.65235900878906], ],
            [ [-69.88502502441406, -139.77005004882812], [-79.82502746582031, -159.65005493164062], ],
            [ [-69.88383483886719, -139.76766967773438], [-79.8238296508789, -159.6476593017578], ],
            [ [-69.88260650634766, -139.7652130126953], [-79.8226089477539, -159.6452178955078], ],
        ];

        #[rustfmt::skip]
        let expected_updates: [[[f64; 2]; 2]; 10] = [
            [ [3.000200033187866, 4.000199794769287], [5.000199794769287, 6.000199794769287], ],
            [ [3.0004000663757324, 4.000399589538574], [5.000399589538574, 6.000399589538574], ],
            [ [3.024625062942505, 4.024624824523926], [5.024624824523926, 6.024624824523926], ],
            [ [3.0555830001831055, 4.0555830001831055], [5.0555830001831055, 6.0555830001831055], ],
            [ [3.0902950763702393, 4.09029483795166], [5.09029483795166, 6.09029483795166], ],
            [ [3.1273159980773926, 4.127315998077393], [5.127315998077393, 6.127315998077393], ],
            [ [3.1658997535705566, 4.165899753570557], [5.165899753570557, 6.165899753570557], ],
            [ [3.205669403076172, 4.205669403076172], [5.205669403076172, 6.205669403076172], ],
            [ [3.246441602706909, 4.24644136428833], [5.24644136428833, 6.24644136428833], ],
            [ [3.2881321907043457, 4.288132190704346], [5.288132190704346, 6.288132190704346], ],
        ];

        #[rustfmt::skip]
        check_against(&dev, x, y, m, opt, expected_prediction, expected_grads, expected_updates);
    }

    #[test]
    fn test_prodigy_l2_decay() {
        let (dev, x, y, m) = init();
        let opt = Prodigy::new(
            &m,
            ProdigyConfig {
                betas: [0.5, 0.25],
                beta3: Some(0.4),
                weight_decay: Some(WeightDecay::L2(1.0)),
                ..Default::default()
            },
        );

        #[rustfmt::skip]
        let expected_prediction: [[[f64; 2]; 1]; 10] = [
            [[1.100000023841858, 1.7000000476837158]], [[1.1000001430511475, 1.700000286102295]],
            [[1.1000003814697266, 1.700000524520874]], [[1.1000007390975952, 1.7000007629394531]],
            [[1.1000009775161743, 1.7000010013580322]], [[1.1000014543533325, 1.7000014781951904]],
            [[1.1000021696090698, 1.7000021934509277]], [[1.1000032424926758, 1.7000033855438232]],
            [[1.10000479221344, 1.7000048160552979]], [[1.1000072956085205, 1.700007438659668]],
        ];

        #[rustfmt::skip]
        let expected_grads: [[[f64; 2]; 2]; 10] = [
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.88999938964844, -139.77999877929688], [-79.83000183105469, -159.66000366210938], ],
        ];
        #[rustfmt::skip]
        let expected_updates: [[[f64; 2]; 2]; 10] = [
            [ [3.000000476837158, 4.000000476837158], [5.000000476837158, 6.000000476837158], ],
            [ [3.0000011920928955, 4.000001430511475], [5.000001430511475, 6.000001430511475], ],
            [ [3.000002145767212, 4.000002384185791], [5.000002384185791, 6.000002384185791], ],
            [ [3.0000030994415283, 4.000003337860107], [5.000003337860107, 6.000003337860107], ],
            [ [3.000004529953003, 4.000004768371582], [5.000004768371582, 6.000004768371582], ],
            [ [3.000006914138794, 4.000007152557373], [5.000007152557373, 6.000007152557373], ],
            [ [3.0000104904174805, 4.000010967254639], [5.000010967254639, 6.000010967254639], ],
            [ [3.0000159740448, 4.000016212463379], [5.000016212463379, 6.000016212463379], ],
            [ [3.0000243186950684, 4.000024318695068], [5.000024318695068, 6.000024318695068], ],
            [ [3.0000367164611816, 4.000036716461182], [5.000036716461182, 6.000036716461182], ],
        ];
        #[rustfmt::skip]
        check_against(&dev, x, y, m, opt, expected_prediction, expected_grads, expected_updates);
    }

    #[test]
    fn test_prodigy_decoupled_decay() {
        let (dev, x, y, m) = init();
        let opt = Prodigy::new(
            &m,
            ProdigyConfig {
                betas: [0.5, 0.25],
                beta3: Some(0.4),
                weight_decay: Some(WeightDecay::Decoupled(1e3)),
                ..Default::default()
            },
        );

        #[rustfmt::skip]
        let expected_prediction: [[[f64; 2]; 1]; 10] = [
            [[1.100000023841858, 1.7000000476837158]], [[1.0989001989364624, 1.6983001232147217]],
            [[1.0978014469146729, 1.69660222530365]], [[1.0967040061950684, 1.6949058771133423]],
            [[1.0956075191497803, 1.693211317062378]], [[1.0945122241973877, 1.6915183067321777]],
            [[1.093418002128601, 1.6898270845413208]], [[1.0923248529434204, 1.6881375312805176]],
            [[1.0912327766418457, 1.686449646949768]], [[1.0901418924331665, 1.6847634315490723]],
        ];

        #[rustfmt::skip]
        let expected_grads: [[[f64; 2]; 2]; 10] = [
            [ [-69.89000701904297, -139.78001403808594], [-79.83000183105469, -159.66000366210938], ],
            [ [-69.8901138305664, -139.7802276611328], [-79.83016967773438, -159.66033935546875], ],
            [ [-69.89022064208984, -139.7804412841797], [-79.8303451538086, -159.6606903076172], ],
            [ [-69.89033508300781, -139.78067016601562], [-79.83051300048828, -159.66102600097656], ],
            [ [-69.89044189453125, -139.7808837890625], [-79.83068084716797, -159.66136169433594], ],
            [ [-69.89055633544922, -139.78111267089844], [-79.83084869384766, -159.6616973876953], ],
            [ [-69.89065551757812, -139.78131103515625], [-79.83101654052734, -159.6620330810547], ],
            [ [-69.8907699584961, -139.7815399169922], [-79.83119201660156, -159.66238403320312], ],
            [ [-69.89087677001953, -139.78175354003906], [-79.83135223388672, -159.66270446777344], ],
            [ [-69.89098358154297, -139.78196716308594], [-79.83152770996094, -159.66305541992188], ],
        ];
        #[rustfmt::skip]
        let expected_updates: [[[f64; 2]; 2]; 10] = [
            [ [2.9970004558563232, 3.9960005283355713], [4.99500036239624, 5.994000434875488], ],
            [ [2.994004249572754, 3.9920053482055664], [4.990006446838379, 5.988007545471191], ],
            [ [2.991011142730713, 3.9880142211914062], [4.9850172996521, 5.982020378112793], ],
            [ [2.9880211353302, 3.984027147293091], [4.9800333976745605, 5.976039409637451], ],
            [ [2.9850339889526367, 3.98004412651062], [4.9750542640686035, 5.970064163208008], ],
            [ [2.9820499420166016, 3.976064920425415], [4.970080375671387, 5.964095115661621], ],
            [ [2.9790687561035156, 3.9720897674560547], [4.965111255645752, 5.958131790161133], ],
            [ [2.976090669631958, 3.968118667602539], [4.960146903991699, 5.952174663543701], ],
            [ [2.9731154441833496, 3.964151620864868], [4.955187797546387, 5.946223258972168], ],
            [ [2.9701433181762695, 3.960188388824463], [4.950233459472656, 5.940278053283691], ],
        ];
        #[rustfmt::skip]
        check_against(&dev, x, y, m, opt, expected_prediction, expected_grads, expected_updates);
    }

    #[test]
    fn test_unused_tensors() {
        let dev: TestDevice = Default::default();
        let mut t: Tensor<Rank1<5>, TestDtype, _> = dev.sample_normal();
        let mut opt = Prodigy::new(&t, Default::default());
        opt.update(&mut t, &Gradients::leaky()).expect_err("");
    }
}
