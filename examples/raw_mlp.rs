use ndarray_rand::rand::prelude::*;
use stag::gradients::GradientTape;
use stag::nn::Linear;
use stag::prelude::*;

#[derive(Default, Debug)]
struct MLP {
    l1: Linear<16, 8>,
    l2: Linear<8, 4>,
    l3: Linear<4, 2>,
}

impl Init for MLP {
    fn init<R: Rng>(&mut self, rng: &mut R) {
        self.l1.init(rng);
        self.l2.init(rng);
        self.l3.init(rng);
    }
}

impl Taped for MLP {
    fn update(&mut self, tape: &GradientTape) {
        self.l1.update(tape);
        self.l2.update(tape);
        self.l3.update(tape);
    }
}

impl Module<Tensor1D<16>, Tensor1D<2>> for MLP {
    fn forward(&mut self, x: &mut Tensor1D<16>) -> Tensor1D<2> {
        let mut x = self.l1.forward(x);
        let mut x = x.relu();
        let mut x = self.l2.forward(&mut x);
        let mut x = x.relu();
        self.l3.forward(&mut x).tanh()
    }
}

impl<const B: usize> Module<Tensor2D<B, 16>, Tensor2D<B, 2>> for MLP {
    fn forward(&mut self, x: &mut Tensor2D<B, 16>) -> Tensor2D<B, 2> {
        let mut x = self.l1.forward(x);
        let mut x = x.relu();
        let mut x = self.l2.forward(&mut x);
        let mut x = x.relu();
        self.l3.forward(&mut x).tanh()
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut mlp = MLP::default();
    mlp.init(&mut rng);

    let mut x: Tensor1D<16> = Tensor1D::rand(&mut rng);
    println!("{:#}", mlp.forward(&mut x).data());
    // [0.98638135, -0.842192]

    let mut x: Tensor2D<2, 16> = Tensor2D::rand(&mut rng);
    println!("{:#}", mlp.forward(&mut x).data());
    // [[0.97975945, -0.7831348],
    //  [0.99406636, -0.92086846]]

    let mut x: Tensor2D<4, 16> = Tensor2D::rand(&mut rng);
    println!("{:#}", mlp.forward(&mut x).data());
    // [[0.95267, -0.58816254],
    //  [0.986069, -0.8392567],
    //  [0.9966356, -0.9511877],
    //  [0.95722795, -0.61708426]]
}
