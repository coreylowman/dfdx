enum MomentumType {
    None,
    Classic,
    Nesterov,
};

enum WeightDecayType {
    WdNone,
    L2,
    Decoupled
};

struct SgdConfig {
    float lr;
    MomentumType momentum_type;
    float momentum;
    WeightDecayType weight_decay_type;
    float weight_decay;
};

extern "C" __global__ void sgd_update(
    const SgdConfig cfg,
    const size_t numel,
    float* param,
    float* velocity,
    const float* grad
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    float p = param[i];
    float g = grad[i];
    float v = velocity[i];

    if (cfg.weight_decay_type == L2) {
        g += cfg.weight_decay * p;
    }

    if (cfg.momentum_type == Classic) {
        v = g + cfg.momentum * v;
        g = v * cfg.lr;
    } else if (cfg.momentum_type == Nesterov) {
        v = g + cfg.momentum * v;
        g = (g + cfg.momentum * v) * cfg.lr;
    } else {
        g *= cfg.lr;
    }

    if (cfg.weight_decay_type == Decoupled) {
        g += cfg.weight_decay * cfg.lr * p;
    }

    velocity[i] = v;
    param[i] -= g;
}
