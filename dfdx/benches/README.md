# Benchmarks

This folder contains a couple of simple utility scripts for running
forward and backward passes of core ops.

- `cargo bench --bench batchnorm2d`
- `cargo bench --bench sum`
- `cargo +nightly bench --bench conv2d`

Additionally you can pass `-F cuda` to use a Cuda.
