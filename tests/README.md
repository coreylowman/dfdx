# Integration tests

This directory contains integration tests for specific models.

## resnet18

This set of tests exports a pretrained resnet18 from pytorch,
and a set of random inputs & the expected outputs. Then
dfdx will load the pretrained weights & inputs and compare
against expected outputs.

1. `cd tests`
2. `python save_resnet18.py`
3. `cargo +nightly test -F test-integrations,numpy resnet18`

## mobilenet_v3_small

This set of tests exports a pretrained MobileNetV3Small from pytorch,
and a set of random inputs & the expected outputs. Then
dfdx will load the pretrained weights & inputs and compare
against expected outputs.

1. `cd tests`
2. `python save_mobilenet_v3_small.py`
3. `cargo +nightly test -F test-integrations,numpy mobilenet_v3_small`
