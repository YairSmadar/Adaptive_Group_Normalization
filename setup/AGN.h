#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <Python.h>

torch::Tensor arrange_tensor_channels(torch::Tensor &input, torch::Tensor &indices);
