#include "AGN.h"
#include <torch/csrc/autograd/python_variable.h>

void print_tensor(torch::Tensor tensor)
{
    std::cout << "Tensor: " << std::endl;
    for (int i = 0; i < tensor.size(0); i++) {
        for (int j = 0; j < tensor.size(1); j++) {
            for (int k = 0; k < tensor.size(2); k++) {
                for (int l = 0; l < tensor.size(3); l++) {
                    std::cout << tensor[i][j][k][l].item<float>() << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

torch::Tensor arrange_tensor_channels(torch::Tensor &input, torch::Tensor &indices)
{

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto shuffled_input = torch::empty({ N, C, H, W }, input.options());

    auto input_data_ptr = input.data_ptr<float>();

    auto indices_data_ptr = indices.data_ptr<int64_t>();

    auto shuffled_input_ptr = shuffled_input.data_ptr<float>();

    int64_t new_index;
    int64_t HW = H * W;
    int64_t channel_size = HW * sizeof(float);

    for (int64_t i = 0; i < N * C; i++)
    {
        new_index = indices_data_ptr[i];
        memcpy(shuffled_input_ptr + (i * HW), input_data_ptr + (new_index * HW), channel_size);
    }

    return shuffled_input;
}
#define COMPILE_TO_PYTHON
#ifdef COMPILE_TO_PYTHON
static PyObject* arrange_tensor_channels_wrap(PyObject* self, PyObject* args)
{
    PyObject* input_obj, * indices_obj;
    if (!PyArg_ParseTuple(args, "OO", &input_obj, &indices_obj)) {
        return NULL;
    }

    // Convert the PyTorch tensors to C++ tensors
    auto input = THPVariable_Unpack(input_obj);
    auto indices = THPVariable_Unpack(indices_obj);

    // Call the C++ function
    auto output = arrange_tensor_channels(input, indices);

    // Convert the C++ tensor back to a PyTorch tensor and return it
    return THPVariable_Wrap(output);
}

PyMODINIT_FUNC PyInit_AGN()
{
    static PyMethodDef ArrangeTensorMethods[] = {
        {"arrange_tensor_channels", (PyCFunction)arrange_tensor_channels_wrap, METH_VARARGS, "BBBHBH" },
        { NULL, NULL, 0, NULL }
    };

    static PyModuleDef ArrangeTensorModule = {
        PyModuleDef_HEAD_INIT,
        "AGN",
        "AGN module",
        -1,
        ArrangeTensorMethods
    };

    PyObject* module = PyModule_Create(&ArrangeTensorModule);

    return module;
}
#endif