#version 460
// basically a carbon copy of cpu_evaluator::eval

#include "include/buffers.glsl"
#include "include/functions.glsl"

layout(local_size_x = 32​, local_size_y = 0​, local_size_z = 0​) in;

double A(double x, uint id) {
    switch (id) {
    case SIGMOID:
        return sigmoid(x);
    default:
        return 0;
    }
}

void main() {
    uint layer = push_constants.layer;
    uint activation_layer_index = layer + 1;

    uint size = layers[layer].size;
    uint previous_size = layers[layer].previous_size;
    uint activation = layers[layer].activation_function;

    for (uint c = 0; c < size; c++) {
        // see network.h
        // weights are laid out row-to-row
        // a row would represent the current layer

        uint weight_offset = c * previous_size;
        double z = layers[layer].data.biases[c];

        for (uint p = 0; p < previous_size; p++) {
            uint weight_index = weight_offset + p;
            double weight = layers[layer].data.weights[weight_index];

            z += weight * activations[activation_layer_index - 1].activations[p];
        }

        activations[activation_layer_index].z[c] = z;
        activations[activation_layer_index].activations[c] = A(z, activation);
    }
}