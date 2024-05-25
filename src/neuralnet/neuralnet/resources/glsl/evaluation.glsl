#version 460
// basically a carbon copy of cpu_evaluator::eval

#include "include/buffers.glsl"
#include "include/functions.glsl"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

float A(float x, uint id) {
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
    uint c = gl_GlobalInvocationID.x;

    layer_t layer_info = network.layers[layer];
    if (c < layer_info.size) {
        // see network.h
        // weights are laid out row-to-row
        // a row would represent the current layer

        float z = imageLoad(layer_data, ivec3(0, int(c), int(layer))).x;
        for (uint p = 0; p < layer_info.previous_size; p++) {
            float weight = imageLoad(layer_data, ivec3(int(p) + 1, int(c), int(layer))).x;
            float a_p = imageLoad(activations, ivec2(int(p), int(activation_layer_index))).x;

            z += weight * a_p;
        }

        float a = A(z, layer_info.activation_function);
        imageStore(z_values, ivec2(int(c), int(layer)), vec4(z, 0.f, 0.f, 0.f));
        imageStore(activations, ivec2(int(c), int(activation_layer_index)), vec4(a, 0.f, 0.f, 0.f));
    }
}