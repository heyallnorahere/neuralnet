#version 460
// basically a carbon copy of cpu_evaluator::eval

#include "include/buffers.glsl"
#include "include/functions.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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
    uint pass = gl_GlobalInvocationID.y;

    layer_t layer_info = network.layers[layer];
    uint pass_count = imageSize(activations).z;

    if (c < layer_info.size && pass < pass_count) {
        // see network.h
        // weights are laid out row-to-row
        // a row would represent the current layer

        float z = imageLoad(layer_data, ivec3(0, int(c), int(layer))).x;
        for (uint p = 0; p < layer_info.previous_size; p++) {
            float weight = imageLoad(layer_data, ivec3(int(p) + 1, int(c), int(layer))).x;
            float a_p = imageLoad(activations, ivec3(int(p), int(activation_layer_index), int(pass))).x;

            z += weight * a_p;
        }

        float a = A(z, layer_info.activation_function);
        imageStore(z_values, ivec3(int(c), int(layer), int(pass)), vec4(z, 0, 0, 0));
        imageStore(activations, ivec3(int(c), int(activation_layer_index), int(pass)), vec4(a, 0, 0, 0));
    }
}