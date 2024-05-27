#version 460

#include "include/buffers.glsl"
#include "include/functions.glsl"

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

float dA_dx(float x, uint id) {
    switch (id) {
    case SIGMOID:
        return dsigmoid_dx(x);
    default:
        return 0;
    }
}

void main() {
    uint layer = push_constants.layer;
    uint c = gl_GlobalInvocationID.x;
    uint pass = gl_GlobalInvocationID.y;

    layer_t layer_info = network.layers[layer];
    uint pass_count = imageSize(activations).z;

    if (c < layer_info.size && pass < pass_count) {
        uint layer_count = imageSize(z_values).y;
        uint delta_z_offset = pass * layer_count;

        float dC_da;
        if (int(layer) == imageSize(layer_data).z - 1) {
            float a = imageLoad(activations, ivec3(int(c), int(layer) + 1, int(pass))).x;
            float y = imageLoad(activations, ivec3(int(c), int(layer) + 2, int(pass))).x;

            dC_da = dC_dx(a, y);
        } else {
            dC_da = 0;

            uint next_layer = layer + 1;
            layer_t next_layer_info = network.layers[next_layer];
            
            for (uint n = 0; n < next_layer_info.size; n++) {
                float weight = imageLoad(layer_data, ivec3(int(n), int(c) + 1, int(next_layer))).x;
                float dC_db_n = imageLoad(deltas, ivec3(int(n), 0, int(next_layer + delta_z_offset))).x;
                float dC_dz_n = dC_db_n / 1; // dz/db

                dC_da += weight * dC_dz_n;
            }
        }

        float z = imageLoad(z_values, ivec3(int(c), int(layer), int(pass))).x;
        float dC_dz = dC_da * dA_dx(z, layer_info.activation_function);
        float dC_db = dC_dz * 1; // dz/db

        imageStore(deltas, ivec3(0, int(c), int(layer + delta_z_offset)), vec4(dC_db, 0, 0, 0));
        for (uint p = 0; p < layer_info.previous_size; p++) {
            // note no increment of 1 on the y coordinate - we need the *previous* layer
            float a_p = imageLoad(activations, ivec3(int(p), int(layer), int(pass))).x;
            float dC_dw = dC_dz * a_p;

            imageStore(deltas, ivec3(int(p) + 1, int(c), int(layer + delta_z_offset)), vec4(dC_dw, 0, 0, 0));
        }
    }
}