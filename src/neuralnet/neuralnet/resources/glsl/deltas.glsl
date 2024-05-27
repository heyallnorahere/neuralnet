#version 460

#include "include/buffers.glsl"
#include "include/functions.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

void main() {
    uvec3 target_coords = gl_GlobalInvocationID;
    vec4 value = imageLoad(layer_data, ivec3(target_coords));

    uvec3 z_size = imageSize(z_values);
    uint layer_count = z_size.y;
    uint pass_count = z_size.z;

    for (uint i = 0; i < pass_count; i++) {
        uvec3 delta_coords = target_coords + uvec3(0, 0, i * layer_count);
        vec4 delta = imageLoad(deltas, ivec3(delta_coords));

        value -= delta * push_constants.delta_scalar;
    }

    imageStore(layer_data, ivec3(target_coords), value);
}