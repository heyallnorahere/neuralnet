#define MAX_LAYERS 32
#define MAX_NEURONS_PER_LAYER 1024

#define SIGMOID 0

layout(set = 0, binding = 0, r32f) uniform image3D activations;
layout(set = 0, binding = 1, r32f) uniform image3D z_values;
layout(set = 0, binding = 2, r32f) uniform image3D deltas;

struct layer_t {
    uint size, previous_size, activation_function;
};

layout(set = 1, binding = 0, std430) buffer network_t {
    layer_t layers[MAX_LAYERS];
} network;

layout(set = 1, binding = 1, r32f) uniform image3D layer_data;

layout(push_constant) uniform push_constants_t {
    uint layer;
} push_constants;