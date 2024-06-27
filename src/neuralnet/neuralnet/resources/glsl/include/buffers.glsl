#define MAX_LAYERS 32

#define SIGMOID 0

// activation value matrix
layout(set = 0, binding = 0, r32f) uniform image3D activations;

// z value matrix
layout(set = 0, binding = 1, r32f) uniform image3D z_values;

// deltas (laid out the same as layer_data, pass data stacked on the z axis)
layout(set = 0, binding = 2, r32f) uniform image3D deltas;

struct layer_t {
    uint size, previous_size;
    uint activation_function;
};

// sizes and metadata for each layer
layout(set = 1, binding = 0, std430) buffer network_t {
    layer_t layers[MAX_LAYERS];
} network;

// rows correspond to a neuron on the current layer
// laid out such that biases come before weights in rows
// each z-layer corresponds to a network layer
layout(set = 1, binding = 1, r32f) uniform image3D layer_data;

// specified per dispatch
layout(push_constant) uniform push_constants_t {
    uint layer;
    float delta_scalar;
} push_constants;