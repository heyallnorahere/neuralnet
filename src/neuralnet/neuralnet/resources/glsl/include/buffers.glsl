#define MAX_LAYERS 32
#define MAX_NEURONS_PER_LAYER 1024

#define SIGMOID 0

layout(set = 0, binding = 0, std430) buffer activations_t {
    double activations[MAX_NEURONS_PER_LAYER];
    double z[MAX_NEURONS_PER_LAYER];
} activations[MAX_LAYERS + 1];

struct layer_data_t {
    double biases[MAX_NEURONS_PER_LAYER];
    double weights[MAX_NEURONS_PER_LAYER * MAX_NEURONS_PER_LAYER];
};

layout(set = 0, binding = 1, std430) buffer deltas_t {
    layer_data_t deltas[MAX_LAYERS];
} deltas;

layout(set = 1, binding = 0, std430) buffer layer_t {
    uint size, previous_size, activation_function;
    layer_data_t data;
} layers[MAX_LAYERS];

layout(push_constant) uniform push_constants_t {
    uint layer;
} push_constants;