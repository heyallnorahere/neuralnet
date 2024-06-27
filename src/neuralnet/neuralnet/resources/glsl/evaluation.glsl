#version 460
// basically a carbon copy of cpu_evaluator::eval

#include "include/buffers.glsl"
#include "include/functions.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// activation function
float A(float x, uint id) {
    switch (id) {
    case SIGMOID:
        return sigmoid(x);
    default:
        return 0;
    }
}

void main() {
    // layer index not including the input layer
    // this refers to the index into GENERATED data
    uint layer = push_constants.layer; // from include/buffers.glsl

    // whereas this is the layer index including the input layer
    // the input activations are included in the activation image
    uint activation_layer_index = layer + 1;

    // current neuron on the current layer (x coordinate within the dispatch)
    uint c = gl_GlobalInvocationID.x;

    // index of the working data point within the evaluation dispatch (y coordinate)
    uint pass = gl_GlobalInvocationID.y;

    // layer info (size, size of the previous layer, activation function, etc.)
    layer_t layer_info = network.layers[layer]; // from include/buffers.glsl

    // batch size, basically
    // z coordinate corresponds to batch data point
    uint pass_count = imageSize(activations).z;

    // check if this cell should actually perform any calculations
    if (c < layer_info.size && pass < pass_count) {
        // see network.h
        // weights are laid out row-to-row
        // a row would represent the current layer

        // we're computing the c'th element (row) of the z vector
        // therefore we need to take the c'th row of the weights matrix

        // z is initially set to the bias (constant term)
        // bias comes before weights in each data image row
        // each texel only has one value, hence we take the x component
        float z = imageLoad(layer_data, ivec3(0, int(c), int(layer))).x;

        // matrix multiplication
        // nxm matrix * mxl matrix = nxl matrix
        // in this case, cxp * px1 = cx1
        // we know c, so we iterate over p
        for (uint p = 0; p < layer_info.previous_size; p++) {
            // see previous explanation on data image
            float weight = imageLoad(layer_data, ivec3(int(p) + 1, int(c), int(layer))).x;

            // each row of the activation image corresponds to the layer
            // each column corresponds to a neuron index
            // and each z-layer corresponds to a batch data point
            float a_p = imageLoad(activations, ivec3(int(p), int(activation_layer_index) - 1, int(pass))).x;

            // dot product
            z += weight * a_p;
        }

        // run z value through activation function
        float a = A(z, layer_info.activation_function);

        // we use two different layer indices due to input layer properties (see above)
        imageStore(z_values, ivec3(int(c), int(layer), int(pass)), vec4(z, 0, 0, 0));
        imageStore(activations, ivec3(int(c), int(activation_layer_index), int(pass)), vec4(a, 0, 0, 0));
    }
}