// see cpu_evaluator.cpp
// lifted straight from there lol

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float dsigmoid_dx(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

float dC_dx(float x, float y) {
    //   d/dx((x - y)^2)
    // = d/dx(x - y) * d((x - y)^2)/d(x - y)
    // = 1 * 2(x - y)
    // = 2(x - y)

    return 2 * (x - y);
}