// see cpu_evaluator.cpp
// lifted straight from there lol

double sigmoid(double x) {
    return 1 / (1 + exp(-float(x)));
}

double dsigmoid_dx(double x) {
    double sig = sigmoid(float(x));
    return sig * (1 - sig);
}