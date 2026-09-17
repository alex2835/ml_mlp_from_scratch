#pragma once
#include "layer.hpp"

struct DenseLayer : Layer
{
    Mat W, dW; // [m x n]
    Vec b, db; // [m]
    Mat X, A;  // cached input and output, [n x B] and [m x B]
    Activation activation;

    DenseLayer(int n, int m, Activation act)
        : W(Mat::Random(m, n) * std::sqrt(2.0f / n)),
          dW(Mat::Zero(m, n)),
          b(Vec::Zero(m)),
          db(Vec::Zero(m)),
          activation(act)
    {}

    Mat forward(const Mat &x) override
    {
        X = x;
        Mat Z = (W * X).colwise() + b;
        A = get_activation(activation)(Z); // [m x B]
        return A;
    }

    Mat backward(const Mat &dY) override
    {
        Mat dZ = dY.cwiseProduct(get_derivative(activation)(A)); // chain rule through the activation
        dW = dZ * X.transpose();   // [m x B][B x n] -> [m x n]
        db = dZ.rowwise().sum();   // [m]
        return W.transpose() * dZ; // [n x m][m x B] -> [n x B]
    }

    // l2 is the weight-decay strength: every step shrinks each weight
    // toward zero in proportion to its own size (gradient of l2/2 * sum(W^2))
    void step(float lr, float l2) override
    {
        W -= lr * (dW + l2 * W);
        b -= lr * db;
    }

    const Mat &weights() const override { return W; }
};
