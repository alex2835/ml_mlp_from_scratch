#pragma once
#include "layer.hpp"

// Reference convolution written as plain loops. Slow, but every index is
// visible — read this one to understand, use ConvIm2col to train.
// Stride is always 1; zero padding of `pad` pixels on each side.
//
// Column layout (see layer.hpp): input index (y*W + x)*C_in + c.
// Kernels live in one matrix K [C_out x C_in*k*k]: row co is kernel co,
// column ci*k*k + a*k + b is its weight for input channel ci at offset (a, b).
struct ConvNaive : Layer
{
    int C_in, H, W, C_out, k, pad, Ho, Wo;
    Mat K, dK;
    Vec b, db;
    Mat X, A;
    Activation activation;

    ConvNaive(int C_in, int H, int W, int C_out, int k, int pad, Activation act)
        : C_in(C_in), H(H), W(W), C_out(C_out), k(k), pad(pad),
          Ho(H + 2 * pad - k + 1), Wo(W + 2 * pad - k + 1),
          K(Mat::Random(C_out, C_in * k * k) * std::sqrt(2.0f / (C_in * k * k))),
          dK(Mat::Zero(C_out, C_in * k * k)),
          b(Vec::Zero(C_out)),
          db(Vec::Zero(C_out)),
          activation(act)
    {}

    int in_index(int y, int x, int c) const  { return (y * W + x) * C_in + c; }
    int out_index(int y, int x, int c) const { return (y * Wo + x) * C_out + c; }
    int k_index(int ci, int a, int bb) const { return ci * k * k + a * k + bb; }

    Mat forward(const Mat &x) override
    {
        X = x;
        const int B = int(x.cols());
        Mat Z(C_out * Ho * Wo, B);

        for (int n = 0; n < B; ++n)
        for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Wo; ++ox)
        for (int co = 0; co < C_out; ++co)
        {
            // one output value = one small dot product over the k x k x C_in window
            float s = b[co];
            for (int a = 0; a < k; ++a)
            for (int bb = 0; bb < k; ++bb)
            {
                int iy = oy + a - pad, ix = ox + bb - pad;
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue; // zero padding
                for (int ci = 0; ci < C_in; ++ci)
                    s += K(co, k_index(ci, a, bb)) * x(in_index(iy, ix, ci), n);
            }
            Z(out_index(oy, ox, co), n) = s;
        }

        A = get_activation(activation)(Z);
        return A;
    }

    Mat backward(const Mat &dY) override
    {
        Mat dZ = dY.cwiseProduct(get_derivative(activation)(A));
        const int B = int(X.cols());
        dK.setZero();
        db.setZero();
        Mat dX = Mat::Zero(X.rows(), B);

        // Same loops as forward. Every weight was used at every position of
        // every sample, so its gradient accumulates over all of them.
        for (int n = 0; n < B; ++n)
        for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Wo; ++ox)
        for (int co = 0; co < C_out; ++co)
        {
            float g = dZ(out_index(oy, ox, co), n);
            db[co] += g;
            for (int a = 0; a < k; ++a)
            for (int bb = 0; bb < k; ++bb)
            {
                int iy = oy + a - pad, ix = ox + bb - pad;
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                for (int ci = 0; ci < C_in; ++ci)
                {
                    int ki = k_index(ci, a, bb), ii = in_index(iy, ix, ci);
                    dK(co, ki) += g * X(ii, n); // weight blame: error x the input it multiplied
                    dX(ii, n) += g * K(co, ki); // input blame: error x the weight it passed through
                }
            }
        }
        return dX;
    }

    void step(float lr, float l2) override
    {
        K -= lr * (dK + l2 * K);
        b -= lr * db;
    }

    const Mat &weights() const override { return K; }
};
