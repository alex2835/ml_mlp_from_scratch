#pragma once
#include "layer.hpp"

// Same convolution as ConvNaive, computed as ONE matrix product.
//
// im2col unrolls every k x k x C_in receptive field into a column of
// `cols` [C_in*k*k x B*Ho*Wo]. Then  Z = K * cols + b  is literally a
// DenseLayer forward with n = C_in*k*k inputs, m = C_out outputs and a
// "batch" of B*Ho*Wo positions — and backward is DenseLayer's backward
// followed by col2im, which scatters the column gradients back into the
// image, ADDING where windows overlapped.
struct ConvIm2col : Layer
{
    int C_in, H, W, C_out, k, pad, Ho, Wo;
    Mat K, dK;
    Vec b, db;
    Mat cols, A;
    Activation activation;

    ConvIm2col(int C_in, int H, int W, int C_out, int k, int pad, Activation act)
        : C_in(C_in), H(H), W(W), C_out(C_out), k(k), pad(pad),
          Ho(H + 2 * pad - k + 1), Wo(W + 2 * pad - k + 1),
          K(Mat::Random(C_out, C_in * k * k) * std::sqrt(2.0f / (C_in * k * k))),
          dK(Mat::Zero(C_out, C_in * k * k)),
          b(Vec::Zero(C_out)),
          db(Vec::Zero(C_out)),
          activation(act)
    {}

    int in_index(int y, int x, int c) const  { return (y * W + x) * C_in + c; }
    int k_index(int ci, int a, int bb) const { return ci * k * k + a * k + bb; }

    // Column n*Ho*Wo + oy*Wo + ox holds the window feeding output (oy, ox) of
    // sample n; entries where the window hangs over the border stay zero.
    Mat im2col(const Mat &x) const
    {
        const int B = int(x.cols()), P = Ho * Wo;
        Mat out = Mat::Zero(C_in * k * k, B * P);
        for (int n = 0; n < B; ++n)
        for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Wo; ++ox)
        {
            int col = n * P + oy * Wo + ox;
            for (int a = 0; a < k; ++a)
            for (int bb = 0; bb < k; ++bb)
            {
                int iy = oy + a - pad, ix = ox + bb - pad;
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                for (int ci = 0; ci < C_in; ++ci)
                    out(k_index(ci, a, bb), col) = x(in_index(iy, ix, ci), n);
            }
        }
        return out;
    }

    // Inverse of im2col for gradients: each input pixel appeared in several
    // windows, so its blame is the SUM of those windows' column gradients.
    Mat col2im(const Mat &dcols, int B) const
    {
        const int P = Ho * Wo;
        Mat dX = Mat::Zero(H * W * C_in, B);
        for (int n = 0; n < B; ++n)
        for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Wo; ++ox)
        {
            int col = n * P + oy * Wo + ox;
            for (int a = 0; a < k; ++a)
            for (int bb = 0; bb < k; ++bb)
            {
                int iy = oy + a - pad, ix = ox + bb - pad;
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                for (int ci = 0; ci < C_in; ++ci)
                    dX(in_index(iy, ix, ci), n) += dcols(k_index(ci, a, bb), col);
            }
        }
        return dX;
    }

    Mat forward(const Mat &x) override
    {
        const int B = int(x.cols()), P = Ho * Wo;
        cols = im2col(x);
        Mat Zf = (K * cols).colwise() + b; // [C_out x B*P] — a DenseLayer forward

        // Zf(co, n*P + p) and Z((p*C_out + co), n) occupy the same memory
        // order (that's why the HWC layout was chosen), so this is a free view.
        Mat Z = Eigen::Map<Mat>(Zf.data(), C_out * P, B);
        A = get_activation(activation)(Z);
        return A;
    }

    Mat backward(const Mat &dY) override
    {
        const int B = int(dY.cols()), P = Ho * Wo;
        Mat dZ = dY.cwiseProduct(get_derivative(activation)(A));
        Eigen::Map<Mat> dZf(dZ.data(), C_out, B * P); // free reshape back

        dK = dZf * cols.transpose();     // [C_out x B*P][B*P x C_in*k*k]
        db = dZf.rowwise().sum();        // [C_out]
        Mat dcols = K.transpose() * dZf; // [C_in*k*k x B*P]
        return col2im(dcols, B);
    }

    void step(float lr, float l2) override
    {
        K -= lr * (dK + l2 * K);
        b -= lr * db;
    }

    const Mat &weights() const override { return K; }
};
