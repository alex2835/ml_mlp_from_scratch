#pragma once
#include "layer.hpp"

// 2x2 max pooling with stride 2: keeps the largest value of every 2x2 block,
// per channel. No parameters. Halves H and W.
struct MaxPool2D : Layer
{
    int C, H, W, Ho, Wo;
    Eigen::MatrixXi argmax; // [C*Ho*Wo x B]: flat input index of each block's winner

    MaxPool2D(int C, int H, int W) : C(C), H(H), W(W), Ho(H / 2), Wo(W / 2) {}

    int in_index(int y, int x, int c) const  { return (y * W + x) * C + c; }
    int out_index(int y, int x, int c) const { return (y * Wo + x) * C + c; }

    Mat forward(const Mat &x) override
    {
        const int B = int(x.cols());
        Mat Y(C * Ho * Wo, B);
        argmax.resize(C * Ho * Wo, B);
        for (int n = 0; n < B; ++n)
        for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Wo; ++ox)
        for (int c = 0; c < C; ++c)
        {
            int best = in_index(2 * oy, 2 * ox, c);
            for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int i = in_index(2 * oy + dy, 2 * ox + dx, c);
                if (x(i, n) > x(best, n)) best = i;
            }
            Y(out_index(oy, ox, c), n) = x(best, n);
            argmax(out_index(oy, ox, c), n) = best;
        }
        return Y;
    }

    // Only the winner influenced the output, so it gets the whole gradient;
    // the other three pixels of the block get zero.
    Mat backward(const Mat &dY) override
    {
        const int B = int(dY.cols());
        Mat dX = Mat::Zero(H * W * C, B);
        for (int n = 0; n < B; ++n)
            for (int o = 0; o < dY.rows(); ++o)
                dX(argmax(o, n), n) += dY(o, n);
        return dX;
    }
};
