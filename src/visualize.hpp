#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include "helpers.hpp"

// Save each row of W as a grayscale PGM image [rows x cols]; needs
// W.cols() == rows*cols. For a first Dense layer on MNIST, row c holds the
// 784 weights that produce output c — reshaped to 28x28 it shows what that
// output "looks for": white = pixels that push it up, black = push it down.
inline void save_weight_images(const Mat &W, int rows, int cols, const std::string &dir)
{
    std::filesystem::create_directories(dir);
    for (int c = 0; c < W.rows(); ++c)
    {
        Vec w = W.row(c).transpose();
        float lo = w.minCoeff();
        float hi = w.maxCoeff();

        std::ofstream out(dir + "/class_" + std::to_string(c) + ".pgm", std::ios::binary);
        out << "P5\n" << cols << " " << rows << "\n255\n";
        for (int p = 0; p < rows * cols; ++p)
            out.put(char(255.0f * (w[p] - lo) / (hi - lo)));
    }
}
