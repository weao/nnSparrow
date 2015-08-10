
#pragma once
#include <fstream>
#include <stdint.h>
#include <vector>

#include <limits>


typedef std::vector<double> vec_t;
typedef size_t label_t;

template<typename T>
T* reverse_endian(T* p) {
    std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
    return p;
}

inline bool is_little_endian() {
    int x = 1;
    return *(char*) &x != 0;
}


void parse_mnist_labels(const std::string label_file, std::vector<label_t> *labels) {
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        printf("failed to open file: %s\n",label_file.c_str());

    uint32_t magic_number, num_items;

    ifs.read((char*) &magic_number, 4);
    ifs.read((char*) &num_items, 4);

    if (is_little_endian()) { // MNIST data is big-endian format
        reverse_endian(&magic_number);
        reverse_endian(&num_items);
    }

    if (magic_number != 0x00000801 || num_items <= 0)
        printf("MNIST label-file format error\n");

    for (size_t i = 0; i < num_items; i++) {
        uint8_t label;
        ifs.read((char*) &label, 1);
        labels->push_back((label_t) label);
    }
}

struct mnist_header {
    uint32_t magic_number;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;
};

void parse_mnist_header(std::ifstream& ifs, mnist_header& header) {
    ifs.read((char*) &header.magic_number, 4);
    ifs.read((char*) &header.num_items, 4);
    ifs.read((char*) &header.num_rows, 4);
    ifs.read((char*) &header.num_cols, 4);

    if (is_little_endian()) {
        reverse_endian(&header.magic_number);
        reverse_endian(&header.num_items);
        reverse_endian(&header.num_rows);
        reverse_endian(&header.num_cols);
    }

    if (header.magic_number != 0x00000803 || header.num_items <= 0)
        printf("MNIST label-file format error\n");
    if (ifs.fail() || ifs.bad())
        printf("file error\n");
}

void parse_mnist_image(std::ifstream& ifs,
    const mnist_header& header,
    double scale_min,
    double scale_max,
    int x_padding,
    int y_padding,
    vec_t& dst) {
    const int width = header.num_cols + 2 * x_padding;
    const int height = header.num_rows + 2 * y_padding;

    std::vector<uint8_t> image_vec(header.num_rows * header.num_cols);

    ifs.read((char*) &image_vec[0], header.num_rows * header.num_cols);

    dst.resize(width * height, scale_min);

    for (size_t y = 0; y < header.num_rows; y++)
    for (size_t x = 0; x < header.num_cols; x++)
        dst[width * (y + y_padding) + x + x_padding]
        = (image_vec[y * header.num_cols + x] / 255.0) * (scale_max - scale_min) + scale_min;
}

void parse_mnist_images(const std::string image_file,
    std::vector<vec_t> *images,
    double scale_min = -1.0,
    double scale_max = 1.0,
    int x_padding = 2,
    int y_padding = 2) {
    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        printf("failed to open file: %s\n", image_file.c_str());

    mnist_header header;

    parse_mnist_header(ifs, header);

    for (size_t i = 0; i < header.num_items; i++) {
        vec_t image;
        parse_mnist_image(ifs, header, scale_min, scale_max, x_padding, y_padding, image);
        images->push_back(image);
    }
}
