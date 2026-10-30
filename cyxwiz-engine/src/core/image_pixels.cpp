#include "image_pixels.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

uint8_t Luminance(const uint8_t* p) {
    const float y = 0.2989f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    return static_cast<uint8_t>(std::clamp(std::lround(y), 0L, 255L));
}

void Finish(ChannelHistogram& h, double sum, double sum_sq, uint64_t n) {
    if (n == 0) return;
    h.mean = sum / static_cast<double>(n);
    h.std_dev = std::sqrt(std::max(0.0, sum_sq / static_cast<double>(n) - h.mean * h.mean));
}

}  // namespace

ImagePixels DecodeImagePixels(const std::string& path) {
    ImagePixels out;
    try {
        cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            out.error = "This file cannot be read as an image.";
            return out;
        }
        // More than 4 channels (a multi-band TIFF): its first three as colour.
        if (image.channels() > 4) {
            cv::Mat three(image.rows, image.cols, CV_MAKETYPE(image.depth(), 3));
            const int pairs[] = {0, 0, 1, 1, 2, 2};
            cv::mixChannels(&image, 1, &three, 1, pairs, 3);
            image = three;
        }
        out.channels = image.channels();
        out.bits = image.depth() == CV_16U ? 16 : (image.depth() == CV_32F || image.depth() == CV_64F) ? 32 : 8;
        if (image.depth() == CV_16U) image.convertTo(image, CV_8U, 1.0 / 257.0);
        else if (image.depth() == CV_32F || image.depth() == CV_64F) image.convertTo(image, CV_8U, 255.0);
        else if (image.depth() != CV_8U) image.convertTo(image, CV_8U);
        cv::Mat rgba;
        switch (image.channels()) {
            case 1: cv::cvtColor(image, rgba, cv::COLOR_GRAY2RGBA); break;
            case 2: {  // grey + alpha
                std::vector<cv::Mat> planes;
                cv::split(image, planes);
                cv::merge(std::vector<cv::Mat>{planes[0], planes[0], planes[0], planes[1]}, rgba);
                break;
            }
            case 4: cv::cvtColor(image, rgba, cv::COLOR_BGRA2RGBA); break;
            default: cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA); break;
        }
        out.width = rgba.cols;
        out.height = rgba.rows;
        out.rgba.assign(rgba.datastart, rgba.dataend);
    } catch (const cv::Exception& e) {
        out = {};
        out.error = std::string("This file cannot be read as an image: ") + e.what();
    }
    return out;
}

ImageDisplay MakeImageDisplay(const ImagePixels& pixels, int max_side) {
    ImageDisplay out;
    if (pixels.rgba.empty() || pixels.width <= 0 || pixels.height <= 0) return out;
    const int side = std::max(pixels.width, pixels.height);
    const double scale = max_side > 0 && side > max_side ? static_cast<double>(max_side) / side : 1.0;
    out.width = std::max(1, static_cast<int>(pixels.width * scale));
    out.height = std::max(1, static_cast<int>(pixels.height * scale));
    ImagePixels shown;
    if (scale < 1.0) {
        const cv::Mat full(pixels.height, pixels.width, CV_8UC4, const_cast<uint8_t*>(pixels.rgba.data()));
        cv::Mat small;
        cv::resize(full, small, cv::Size(out.width, out.height), 0, 0, cv::INTER_AREA);
        shown.width = out.width;
        shown.height = out.height;
        shown.channels = pixels.channels;
        shown.rgba.assign(small.datastart, small.dataend);
    }
    const ImagePixels& source = scale < 1.0 ? shown : pixels;
    out.colour = source.rgba;
    out.luminance = PlaneRgba(source, ImagePlane::Luminance);
    if (pixels.HasAlpha()) out.alpha = PlaneRgba(source, ImagePlane::Alpha);
    return out;
}

ImageHistograms ComputeHistograms(const ImagePixels& pixels) {
    ImageHistograms h;
    const uint64_t n = static_cast<uint64_t>(pixels.width) * static_cast<uint64_t>(pixels.height);
    if (pixels.rgba.size() < n * 4) return h;
    ChannelHistogram* channel[5] = {&h.red, &h.green, &h.blue, &h.luminance, &h.alpha};
    double sum[5] = {}, sum_sq[5] = {};
    const int count = pixels.HasAlpha() ? 5 : 4;
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t* p = pixels.rgba.data() + i * 4;
        const uint8_t value[5] = {p[0], p[1], p[2], Luminance(p), p[3]};
        for (int c = 0; c < count; ++c) {
            ++channel[c]->bins[value[c] / (256 / kHistogramBins)];
            sum[c] += value[c];
            sum_sq[c] += static_cast<double>(value[c]) * value[c];
        }
    }
    for (int c = 0; c < count; ++c) Finish(*channel[c], sum[c], sum_sq[c], n);
    return h;
}

std::vector<uint8_t> PlaneRgba(const ImagePixels& pixels, ImagePlane plane) {
    const size_t n = static_cast<size_t>(pixels.width) * static_cast<size_t>(pixels.height);
    std::vector<uint8_t> out(n * 4, 255);
    if (pixels.rgba.size() < n * 4) return {};
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* p = pixels.rgba.data() + i * 4;
        const uint8_t v = plane == ImagePlane::Luminance ? Luminance(p) : p[3];
        out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = v;
    }
    return out;
}

}  // namespace cyxwiz
