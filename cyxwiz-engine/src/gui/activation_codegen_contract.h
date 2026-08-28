#pragma once

#include "../core/dense_activation_configuration_policy.h"

#include <charconv>
#include <optional>
#include <string>
#include <system_error>

namespace gui {

enum class ActivationCodegenTarget {
    PyTorch,
    TensorFlow,
    Keras,
    PyCyxWiz,
};

struct ActivationCodegenResult {
    bool handled = false;
    std::string expression;
    std::optional<std::string> error;
};

inline std::string ActivationCodegenFloat(float value) {
    char buffer[64]{};
    const auto converted =
        std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (converted.ec == std::errc{}) {
        return std::string(buffer, converted.ptr);
    }
    return std::to_string(value);
}

// Activation exports are intentionally functional for class-based generated
// models. They therefore do not consume a numbered stateful-layer slot. Keras
// Sequential export receives an equivalent layer expression from the same
// contract.
inline ActivationCodegenResult BuildActivationCodegen(
    ActivationCodegenTarget target,
    NodeType node_type,
    const std::map<std::string, std::string>& parameters,
    const std::string& input_expression = "x") {
    ActivationCodegenResult result;
    result.handled = cyxwiz::IsExecutableActivationNode(node_type);
    if (!result.handled) return result;

    cyxwiz::DenseActivationConfiguration configuration;
    result.error = cyxwiz::ResolveDenseActivationConfiguration(
        node_type, parameters, configuration);
    if (result.error) return result;

    const std::string slope =
        ActivationCodegenFloat(configuration.negative_slope);
    const std::string alpha =
        ActivationCodegenFloat(configuration.elu_alpha);

    if (target == ActivationCodegenTarget::PyTorch) {
        switch (node_type) {
            case NodeType::ReLU:
                result.expression = "F.relu(" + input_expression + ")";
                break;
            case NodeType::LeakyReLU:
                result.expression = "F.leaky_relu(" + input_expression +
                    ", negative_slope=" + slope + ")";
                break;
            case NodeType::ELU:
                result.expression = "F.elu(" + input_expression +
                    ", alpha=" + alpha + ")";
                break;
            case NodeType::GELU:
                result.expression = "F.gelu(" + input_expression +
                    ", approximate='tanh')";
                break;
            case NodeType::Swish:
                result.expression = "F.silu(" + input_expression + ")";
                break;
            case NodeType::Mish:
                result.expression = "F.mish(" + input_expression + ")";
                break;
            case NodeType::Sigmoid:
                result.expression = "torch.sigmoid(" + input_expression + ")";
                break;
            case NodeType::Tanh:
                result.expression = "torch.tanh(" + input_expression + ")";
                break;
            case NodeType::Softmax:
                result.expression = "F.softmax(" + input_expression +
                    ", dim=1)";
                break;
            default:
                break;
        }
        return result;
    }

    if (target == ActivationCodegenTarget::TensorFlow) {
        switch (node_type) {
            case NodeType::ReLU:
                result.expression = "tf.nn.relu(" + input_expression + ")";
                break;
            case NodeType::LeakyReLU:
                result.expression = "tf.nn.leaky_relu(" + input_expression +
                    ", alpha=" + slope + ")";
                break;
            case NodeType::ELU:
                result.expression = "tf.where(" + input_expression + " > 0, " +
                    input_expression + ", " + alpha + " * tf.math.expm1(" +
                    input_expression + "))";
                break;
            case NodeType::GELU:
                result.expression = "tf.nn.gelu(" + input_expression +
                    ", approximate=True)";
                break;
            case NodeType::Swish:
                result.expression = "tf.nn.silu(" + input_expression + ")";
                break;
            case NodeType::Mish:
                result.expression = input_expression +
                    " * tf.math.tanh(tf.math.softplus(" + input_expression + "))";
                break;
            case NodeType::Sigmoid:
                result.expression = "tf.nn.sigmoid(" + input_expression + ")";
                break;
            case NodeType::Tanh:
                result.expression = "tf.nn.tanh(" + input_expression + ")";
                break;
            case NodeType::Softmax:
                result.expression = "tf.nn.softmax(" + input_expression +
                    ", axis=1)";
                break;
            default:
                break;
        }
        return result;
    }

    if (target == ActivationCodegenTarget::PyCyxWiz) {
        switch (node_type) {
            case NodeType::ReLU:
                result.expression = "cx.relu(" + input_expression + ")";
                break;
            case NodeType::LeakyReLU:
                result.expression = "cx.leaky_relu(" + input_expression +
                    ", negative_slope=" + slope + ")";
                break;
            case NodeType::ELU:
                result.expression = "cx.elu(" + input_expression +
                    ", alpha=" + alpha + ")";
                break;
            case NodeType::GELU:
                result.expression = "cx.gelu(" + input_expression + ")";
                break;
            case NodeType::Swish:
                result.expression = "cx.swish(" + input_expression + ")";
                break;
            case NodeType::Mish:
                result.expression = "cx.mish(" + input_expression + ")";
                break;
            case NodeType::Sigmoid:
                result.expression = "cx.sigmoid(" + input_expression + ")";
                break;
            case NodeType::Tanh:
                result.expression = "cx.tanh(" + input_expression + ")";
                break;
            case NodeType::Softmax:
                result.expression = "cx.softmax(" + input_expression +
                    ", dim=1)";
                break;
            default:
                break;
        }
        return result;
    }

    switch (node_type) {
        case NodeType::ReLU:
            result.expression = "layers.ReLU()";
            break;
        case NodeType::LeakyReLU:
            result.expression = "layers.LeakyReLU(negative_slope=" + slope + ")";
            break;
        case NodeType::ELU:
            result.expression = "layers.ELU(alpha=" + alpha + ")";
            break;
        case NodeType::GELU:
            result.expression =
                "layers.Lambda(lambda x: tf.nn.gelu(x, approximate=True))";
            break;
        case NodeType::Swish:
            result.expression = "layers.Activation('swish')";
            break;
        case NodeType::Mish:
            result.expression =
                "layers.Lambda(lambda x: x * tf.math.tanh(tf.math.softplus(x)))";
            break;
        case NodeType::Sigmoid:
            result.expression = "layers.Activation('sigmoid')";
            break;
        case NodeType::Tanh:
            result.expression = "layers.Activation('tanh')";
            break;
        case NodeType::Softmax:
            result.expression = "layers.Softmax(axis=1)";
            break;
        default:
            break;
    }
    return result;
}

} // namespace gui
