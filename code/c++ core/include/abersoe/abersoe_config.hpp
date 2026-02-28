#pragma once

#include <cstddef>
#include <string>

#include "abersoe/abersoe_model.hpp"

namespace abersoe {

struct AberSOERuntimeConfig {
    std::size_t steps = 1000;
    std::size_t sample_stride = 1;
    bool strict_finite = true;
    bool stochastic_forcing = false;
    double forcing_noise_std = 0.0;
    std::size_t stochastic_channel = 0;
    unsigned long long seed = 123456789ULL;
};

bool validate_model_spec(const AberSOEModelSpec& model, std::string* error = nullptr);
bool validate_runtime_config(const AberSOERuntimeConfig& cfg, std::string* error = nullptr);

} // namespace abersoe
