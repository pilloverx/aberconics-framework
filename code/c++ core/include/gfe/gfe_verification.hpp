#pragma once

#include <string>
#include <vector>

namespace gfe {

struct VerificationMetric {
    std::string name;
    double value;
};

struct VerificationResult {
    std::string operation;
    bool pass = false;
    std::string note;
    std::vector<VerificationMetric> metrics;
};

std::vector<std::string> list_verification_operations();
std::vector<std::string> default_verification_operations();

VerificationResult run_verification_operation(const std::string& operation);
std::vector<VerificationResult> run_verification_operations(const std::vector<std::string>& operations);

} // namespace gfe
