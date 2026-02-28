#include "gfe/gfe_verification.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [--list] [--all] [--op NAME ...] [--csv PATH] [--strict|--no-strict]\n"
        << "\n"
        << "Defaults: run important GFE operations (same as --all).\n"
        << "\n"
        << "Options:\n"
        << "  --list        List available verification operations and exit\n"
        << "  --all         Run all registered operations\n"
        << "  --op NAME     Run a specific operation (repeatable)\n"
        << "  --csv PATH    Write metrics report as CSV\n"
        << "  --strict      Return non-zero if any operation fails (default)\n"
        << "  --no-strict   Always return zero unless CLI parsing fails\n";
}

void write_csv(const std::string& path, const std::vector<gfe::VerificationResult>& results) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open CSV path: " + path);
    }

    out << "operation,pass,note,metric,value\n";
    for (const auto& r : results) {
        if (r.metrics.empty()) {
            out << r.operation << ',' << (r.pass ? 1 : 0) << ',' << '"' << r.note << '"' << ",," << "\n";
            continue;
        }
        for (const auto& m : r.metrics) {
            out << r.operation << ','
                << (r.pass ? 1 : 0) << ','
                << '"' << r.note << '"' << ','
                << m.name << ','
                << std::setprecision(17) << m.value << "\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    bool list_only = false;
    bool run_all = false;
    bool strict = true;
    std::string csv_path;
    std::vector<std::string> ops;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--strict") {
            strict = true;
        } else if (arg == "--no-strict") {
            strict = false;
        } else if (arg == "--csv") {
            if (i + 1 >= argc) {
                std::cerr << "--csv requires a path\n";
                return 2;
            }
            csv_path = argv[++i];
        } else if (arg == "--op") {
            if (i + 1 >= argc) {
                std::cerr << "--op requires a name\n";
                return 2;
            }
            ops.emplace_back(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (list_only) {
        const auto available = gfe::list_verification_operations();
        for (const auto& name : available) {
            std::cout << name << '\n';
        }
        return 0;
    }

    if (run_all || ops.empty()) {
        ops = gfe::default_verification_operations();
    }

    std::vector<gfe::VerificationResult> results;
    try {
        results = gfe::run_verification_operations(ops);
    } catch (const std::exception& e) {
        std::cerr << "Verification failed to execute: " << e.what() << "\n";
        return 2;
    }

    bool all_pass = true;
    std::cout << "GFE CLI verification report\n";
    std::cout << "===========================\n";
    for (const auto& r : results) {
        all_pass = all_pass && r.pass;
        std::cout << (r.pass ? "[PASS] " : "[FAIL] ") << r.operation;
        if (!r.note.empty()) {
            std::cout << " - " << r.note;
        }
        std::cout << "\n";
        for (const auto& m : r.metrics) {
            std::cout << "  " << m.name << " = " << std::setprecision(10) << m.value << "\n";
        }
    }

    if (!csv_path.empty()) {
        try {
            write_csv(csv_path, results);
            std::cout << "CSV report written to: " << csv_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "CSV write failed: " << e.what() << "\n";
            return 2;
        }
    }

    if (strict && !all_pass) {
        return 1;
    }
    return 0;
}
