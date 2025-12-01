/**
 * ARES-HU CFR Training Executable
 *
 * Trains a DCFR+ solver with QRE for heads-up no-limit hold'em.
 *
 * Usage:
 *   ./train_cfr --iterations 100000 --stack 100 --output strategy.bin
 */

#include "ares/cfr/dcfr_solver.hpp"
#include "ares/game/holdem_state.hpp"
#include "ares/core/hand_evaluator.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>

using namespace ares;

void print_usage(const char* program) {
    std::cout << "ARES-HU CFR Trainer\n\n"
              << "Usage: " << program << " [options]\n\n"
              << "Options:\n"
              << "  --iterations N    Number of CFR iterations (default: 10000)\n"
              << "  --stack N         Starting stack in BB (default: 20)\n"
              << "  --threads N       Number of threads (default: 8)\n"
              << "  --qre-tau N       QRE temperature (default: 1.0, 0=Nash)\n"
              << "  --output FILE     Output file for strategy (default: strategy.bin)\n"
              << "  --checkpoint DIR  Directory for checkpoints\n"
              << "  --checkpoint-freq N  Checkpoint every N iterations (default: 10000)\n"
              << "  --export-training-data FILE  Export training data for neural network\n"
              << "  --num-samples N   Number of training samples to export (default: 100000)\n"
              << "  --quiet           Suppress progress output\n"
              << "  --help            Show this help\n";
}

int main(int argc, char* argv[]) {
    // Default configuration
    DCFRConfig config;
    config.iterations = 10000;
    config.starting_stack = 20.0f;
    config.num_threads = 8;
    config.equilibrium = EquilibriumType::QRE;
    config.qre_tau = 1.0f;
    config.verbose = true;
    config.checkpoint_freq = 10000;

    std::string output_file = "strategy.bin";
    std::string training_data_file = "";
    int num_samples = 100000;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.iterations = std::atoi(argv[++i]);
        } else if (arg == "--stack" && i + 1 < argc) {
            config.starting_stack = std::atof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::atoi(argv[++i]);
        } else if (arg == "--qre-tau" && i + 1 < argc) {
            float tau = std::atof(argv[++i]);
            config.qre_tau = tau;
            config.equilibrium = (tau > 0) ? EquilibriumType::QRE : EquilibriumType::NASH;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            config.checkpoint_dir = argv[++i];
        } else if (arg == "--checkpoint-freq" && i + 1 < argc) {
            config.checkpoint_freq = std::atoi(argv[++i]);
        } else if (arg == "--export-training-data" && i + 1 < argc) {
            training_data_file = argv[++i];
        } else if (arg == "--num-samples" && i + 1 < argc) {
            num_samples = std::atoi(argv[++i]);
        } else if (arg == "--quiet") {
            config.verbose = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Print configuration
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════╗\n"
              << "║            ARES-HU CFR Trainer v1.0                      ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n"
              << "Configuration:\n"
              << "  Iterations:    " << config.iterations << "\n"
              << "  Stack size:    " << config.starting_stack << " BB\n"
              << "  Threads:       " << config.num_threads << "\n"
              << "  Equilibrium:   " << (config.equilibrium == EquilibriumType::QRE ? "QRE" : "Nash") << "\n";

    if (config.equilibrium == EquilibriumType::QRE) {
        std::cout << "  QRE tau:       " << config.qre_tau << "\n";
    }

    std::cout << "  Output:        " << output_file << "\n";

    if (!config.checkpoint_dir.empty()) {
        std::cout << "  Checkpoints:   " << config.checkpoint_dir
                  << " (every " << config.checkpoint_freq << " iters)\n";
    }

    std::cout << "\n";

    // Initialize hand evaluator lookup tables
    std::cout << "Initializing hand evaluator...\n";
    HandEvaluator::initialize();

    // Create and train solver
    std::cout << "Starting training...\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    DCFRPlusSolver solver(config);
    solver.train();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n\nTraining completed in " << duration.count() / 60 << "m "
              << duration.count() % 60 << "s\n";

    // Save final strategy
    std::cout << "Saving strategy to " << output_file << "...\n";
    solver.save(output_file);

    // Export training data if requested
    if (!training_data_file.empty()) {
        std::cout << "\nExporting " << num_samples << " training samples to "
                  << training_data_file << "...\n";
        solver.export_training_data(training_data_file, num_samples);
    }

    // Print statistics
    std::cout << "\nFinal Statistics:\n"
              << "  Information sets: " << solver.num_info_sets() << "\n"
              << "  Iterations:       " << solver.current_iteration() << "\n";

    std::cout << "\nDone!\n";

    return 0;
}
