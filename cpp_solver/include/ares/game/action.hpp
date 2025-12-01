#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace ares {

// Action types
enum class ActionType : uint8_t {
    Fold = 0,
    Check,
    Call,
    Bet,
    Raise,
    AllIn
};

// Action representation
struct Action {
    ActionType type;
    float amount;  // In big blinds

    // Default constructor
    Action() : type(ActionType::Fold), amount(0.0f) {}

    // Explicit constructors for each type
    static Action fold() { return {ActionType::Fold, 0.0f}; }
    static Action check() { return {ActionType::Check, 0.0f}; }
    static Action call() { return {ActionType::Call, 0.0f}; }
    static Action bet(float amt) { return {ActionType::Bet, amt}; }
    static Action raise_to(float amt) { return {ActionType::Raise, amt}; }
    static Action all_in(float amt) { return {ActionType::AllIn, amt}; }

    bool operator==(const Action& other) const {
        if (type != other.type) return false;
        // For types with amounts, compare amounts
        if (type == ActionType::Bet || type == ActionType::Raise ||
            type == ActionType::AllIn) {
            // Allow small floating point differences
            return std::abs(amount - other.amount) < 0.01f;
        }
        return true;
    }

    bool operator!=(const Action& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        switch (type) {
            case ActionType::Fold: return "Fold";
            case ActionType::Check: return "Check";
            case ActionType::Call: return "Call";
            case ActionType::Bet: return "Bet " + std::to_string(amount);
            case ActionType::Raise: return "Raise to " + std::to_string(amount);
            case ActionType::AllIn: return "All-in " + std::to_string(amount);
        }
        return "Unknown";
    }

    // For action sequence encoding
    uint8_t encode() const {
        // Simple encoding: type + discretized amount
        // Amounts are discretized into buckets: 0.5x, 1x, 2x pot, all-in
        uint8_t base = static_cast<uint8_t>(type);
        if (type == ActionType::Bet || type == ActionType::Raise) {
            // Could add size bucket here
            return base;
        }
        return base;
    }

private:
    Action(ActionType t, float a) : type(t), amount(a) {}
};

// Street representation
enum class Street : uint8_t {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3
};

constexpr int NUM_STREETS = 4;

inline std::string street_to_string(Street s) {
    switch (s) {
        case Street::Preflop: return "Preflop";
        case Street::Flop: return "Flop";
        case Street::Turn: return "Turn";
        case Street::River: return "River";
    }
    return "Unknown";
}

inline int street_board_cards(Street s) {
    switch (s) {
        case Street::Preflop: return 0;
        case Street::Flop: return 3;
        case Street::Turn: return 4;
        case Street::River: return 5;
    }
    return 0;
}

}  // namespace ares
