#pragma once

#include "cards.hpp"
#include <array>
#include <string>
#include <cmath>

namespace rl_env {

// Action types matching Python ActionType enum
enum class ActionType : uint8_t {
    FOLD = 0,
    CHECK = 1,
    CALL = 2,
    BET = 3,
    RAISE = 4,
    ALL_IN = 5
};

struct Action {
    ActionType type;
    float amount;  // In big blinds

    Action() : type(ActionType::CHECK), amount(0) {}
    Action(ActionType t, float a = 0) : type(t), amount(a) {}

    static Action fold() { return Action(ActionType::FOLD); }
    static Action check() { return Action(ActionType::CHECK); }
    static Action call() { return Action(ActionType::CALL); }
    static Action bet(float a) { return Action(ActionType::BET, a); }
    static Action raise_to(float a) { return Action(ActionType::RAISE, a); }
    static Action all_in(float a) { return Action(ActionType::ALL_IN, a); }

    char to_code() const {
        switch (type) {
            case ActionType::FOLD: return 'f';
            case ActionType::CHECK: return 'x';
            case ActionType::CALL: return 'c';
            case ActionType::BET: return 'b';
            case ActionType::RAISE: return 'r';
            case ActionType::ALL_IN: return 'a';
        }
        return '?';
    }
};

// Game state matching Python HoldemState
struct GameState {
    // Cards
    std::array<Card, 2> hero_hole;
    std::array<Card, 2> villain_hole;
    std::array<Card, 5> board;
    int board_size;

    // Game state
    int street;  // 0=preflop, 1=flop, 2=turn, 3=river
    float pot;
    float hero_stack;
    float villain_stack;
    float hero_invested;  // This street
    float villain_invested;  // This street

    // Tracking
    std::string action_history;
    int current_player;  // 0=hero, 1=villain
    int button;  // 0=hero is button, 1=villain is button

    // Terminal state
    bool terminal;
    bool folded;  // Someone folded
    int folder;   // Who folded (-1 if no one)

    GameState() {
        hero_hole.fill(NO_CARD);
        villain_hole.fill(NO_CARD);
        board.fill(NO_CARD);
        board_size = 0;
        street = 0;
        pot = 0;
        hero_stack = 0;
        villain_stack = 0;
        hero_invested = 0;
        villain_invested = 0;
        current_player = 0;
        button = 0;
        terminal = false;
        folded = false;
        folder = -1;
    }

    float to_call() const {
        if (current_player == 0) {
            return std::max(0.0f, villain_invested - hero_invested);
        } else {
            return std::max(0.0f, hero_invested - villain_invested);
        }
    }

    float current_stack() const {
        return current_player == 0 ? hero_stack : villain_stack;
    }

    float min_raise() const {
        float tc = to_call();
        if (tc > 0) {
            return tc * 2;  // Must at least double
        }
        return street == 0 ? 2.0f : 1.0f;  // 2bb preflop, 1bb postflop
    }

    bool is_terminal() const {
        return terminal;
    }

    // Get payoff for player (0=hero, 1=villain)
    float get_payoff(int player) const;

    // Check if betting round should advance
    bool should_advance_street() const;

    // Apply action and return new state
    GameState apply_action(const Action& action) const;
};

}  // namespace rl_env
