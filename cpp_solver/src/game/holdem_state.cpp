#include "ares/game/holdem_state.hpp"
#include "ares/core/hand_evaluator.hpp"
#include <sstream>
#include <algorithm>

namespace ares {

HoldemState HoldemState::create_initial(const Config& config) {
    HoldemState state;
    state.config_ = config;

    // Initialize stacks
    state.stacks_[0] = config.starting_stack;
    state.stacks_[1] = config.starting_stack;

    // Post blinds
    state.bets_[BUTTON] = config.small_blind;      // Button posts SB
    state.bets_[BIG_BLIND] = config.big_blind;     // BB posts BB
    state.stacks_[BUTTON] -= config.small_blind;
    state.stacks_[BIG_BLIND] -= config.big_blind;

    state.pot_ = config.small_blind + config.big_blind;

    // Initialize game state
    state.street_ = Street::Preflop;
    state.current_player_ = BUTTON;  // Button acts first preflop
    state.last_aggressor_ = BIG_BLIND;  // BB is "aggressor" due to blind
    state.actions_this_round_ = 0;
    state.is_terminal_ = false;
    state.needs_deal_ = true;  // Need to deal hole cards
    state.showdown_ = false;

    // Initialize cards to NO_CARD
    state.hands_[0] = HoleCards();
    state.hands_[1] = HoleCards();
    state.board_.fill(NO_CARD);
    state.board_count_ = 0;

    return state;
}

HoldemState HoldemState::create_with_hands(
    const HoleCards& hand0,
    const HoleCards& hand1,
    const Config& config
) {
    HoldemState state = create_initial(config);
    state = state.deal_hole_cards(hand0, hand1);
    return state;
}

HoldemState HoldemState::create_from_position(
    const HoleCards& hero_hand,
    int hero_position,
    const std::vector<Card>& board,
    float pot,
    float hero_stack,
    float villain_stack,
    float hero_bet,
    float villain_bet,
    int current_player,
    const Config& config
) {
    HoldemState state;
    state.config_ = config;

    // Position mapping:
    // hero_position=0 means hero is BB/OOP -> HoldemState player BIG_BLIND (1)
    // hero_position=1 means hero is BTN/IP -> HoldemState player BUTTON (0)
    //
    // HoldemState convention:
    // Player 0 = BUTTON = SB preflop, IP postflop
    // Player 1 = BIG_BLIND = BB preflop, OOP postflop

    // Create placeholder villain hand (different from hero)
    Card v1 = index_to_card(0);
    Card v2 = index_to_card(1);
    int h0 = card_to_index(hero_hand[0]);
    int h1 = card_to_index(hero_hand[1]);
    int vi = 0;
    while (vi == h0 || vi == h1) vi++;
    v1 = index_to_card(vi);
    vi++;
    while (vi == h0 || vi == h1) vi++;
    v2 = index_to_card(vi);
    HoleCards villain_hand(v1, v2);

    if (hero_position == 0) {
        // Hero is BB/OOP -> HoldemState player 1 (BIG_BLIND)
        // Villain is BTN/IP -> HoldemState player 0 (BUTTON)
        state.stacks_[0] = villain_stack;  // Villain is P0 (BUTTON)
        state.stacks_[1] = hero_stack;     // Hero is P1 (BIG_BLIND)
        state.bets_[0] = villain_bet;
        state.bets_[1] = hero_bet;
        state.hands_[0] = villain_hand;
        state.hands_[1] = hero_hand;
    } else {
        // Hero is BTN/IP -> HoldemState player 0 (BUTTON)
        // Villain is BB/OOP -> HoldemState player 1 (BIG_BLIND)
        state.stacks_[0] = hero_stack;     // Hero is P0 (BUTTON)
        state.stacks_[1] = villain_stack;  // Villain is P1 (BIG_BLIND)
        state.bets_[0] = hero_bet;
        state.bets_[1] = villain_bet;
        state.hands_[0] = hero_hand;
        state.hands_[1] = villain_hand;
    }

    state.pot_ = pot;

    // Determine street from board count
    if (board.empty()) {
        state.street_ = Street::Preflop;
    } else if (board.size() == 3) {
        state.street_ = Street::Flop;
    } else if (board.size() == 4) {
        state.street_ = Street::Turn;
    } else {
        state.street_ = Street::River;
    }

    // Copy board cards
    state.board_.fill(NO_CARD);
    state.board_count_ = static_cast<int>(board.size());
    for (size_t i = 0; i < board.size() && i < 5; ++i) {
        state.board_[i] = board[i];
    }

    state.current_player_ = current_player;
    state.last_aggressor_ = -1;  // Unknown
    state.actions_this_round_ = 0;
    state.is_terminal_ = false;
    state.needs_deal_ = false;  // Cards are dealt
    state.showdown_ = false;

    return state;
}

float HoldemState::to_call() const {
    int opp = 1 - current_player_;
    float diff = bets_[opp] - bets_[current_player_];
    return std::max(0.0f, diff);
}

float HoldemState::effective_stack() const {
    return std::min(stacks_[0] + bets_[0], stacks_[1] + bets_[1]);
}

float HoldemState::utility(int player) const {
    if (!is_terminal_) return 0.0f;

    if (showdown_) {
        // Determine winner by hand strength
        std::array<Card, 5> b = board_;

        HandEvaluator::initialize();
        auto rank0 = HandEvaluator::evaluate(hands_[0], b);
        auto rank1 = HandEvaluator::evaluate(hands_[1], b);

        if (rank0 < rank1) {
            // Player 0 wins
            return player == 0 ? pot_ / 2 : -pot_ / 2;
        } else if (rank1 < rank0) {
            // Player 1 wins
            return player == 1 ? pot_ / 2 : -pot_ / 2;
        } else {
            // Tie - split pot
            return 0.0f;
        }
    } else {
        // Someone folded - current player folded, opponent wins
        // The utility is the pot that was won/lost
        // Player who folded loses their investment
        float invested = config_.starting_stack - stacks_[player] - bets_[player];
        invested += bets_[player];  // Current bet is also lost on fold

        // Actually simpler: winner gets pot, loser gets nothing
        // If current player folded, opponent wins
        int winner = 1 - current_player_;
        if (player == winner) {
            return pot_ - (config_.starting_stack - stacks_[player]);
        } else {
            return -(config_.starting_stack - stacks_[player]);
        }
    }
}

std::vector<Action> HoldemState::get_legal_actions() const {
    std::vector<Action> actions;

    if (is_terminal_ || needs_deal_) {
        return actions;
    }

    float call_amount = to_call();
    float stack = stacks_[current_player_];
    float current_bet = bets_[current_player_];
    float opp_bet = bets_[1 - current_player_];

    // Can always fold if there's a bet to call
    if (call_amount > 0) {
        actions.push_back(Action::fold());
    }

    // Check is only valid if no bet to call
    if (call_amount == 0) {
        actions.push_back(Action::check());
    }

    // Call if there's a bet and we have chips
    if (call_amount > 0 && stack > 0) {
        if (call_amount >= stack) {
            // All-in call
            actions.push_back(Action::all_in(current_bet + stack));
        } else {
            actions.push_back(Action::call());
        }
    }

    // Bet/Raise sizing
    float min_raise = config_.big_blind;  // Minimum raise is 1 BB
    if (call_amount > 0) {
        // Minimum raise is the size of the previous raise
        min_raise = std::max(min_raise, call_amount);
    }

    float chips_after_call = stack - std::min(call_amount, stack);

    if (chips_after_call > 0) {
        // Can bet or raise
        if (call_amount == 0) {
            // Betting (no bet yet)
            // Discrete bet sizes: 0.5x, 1x pot
            float pot_bet = pot_;
            float half_pot = pot_ * 0.5f;

            if (half_pot <= stack) {
                actions.push_back(Action::bet(half_pot));
            }
            if (pot_bet <= stack && pot_bet > half_pot + 0.1f) {
                actions.push_back(Action::bet(pot_bet));
            }

            // All-in if not already covered
            if (stack > pot_bet) {
                actions.push_back(Action::all_in(current_bet + stack));
            }
        } else {
            // Raising
            float raise_to_min = opp_bet + min_raise;
            float pot_after_call = pot_ + call_amount;

            // Raise sizes: min raise, pot raise
            if (raise_to_min <= current_bet + stack) {
                actions.push_back(Action::raise_to(raise_to_min));
            }

            float pot_raise = opp_bet + pot_after_call;
            if (pot_raise <= current_bet + stack && pot_raise > raise_to_min + 0.1f) {
                actions.push_back(Action::raise_to(pot_raise));
            }

            // All-in raise
            float all_in_amount = current_bet + stack;
            if (all_in_amount > opp_bet && all_in_amount > pot_raise) {
                actions.push_back(Action::all_in(all_in_amount));
            }
        }
    }

    return actions;
}

HoldemState HoldemState::apply_action(const Action& action) const {
    HoldemState new_state = *this;
    // Store action with player info for history encoding
    new_state.history_.push_back(action.with_player(current_player_));

    int player = current_player_;
    int opp = 1 - player;

    switch (action.type) {
        case ActionType::Fold:
            new_state.is_terminal_ = true;
            new_state.showdown_ = false;
            break;

        case ActionType::Check:
            new_state.actions_this_round_++;
            // If both players checked, move to next street
            if (new_state.actions_this_round_ >= 2 ||
                (street_ == Street::Preflop && player == BIG_BLIND)) {
                new_state.advance_street();
            } else {
                new_state.current_player_ = opp;
            }
            break;

        case ActionType::Call: {
            float call_amount = std::min(to_call(), stacks_[player]);
            new_state.stacks_[player] -= call_amount;
            new_state.bets_[player] += call_amount;
            new_state.pot_ += call_amount;
            new_state.actions_this_round_++;

            // Check if this closes the action
            if (new_state.bets_[0] == new_state.bets_[1]) {
                new_state.advance_street();
            } else {
                new_state.current_player_ = opp;
            }
            break;
        }

        case ActionType::Bet:
        case ActionType::Raise:
            new_state.stacks_[player] -= (action.amount - bets_[player]);
            new_state.pot_ += (action.amount - bets_[player]);
            new_state.bets_[player] = action.amount;
            new_state.last_aggressor_ = player;
            new_state.actions_this_round_++;
            new_state.current_player_ = opp;
            break;

        case ActionType::AllIn: {
            float all_in = stacks_[player];
            new_state.stacks_[player] = 0;
            new_state.pot_ += all_in;
            new_state.bets_[player] += all_in;
            new_state.last_aggressor_ = player;
            new_state.actions_this_round_++;

            // Check if opponent also all-in or if action closes
            if (new_state.stacks_[opp] == 0 ||
                new_state.bets_[player] <= new_state.bets_[opp]) {
                // All-in call or smaller all-in - advance to showdown
                new_state.advance_street();
            } else {
                new_state.current_player_ = opp;
            }
            break;
        }
    }

    return new_state;
}

void HoldemState::advance_street() {
    // Pot is already accumulated during betting (in apply_action)
    // Just reset bet tracking for the new street
    bets_[0] = 0;
    bets_[1] = 0;
    actions_this_round_ = 0;

    // Check if either player is all-in
    bool all_in = (stacks_[0] == 0 || stacks_[1] == 0);

    if (street_ == Street::River || all_in) {
        // Showdown
        is_terminal_ = true;
        showdown_ = true;
        return;
    }

    // Move to next street
    street_ = next_street();
    needs_deal_ = true;

    // Postflop: BB acts first
    current_player_ = BIG_BLIND;
}

Street HoldemState::next_street() const {
    switch (street_) {
        case Street::Preflop: return Street::Flop;
        case Street::Flop: return Street::Turn;
        case Street::Turn: return Street::River;
        case Street::River: return Street::River;  // No next street
    }
    return Street::River;
}

HandMask HoldemState::used_cards() const {
    HandMask mask = 0;

    // Add hole cards
    mask |= hands_[0].to_mask();
    mask |= hands_[1].to_mask();

    // Add board cards
    for (int i = 0; i < board_count_; ++i) {
        mask = add_card_to_hand(mask, board_[i]);
    }

    return mask;
}

HoldemState HoldemState::deal_hole_cards(
    const HoleCards& hand0,
    const HoleCards& hand1
) const {
    HoldemState new_state = *this;
    new_state.hands_[0] = hand0;
    new_state.hands_[1] = hand1;
    new_state.needs_deal_ = false;
    return new_state;
}

HoldemState HoldemState::deal_board(const std::vector<Card>& cards) const {
    HoldemState new_state = *this;

    for (const Card& c : cards) {
        if (new_state.board_count_ < 5) {
            new_state.board_[new_state.board_count_++] = c;
        }
    }

    new_state.needs_deal_ = false;
    return new_state;
}

uint64_t HoldemState::info_set_key(int player) const {
    // Create a hash key combining:
    // 1. Player's hole cards (canonical)
    // 2. Board cards
    // 3. Betting history

    uint64_t key = 0;

    // Hole cards contribution
    key ^= static_cast<uint64_t>(hands_[player].canonical_index()) << 40;

    // Board contribution
    uint64_t board_key = 0;
    for (int i = 0; i < board_count_; ++i) {
        board_key ^= static_cast<uint64_t>(card_to_index(board_[i])) << (i * 6);
    }
    key ^= board_key << 16;

    // Action history contribution (simplified)
    uint64_t action_key = 0;
    for (size_t i = 0; i < history_.size() && i < 8; ++i) {
        action_key |= static_cast<uint64_t>(history_[i].encode()) << (i * 4);
    }
    key ^= action_key;

    return key;
}

std::string HoldemState::to_string() const {
    std::ostringstream ss;

    ss << "Street: " << street_to_string(street_) << "\n";
    ss << "Pot: " << pot_ << "bb\n";
    ss << "Stacks: [" << stacks_[0] << ", " << stacks_[1] << "]\n";
    ss << "Bets: [" << bets_[0] << ", " << bets_[1] << "]\n";

    if (hands_[0][0] != NO_CARD) {
        ss << "P0 hand: " << hands_[0].to_string() << "\n";
    }
    if (hands_[1][0] != NO_CARD) {
        ss << "P1 hand: " << hands_[1].to_string() << "\n";
    }

    if (board_count_ > 0) {
        ss << "Board: ";
        for (int i = 0; i < board_count_; ++i) {
            ss << card_to_string(board_[i]) << " ";
        }
        ss << "\n";
    }

    ss << "Current player: " << current_player_ << "\n";
    ss << "Terminal: " << (is_terminal_ ? "yes" : "no") << "\n";

    return ss.str();
}

}  // namespace ares
