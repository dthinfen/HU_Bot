#include "ares/belief/public_belief_state.hpp"
#include "ares/core/hand_evaluator.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace ares {

// =============================================================================
// Hand Distribution Implementation
// =============================================================================

HandDistribution::HandDistribution() {
    probs_.fill(0.0f);
}

void HandDistribution::set_uniform() {
    float prob = 1.0f / NUM_HOLE_COMBOS;
    probs_.fill(prob);
}

void HandDistribution::set_uniform_unblocked(HandMask dead_cards) {
    int count = 0;

    // First count unblocked hands
    for (int i = 0; i < NUM_HOLE_COMBOS; ++i) {
        auto [c1, c2] = range_utils::combo_to_cards(i);
        HandMask hand_mask = (1ULL << c1) | (1ULL << c2);
        if ((hand_mask & dead_cards) == 0) {
            count++;
        }
    }

    // Set probabilities
    float prob = (count > 0) ? 1.0f / count : 0.0f;
    for (int i = 0; i < NUM_HOLE_COMBOS; ++i) {
        auto [c1, c2] = range_utils::combo_to_cards(i);
        HandMask hand_mask = (1ULL << c1) | (1ULL << c2);
        probs_[i] = ((hand_mask & dead_cards) == 0) ? prob : 0.0f;
    }
}

float HandDistribution::get(const HoleCards& hand) const {
    return probs_[hand_to_index(hand)];
}

void HandDistribution::set(const HoleCards& hand, float prob) {
    probs_[hand_to_index(hand)] = prob;
}

void HandDistribution::normalize() {
    float sum = std::accumulate(probs_.begin(), probs_.end(), 0.0f);
    if (sum > 0) {
        for (auto& p : probs_) {
            p /= sum;
        }
    }
}

void HandDistribution::block(HandMask dead_cards) {
    for (int i = 0; i < NUM_HOLE_COMBOS; ++i) {
        auto [c1, c2] = range_utils::combo_to_cards(i);
        HandMask hand_mask = (1ULL << c1) | (1ULL << c2);
        if (hand_mask & dead_cards) {
            probs_[i] = 0.0f;
        }
    }
    normalize();
}

void HandDistribution::update_with_action(
    const std::vector<float>& action_probs_per_hand,
    int action_taken
) {
    // Bayesian update: P(hand | action) ∝ P(action | hand) * P(hand)
    for (int i = 0; i < NUM_HOLE_COMBOS; ++i) {
        // action_probs_per_hand[i] is probability of taking this action with hand i
        probs_[i] *= action_probs_per_hand[i];
    }
    normalize();
}

float HandDistribution::entropy() const {
    float h = 0.0f;
    for (float p : probs_) {
        if (p > 1e-10f) {
            h -= p * std::log2(p);
        }
    }
    return h;
}

int HandDistribution::support_size() const {
    int count = 0;
    for (float p : probs_) {
        if (p > 1e-10f) count++;
    }
    return count;
}

int HandDistribution::hand_to_index(const HoleCards& hand) {
    int c1 = card_to_index(hand[0]);
    int c2 = card_to_index(hand[1]);
    return range_utils::cards_to_combo(c1, c2);
}

HoleCards HandDistribution::index_to_hand(int index) {
    auto [c1, c2] = range_utils::combo_to_cards(index);
    return HoleCards(index_to_card(c1), index_to_card(c2));
}

// =============================================================================
// Public Belief State Implementation
// =============================================================================

PublicBeliefState PublicBeliefState::from_state(const HoldemState& state) {
    PublicBeliefState pbs;

    pbs.street_ = state.street();
    pbs.pot_ = state.pot();
    pbs.stacks_[0] = state.stack(0);
    pbs.stacks_[1] = state.stack(1);
    pbs.bets_[0] = state.bet(0);
    pbs.bets_[1] = state.bet(1);
    pbs.board_ = state.board();
    pbs.board_count_ = state.board_count();
    pbs.current_player_ = state.current_player();
    pbs.is_terminal_ = state.is_terminal();
    pbs.history_ = state.action_history();

    // Get dead cards (board)
    HandMask dead = 0;
    for (int i = 0; i < pbs.board_count_; ++i) {
        dead = add_card_to_hand(dead, pbs.board_[i]);
    }

    // Initialize beliefs to uniform over unblocked hands
    pbs.beliefs_[0].set_uniform_unblocked(dead);
    pbs.beliefs_[1].set_uniform_unblocked(dead);

    return pbs;
}

PublicBeliefState PublicBeliefState::create_initial(float starting_stack) {
    PublicBeliefState pbs;

    pbs.starting_stack_ = starting_stack;
    pbs.street_ = Street::Preflop;
    pbs.pot_ = 1.5f;  // SB + BB
    pbs.stacks_[0] = starting_stack - 0.5f;  // Button posted SB
    pbs.stacks_[1] = starting_stack - 1.0f;  // BB posted BB
    pbs.bets_[0] = 0.5f;
    pbs.bets_[1] = 1.0f;
    pbs.board_.fill(NO_CARD);
    pbs.board_count_ = 0;
    pbs.current_player_ = 0;  // Button acts first preflop
    pbs.is_terminal_ = false;

    // Uniform beliefs over all hands
    pbs.beliefs_[0].set_uniform();
    pbs.beliefs_[1].set_uniform();

    return pbs;
}

PublicBeliefState PublicBeliefState::apply_action(
    const Action& action,
    const std::vector<std::vector<float>>& action_probs
) const {
    PublicBeliefState next = *this;
    next.history_.push_back(action);

    int player = current_player_;
    int opp = 1 - player;

    switch (action.type) {
        case ActionType::Fold:
            next.is_terminal_ = true;
            break;

        case ActionType::Check:
            // Check if action closes the round
            if (next.history_.size() >= 2) {
                // Simplified: advance street
                // Full implementation would track betting rounds properly
            }
            next.current_player_ = opp;
            break;

        case ActionType::Call: {
            float call_amount = bets_[opp] - bets_[player];
            next.stacks_[player] -= call_amount;
            next.bets_[player] = bets_[opp];
            next.pot_ += call_amount;
            next.current_player_ = opp;
            break;
        }

        case ActionType::Bet:
        case ActionType::Raise:
            next.stacks_[player] -= (action.amount - bets_[player]);
            next.pot_ += (action.amount - bets_[player]);
            next.bets_[player] = action.amount;
            next.current_player_ = opp;
            break;

        case ActionType::AllIn:
            next.pot_ += stacks_[player];
            next.bets_[player] += stacks_[player];
            next.stacks_[player] = 0;
            next.current_player_ = opp;
            break;
    }

    // Update opponent's belief about acting player
    // P(hand | action) ∝ P(action | hand) * P(hand)
    if (!action_probs.empty() && action_probs[player].size() == NUM_HOLE_COMBOS) {
        next.beliefs_[player].update_with_action(action_probs[player], 0);
    }

    return next;
}

PublicBeliefState PublicBeliefState::apply_board(const std::vector<Card>& new_cards) const {
    PublicBeliefState next = *this;

    for (const Card& c : new_cards) {
        if (next.board_count_ < 5) {
            next.board_[next.board_count_++] = c;
        }
    }

    // Update street
    if (next.board_count_ == 3) {
        next.street_ = Street::Flop;
    } else if (next.board_count_ == 4) {
        next.street_ = Street::Turn;
    } else if (next.board_count_ == 5) {
        next.street_ = Street::River;
    }

    // Block hands containing board cards
    HandMask dead = 0;
    for (int i = 0; i < next.board_count_; ++i) {
        dead = add_card_to_hand(dead, next.board_[i]);
    }
    next.beliefs_[0].block(dead);
    next.beliefs_[1].block(dead);

    // Reset bets for new street
    next.bets_[0] = 0;
    next.bets_[1] = 0;
    next.current_player_ = 1;  // BB acts first postflop

    return next;
}

std::vector<float> PublicBeliefState::encode() const {
    return PBSEncoder::encode(*this);
}

int PublicBeliefState::encoding_dim() {
    return PBSEncoder::encoding_dim();
}

std::vector<Action> PublicBeliefState::get_legal_actions() const {
    // Simplified legal action generation
    // Full implementation would mirror HoldemState::get_legal_actions()
    std::vector<Action> actions;

    if (is_terminal_) return actions;

    float call_amount = bets_[1 - current_player_] - bets_[current_player_];
    float stack = stacks_[current_player_];

    if (call_amount > 0) {
        actions.push_back(Action::fold());
        if (call_amount <= stack) {
            actions.push_back(Action::call());
        }
    } else {
        actions.push_back(Action::check());
    }

    // Add bet/raise options
    if (stack > call_amount) {
        float remaining = stack - call_amount;
        float pot_bet = pot_ + call_amount;

        // Half pot, pot, all-in
        if (remaining >= pot_bet * 0.5f) {
            actions.push_back(Action::bet(pot_bet * 0.5f));
        }
        if (remaining >= pot_bet) {
            actions.push_back(Action::bet(pot_bet));
        }
        actions.push_back(Action::all_in(bets_[current_player_] + stack));
    }

    return actions;
}

// =============================================================================
// PBS Encoder Implementation
// =============================================================================

std::vector<float> PBSEncoder::encode(const PublicBeliefState& pbs) {
    std::vector<float> features;
    features.reserve(encoding_dim());

    // Board encoding
    auto board_enc = encode_board(pbs.board(), pbs.board_count());
    features.insert(features.end(), board_enc.begin(), board_enc.end());

    // Pot and stacks
    auto pot_enc = encode_pot_and_stacks(
        pbs.pot(), pbs.stack(0), pbs.stack(1), 100.0f  // Assume 100bb starting
    );
    features.insert(features.end(), pot_enc.begin(), pot_enc.end());

    // History encoding
    auto hist_enc = encode_history(pbs.history());
    features.insert(features.end(), hist_enc.begin(), hist_enc.end());

    // Belief distributions
    auto belief_enc = encode_beliefs(pbs.belief(0), pbs.belief(1));
    features.insert(features.end(), belief_enc.begin(), belief_enc.end());

    // Board texture
    auto texture_enc = encode_board_texture(pbs.board(), pbs.board_count());
    features.insert(features.end(), texture_enc.begin(), texture_enc.end());

    // Position
    auto pos_enc = encode_position(pbs.current_player(), pbs.street());
    features.insert(features.end(), pos_enc.begin(), pos_enc.end());

    return features;
}

int PBSEncoder::encoding_dim() {
    return BOARD_DIM + POT_STACK_DIM + HISTORY_DIM + BELIEF_DIM + TEXTURE_DIM + POSITION_DIM;
}

std::vector<float> PBSEncoder::encode_board(const std::array<Card, 5>& board, int count) {
    std::vector<float> enc(BOARD_DIM, 0.0f);

    for (int i = 0; i < count; ++i) {
        int card_idx = card_to_index(board[i]);
        enc[i * 52 + card_idx] = 1.0f;
    }

    return enc;
}

std::vector<float> PBSEncoder::encode_pot_and_stacks(
    float pot, float stack0, float stack1, float starting
) {
    // Normalize by starting stack
    return {
        pot / starting,
        stack0 / starting,
        stack1 / starting,
        pot / std::max(1.0f, std::min(stack0, stack1))  // SPR
    };
}

std::vector<float> PBSEncoder::encode_history(const std::vector<Action>& history) {
    std::vector<float> enc(HISTORY_DIM, 0.0f);

    for (size_t i = 0; i < std::min(history.size(), static_cast<size_t>(MAX_HISTORY_LENGTH)); ++i) {
        int offset = i * 5;
        const Action& a = history[i];

        // One-hot action type
        enc[offset + static_cast<int>(a.type)] = 1.0f;

        // Normalized amount (for bets/raises)
        if (a.amount > 0) {
            enc[offset + 4] = a.amount / 100.0f;  // Normalize by assumed max bet
        }
    }

    return enc;
}

std::vector<float> PBSEncoder::encode_beliefs(
    const HandDistribution& belief0,
    const HandDistribution& belief1
) {
    std::vector<float> enc;
    enc.reserve(BELIEF_DIM);

    for (float p : belief0.probs()) {
        enc.push_back(p);
    }
    for (float p : belief1.probs()) {
        enc.push_back(p);
    }

    return enc;
}

std::vector<float> PBSEncoder::encode_board_texture(
    const std::array<Card, 5>& board,
    int count
) {
    std::vector<float> enc(TEXTURE_DIM, 0.0f);

    if (count == 0) return enc;

    // Count suits
    int suit_counts[4] = {0, 0, 0, 0};
    int rank_counts[13] = {0};
    uint16_t rank_bits = 0;

    for (int i = 0; i < count; ++i) {
        int s = get_suit_int(board[i]);
        int r = get_rank_int(board[i]);
        suit_counts[s]++;
        rank_counts[r]++;
        rank_bits |= (1 << r);
    }

    // Flush draw / made flush
    int max_suit = *std::max_element(suit_counts, suit_counts + 4);
    enc[0] = (max_suit >= 3) ? 1.0f : 0.0f;  // Flush draw possible
    enc[1] = (max_suit >= 4) ? 1.0f : 0.0f;  // 4-flush
    enc[2] = (max_suit >= 5) ? 1.0f : 0.0f;  // Made flush

    // Straight draw potential
    // Count connected cards
    int max_connected = 0;
    int connected = 0;
    for (int i = 0; i < 13; ++i) {
        if (rank_bits & (1 << i)) {
            connected++;
            max_connected = std::max(max_connected, connected);
        } else {
            connected = 0;
        }
    }
    enc[3] = max_connected / 5.0f;  // Straight potential

    // Pairs, trips, etc.
    int pairs = 0, trips = 0, quads = 0;
    for (int c : rank_counts) {
        if (c == 2) pairs++;
        else if (c == 3) trips++;
        else if (c == 4) quads++;
    }
    enc[4] = pairs > 0 ? 1.0f : 0.0f;
    enc[5] = trips > 0 ? 1.0f : 0.0f;
    enc[6] = quads > 0 ? 1.0f : 0.0f;

    // High card presence
    enc[7] = (rank_bits & (1 << 12)) ? 1.0f : 0.0f;  // Ace
    enc[8] = (rank_bits & (1 << 11)) ? 1.0f : 0.0f;  // King
    enc[9] = (rank_bits & (1 << 10)) ? 1.0f : 0.0f;  // Queen

    return enc;
}

std::vector<float> PBSEncoder::encode_position(int current_player, Street street) {
    return {
        static_cast<float>(current_player),
        street == Street::Preflop ? 1.0f : 0.0f,
        street == Street::Flop ? 1.0f : 0.0f,
        street == Street::Turn || street == Street::River ? 1.0f : 0.0f
    };
}

// =============================================================================
// Range Utilities Implementation
// =============================================================================

namespace range_utils {

std::pair<int, int> combo_to_cards(int combo_index) {
    // Convert combo index (0-1325) to two card indices
    // Uses triangular indexing: combo = c1 * (c1 - 1) / 2 + c2
    // where c1 > c2

    int c1 = 1;
    while (c1 * (c1 - 1) / 2 <= combo_index) {
        c1++;
    }
    c1--;

    int c2 = combo_index - c1 * (c1 - 1) / 2;
    return {c1, c2};
}

int cards_to_combo(int card1, int card2) {
    // Ensure card1 > card2
    if (card1 < card2) std::swap(card1, card2);
    return card1 * (card1 - 1) / 2 + card2;
}

std::vector<int> unblocked_combos(HandMask dead_cards) {
    std::vector<int> combos;
    combos.reserve(NUM_HOLE_COMBOS);

    for (int i = 0; i < NUM_HOLE_COMBOS; ++i) {
        auto [c1, c2] = combo_to_cards(i);
        HandMask hand = (1ULL << c1) | (1ULL << c2);
        if ((hand & dead_cards) == 0) {
            combos.push_back(i);
        }
    }

    return combos;
}

float range_vs_range_equity(
    const HandDistribution& range1,
    const HandDistribution& range2,
    const std::array<Card, 5>& board,
    int board_count
) {
    if (board_count < 5) {
        // Incomplete board - would need to enumerate runouts
        // For now, return 0.5 (equal equity)
        return 0.5f;
    }

    HandEvaluator::initialize();

    float total_weight = 0.0f;
    float total_equity = 0.0f;

    // Get dead cards from board
    HandMask dead = 0;
    for (int i = 0; i < board_count; ++i) {
        dead = add_card_to_hand(dead, board[i]);
    }

    // Enumerate all hand pairs
    for (int h1 = 0; h1 < NUM_HOLE_COMBOS; ++h1) {
        float p1 = range1.probs()[h1];
        if (p1 < 1e-10f) continue;

        auto [c1a, c1b] = combo_to_cards(h1);
        HandMask hand1_mask = (1ULL << c1a) | (1ULL << c1b);
        if (hand1_mask & dead) continue;

        HoleCards hand1(index_to_card(c1a), index_to_card(c1b));

        for (int h2 = 0; h2 < NUM_HOLE_COMBOS; ++h2) {
            float p2 = range2.probs()[h2];
            if (p2 < 1e-10f) continue;

            auto [c2a, c2b] = combo_to_cards(h2);
            HandMask hand2_mask = (1ULL << c2a) | (1ULL << c2b);

            // Check for card conflicts
            if ((hand2_mask & dead) || (hand2_mask & hand1_mask)) continue;

            HoleCards hand2(index_to_card(c2a), index_to_card(c2b));

            float weight = p1 * p2;
            total_weight += weight;

            // Evaluate hands
            auto rank1 = HandEvaluator::evaluate(hand1, board);
            auto rank2 = HandEvaluator::evaluate(hand2, board);

            if (rank1 < rank2) {
                total_equity += weight;
            } else if (rank1 == rank2) {
                total_equity += weight * 0.5f;
            }
        }
    }

    return (total_weight > 0) ? total_equity / total_weight : 0.5f;
}

}  // namespace range_utils

}  // namespace ares
