#pragma once

#include <cstdint>
#include <array>
#include <random>
#include <algorithm>
#include <string>

namespace rl_env {

// Card representation: 0-51
// rank = card / 4 (0=2, 1=3, ..., 12=A)
// suit = card % 4 (0=clubs, 1=diamonds, 2=hearts, 3=spades)
using Card = uint8_t;
constexpr Card NO_CARD = 255;

inline int card_rank(Card c) { return c / 4; }
inline int card_suit(Card c) { return c % 4; }
inline Card make_card(int rank, int suit) { return rank * 4 + suit; }

inline std::string card_to_string(Card c) {
    if (c == NO_CARD) return "??";
    const char* ranks = "23456789TJQKA";
    const char* suits = "cdhs";
    return std::string(1, ranks[card_rank(c)]) + suits[card_suit(c)];
}

// Simple deck class
class Deck {
public:
    Deck() { reset(); }

    void reset() {
        for (int i = 0; i < 52; i++) {
            cards_[i] = static_cast<Card>(i);
        }
        pos_ = 0;
    }

    void shuffle(std::mt19937& rng) {
        std::shuffle(cards_.begin(), cards_.end(), rng);
        pos_ = 0;
    }

    Card deal() {
        return cards_[pos_++];
    }

    std::array<Card, 2> deal_hole() {
        return {deal(), deal()};
    }

    std::array<Card, 3> deal_flop() {
        return {deal(), deal(), deal()};
    }

private:
    std::array<Card, 52> cards_;
    int pos_ = 0;
};

// Hand evaluation using lookup tables (simplified 7-card eval)
class HandEvaluator {
public:
    // Returns rank where LOWER is BETTER (like treys)
    static uint16_t evaluate_7cards(const Card* cards);

    // Evaluate with hole cards + board
    static uint16_t evaluate(Card h1, Card h2, const Card* board, int board_size);

    // Initialize lookup tables
    static void initialize();

private:
    static bool initialized_;
    static std::array<uint16_t, 7462> flush_lookup_;
    static std::array<uint16_t, 7462> unique5_lookup_;
};

}  // namespace rl_env
