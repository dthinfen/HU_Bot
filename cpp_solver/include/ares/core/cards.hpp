#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ares {

// Card representation
// Bits 0-3: Rank (0=2, 1=3, ..., 12=A)
// Bits 4-5: Suit (0=clubs, 1=diamonds, 2=hearts, 3=spades)
using Card = uint8_t;

// Special value for no card
constexpr Card NO_CARD = 0xFF;

// Ranks (0-indexed from 2)
enum class Rank : uint8_t {
    Two = 0, Three, Four, Five, Six, Seven, Eight,
    Nine, Ten, Jack, Queen, King, Ace
};

// Suits
enum class Suit : uint8_t {
    Clubs = 0, Diamonds, Hearts, Spades
};

// Constants
constexpr int NUM_RANKS = 13;
constexpr int NUM_SUITS = 4;
constexpr int NUM_CARDS = 52;
constexpr int NUM_HOLE_CARD_COMBOS = 1326;  // 52 choose 2

// Card construction
constexpr Card make_card(Rank rank, Suit suit) {
    return static_cast<Card>((static_cast<uint8_t>(suit) << 4) |
                             static_cast<uint8_t>(rank));
}

constexpr Card make_card(int rank, int suit) {
    return static_cast<Card>((suit << 4) | rank);
}

// Card accessors
constexpr Rank get_rank(Card c) {
    return static_cast<Rank>(c & 0x0F);
}

constexpr Suit get_suit(Card c) {
    return static_cast<Suit>((c >> 4) & 0x03);
}

constexpr int get_rank_int(Card c) {
    return c & 0x0F;
}

constexpr int get_suit_int(Card c) {
    return (c >> 4) & 0x03;
}

// Convert card to 0-51 index
constexpr int card_to_index(Card c) {
    return get_suit_int(c) * 13 + get_rank_int(c);
}

// Convert 0-51 index to card
constexpr Card index_to_card(int idx) {
    return make_card(idx % 13, idx / 13);
}

// String conversion
inline std::string card_to_string(Card c) {
    if (c == NO_CARD) return "??";
    constexpr std::string_view ranks = "23456789TJQKA";
    constexpr std::string_view suits = "cdhs";
    return std::string{ranks[get_rank_int(c)], suits[get_suit_int(c)]};
}

inline Card string_to_card(std::string_view s) {
    if (s.size() != 2) return NO_CARD;

    constexpr std::string_view ranks = "23456789TJQKA";
    constexpr std::string_view suits = "cdhs";

    auto rank_pos = ranks.find(s[0]);
    auto suit_pos = suits.find(s[1]);

    if (rank_pos == std::string_view::npos ||
        suit_pos == std::string_view::npos) {
        return NO_CARD;
    }

    return make_card(static_cast<int>(rank_pos), static_cast<int>(suit_pos));
}

// Hand as a bitmask (64 bits, using 52)
using HandMask = uint64_t;

constexpr HandMask card_to_mask(Card c) {
    return 1ULL << card_to_index(c);
}

constexpr bool hand_contains(HandMask hand, Card c) {
    return (hand & card_to_mask(c)) != 0;
}

constexpr HandMask add_card_to_hand(HandMask hand, Card c) {
    return hand | card_to_mask(c);
}

constexpr int count_cards(HandMask hand) {
    return __builtin_popcountll(hand);
}

// Hole cards representation
struct HoleCards {
    Card cards[2];

    HoleCards() : cards{NO_CARD, NO_CARD} {}
    HoleCards(Card c1, Card c2) : cards{c1, c2} {
        // Normalize: higher card first
        if (card_to_index(cards[0]) < card_to_index(cards[1])) {
            std::swap(cards[0], cards[1]);
        }
    }

    Card operator[](int i) const { return cards[i]; }

    // Get canonical index (0 to 1325)
    int canonical_index() const {
        int idx1 = card_to_index(cards[0]);
        int idx2 = card_to_index(cards[1]);
        // Combinatorial number system
        return idx1 * (idx1 - 1) / 2 + idx2;
    }

    HandMask to_mask() const {
        return card_to_mask(cards[0]) | card_to_mask(cards[1]);
    }

    std::string to_string() const {
        return card_to_string(cards[0]) + card_to_string(cards[1]);
    }

    bool operator==(const HoleCards& other) const {
        return cards[0] == other.cards[0] && cards[1] == other.cards[1];
    }
};

// Deck for dealing
class Deck {
public:
    Deck() { reset(); }

    void reset() {
        for (int i = 0; i < NUM_CARDS; ++i) {
            cards_[i] = index_to_card(i);
        }
        top_ = 0;
        used_mask_ = 0;
    }

    // Shuffle using Fisher-Yates
    template<typename RNG>
    void shuffle(RNG& rng) {
        for (int i = NUM_CARDS - 1; i > 0; --i) {
            int j = rng() % (i + 1);
            std::swap(cards_[i], cards_[j]);
        }
        top_ = 0;
    }

    Card deal() {
        if (top_ >= NUM_CARDS) return NO_CARD;
        Card c = cards_[top_++];
        used_mask_ |= card_to_mask(c);
        return c;
    }

    // Deal a specific card (for setting up known hands)
    bool deal_specific(Card c) {
        if (hand_contains(used_mask_, c)) return false;
        used_mask_ |= card_to_mask(c);
        return true;
    }

    // Remove cards from deck (for dealing around known cards)
    void remove(HandMask cards) {
        used_mask_ |= cards;
    }

    // Deal random card not in used set
    template<typename RNG>
    Card deal_random(RNG& rng) {
        int remaining = NUM_CARDS - count_cards(used_mask_);
        if (remaining <= 0) return NO_CARD;

        int target = rng() % remaining;
        int count = 0;
        for (int i = 0; i < NUM_CARDS; ++i) {
            if (!hand_contains(used_mask_, index_to_card(i))) {
                if (count == target) {
                    Card c = index_to_card(i);
                    used_mask_ |= card_to_mask(c);
                    return c;
                }
                ++count;
            }
        }
        return NO_CARD;
    }

    HandMask used_cards() const { return used_mask_; }
    int remaining() const { return NUM_CARDS - count_cards(used_mask_); }

private:
    std::array<Card, NUM_CARDS> cards_;
    int top_;
    HandMask used_mask_;
};

}  // namespace ares
