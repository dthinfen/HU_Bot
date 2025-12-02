#!/usr/bin/env python3
"""
ARES-HU Slumbot Benchmark

Play against Slumbot (2018 ACPC Champion) to validate our solver.
This is the gold standard benchmark for HUNL poker AI.

Slumbot specs:
- Blinds: 50/100 (0.5bb/1bb)
- Stack: 200 BB (20,000 chips)
- Position: client_pos 0 = BB, 1 = SB

Usage:
    python benchmark_slumbot.py --hands 1000
    python benchmark_slumbot.py --hands 1000 --username USER --password PASS  # For leaderboard

Reference win rates against Slumbot:
- ReBeL: +45 mbb/hand
- Ruse: +194 mbb/hand (19.4 bb/100)
- DecisionHoldem: +730 mbb/hand
- Random/Call-station: ~-500 mbb/hand
"""

import argparse
import urllib.request
import json as json_lib
import sys
import os
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add solver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp_solver', 'build'))

# Add alphaholdem to path
sys.path.insert(0, os.path.dirname(__file__))

# Also ensure we can import ares_solver
try:
    import ares_solver
except ImportError:
    pass  # Will be handled by agent

HOST = 'slumbot.com'
NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000  # 200 BB


@dataclass
class GameState:
    """Current game state parsed from Slumbot action string."""
    street: int  # 0=preflop, 1=flop, 2=turn, 3=river
    position: int  # 0=BB, 1=SB, -1=hand over
    pot: int
    to_call: int
    my_stack: int
    opp_stack: int
    my_bet_this_street: int
    opp_bet_this_street: int
    last_bet_size: int
    can_raise: bool


def parse_action(action: str, client_pos: int = 0) -> GameState:
    """
    Parse Slumbot action string into GameState with proper pot/stack tracking.

    Slumbot format:
    - 'k' = check
    - 'c' = call
    - 'f' = fold
    - 'b###' = bet/raise to ### chips (total this street)
    - '/' = street separator

    client_pos: 0 = we are BB, 1 = we are SB/BTN
    """
    # Player 0 = SB (button), Player 1 = BB
    # client_pos 0 = BB = player 1
    # client_pos 1 = SB = player 0

    # Track chips invested by each player (total across all streets)
    invested = [SMALL_BLIND, BIG_BLIND]  # [SB player, BB player]

    # Track bets this street
    street_bets = [0, 0]  # Reset each street

    # On preflop, blinds count as bets this street
    street_bets = [SMALL_BLIND, BIG_BLIND]

    street = 0
    current_player = 0  # SB acts first preflop
    last_bet_size = BIG_BLIND - SMALL_BLIND  # Initial "raise" is BB over SB
    last_aggressor = 1  # BB is initial aggressor
    check_or_call_ends_street = False
    hand_over = False

    i = 0
    while i < len(action) and not hand_over:
        c = action[i]
        i += 1

        if c == 'k':  # Check
            if check_or_call_ends_street:
                # Street ends - but don't increment here, let '/' handle it
                # Just reset for potential next action
                current_player = 1 - current_player
                check_or_call_ends_street = False
                last_bet_size = 0
            else:
                current_player = 1 - current_player
                check_or_call_ends_street = True

        elif c == 'c':  # Call
            # Caller matches opponent's bet
            call_amount = street_bets[1 - current_player] - street_bets[current_player]
            invested[current_player] += call_amount
            street_bets[current_player] = street_bets[1 - current_player]

            # Check if all-in call
            if invested[current_player] >= STACK_SIZE or invested[1 - current_player] >= STACK_SIZE:
                hand_over = True
            elif check_or_call_ends_street:
                # Street ends - but don't increment here, let '/' handle it
                # Just reset state for potential next action
                current_player = 1 - current_player
                check_or_call_ends_street = False
                last_bet_size = 0
            else:
                current_player = 1 - current_player
                check_or_call_ends_street = True

        elif c == 'f':  # Fold
            hand_over = True

        elif c == 'b':  # Bet/Raise
            # Parse bet amount
            j = i
            while i < len(action) and action[i].isdigit():
                i += 1
            bet_to = int(action[j:i])  # Total bet this street

            bet_increase = bet_to - street_bets[current_player]
            last_bet_size = bet_to - street_bets[1 - current_player]  # Raise size

            invested[current_player] += bet_increase
            street_bets[current_player] = bet_to
            last_aggressor = current_player

            current_player = 1 - current_player
            check_or_call_ends_street = True

        elif c == '/':  # Street separator
            street += 1
            street_bets = [0, 0]
            current_player = 1  # BB acts first postflop
            check_or_call_ends_street = False
            last_bet_size = 0

    # Calculate final state
    pot = invested[0] + invested[1]

    # Map to our perspective (client_pos)
    if client_pos == 0:
        # We are BB = player 1
        my_invested = invested[1]
        opp_invested = invested[0]
        my_street_bet = street_bets[1]
        opp_street_bet = street_bets[0]
        # Position: who acts next? Map current_player to our view
        # current_player 0 = SB = opponent, current_player 1 = BB = us
        acting_player = 0 if current_player == 1 else 1  # 0 = us (BB), 1 = opponent
    else:
        # We are SB = player 0
        my_invested = invested[0]
        opp_invested = invested[1]
        my_street_bet = street_bets[0]
        opp_street_bet = street_bets[1]
        # current_player 0 = SB = us, current_player 1 = BB = opponent
        acting_player = 1 if current_player == 0 else 0  # 1 = us (SB), 0 = opponent

    my_stack = STACK_SIZE - my_invested
    opp_stack = STACK_SIZE - opp_invested

    to_call = max(0, opp_street_bet - my_street_bet)

    # Position: -1 if hand over, else our client_pos
    position = -1 if hand_over else client_pos

    can_raise = max(invested) < STACK_SIZE

    return GameState(
        street=street,
        position=position,
        pot=pot,
        to_call=to_call,
        my_stack=my_stack,
        opp_stack=opp_stack,
        my_bet_this_street=my_street_bet,
        opp_bet_this_street=opp_street_bet,
        last_bet_size=last_bet_size,
        can_raise=can_raise
    )


class SlumbotClient:
    """Client for Slumbot API using urllib (avoids crash with C++ extension)."""

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.token = None
        if username and password:
            self.login(username, password)

    def _post(self, endpoint: str, data: Dict) -> Dict:
        """Make a POST request to Slumbot API."""
        url = f'https://{HOST}/slumbot/api/{endpoint}'
        json_data = json_lib.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            response = urllib.request.urlopen(req)
            return json_lib.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            raise Exception(f"API error: {e.code}")

    def login(self, username: str, password: str):
        """Login for leaderboard tracking."""
        data = {"username": username, "password": password}
        try:
            r = self._post('login', data)
            if 'error_msg' in r:
                print(f"Login error: {r['error_msg']}")
                return
            self.token = r.get('token')
            print(f"Logged in successfully")
        except Exception as e:
            print(f"Login failed: {e}")

    def new_hand(self) -> Dict:
        """Start a new hand."""
        data = {'token': self.token} if self.token else {}
        r = self._post('new_hand', data)
        if 'error_msg' in r:
            raise Exception(f"Slumbot error: {r['error_msg']}")
        if 'token' in r:
            self.token = r['token']
        return r

    def act(self, action: str) -> Dict:
        """Send an action to Slumbot."""
        data = {'token': self.token, 'incr': action}
        r = self._post('act', data)
        if 'error_msg' in r:
            raise Exception(f"Slumbot error: {r['error_msg']}")
        if 'token' in r:
            self.token = r['token']
        return r


class BaseAgent:
    """Base class for poker agents."""

    def get_action(
        self,
        hole_cards: List[str],
        board: List[str],
        action_history: str,
        client_pos: int,
        state: GameState
    ) -> str:
        """Return action string (k, c, f, or b[amount])."""
        raise NotImplementedError


class CallStationAgent(BaseAgent):
    """Always check or call - baseline for comparison."""

    def get_action(self, hole_cards, board, action_history, client_pos, state):
        if state.to_call > 0:
            return 'c'
        return 'k'


class RandomAgent(BaseAgent):
    """Random legal actions - another baseline."""

    def get_action(self, hole_cards, board, action_history, client_pos, state):
        actions = ['k' if state.to_call == 0 else 'c']
        if state.to_call > 0:
            actions.append('f')
        if state.can_raise:
            # Random bet size
            min_raise = max(BIG_BLIND, state.last_bet_size)
            max_raise = state.my_stack
            if max_raise >= min_raise:
                bet_size = random.randint(min_raise, max_raise)
                actions.append(f'b{bet_size + state.opp_bet_this_street}')
        return random.choice(actions)


class SimpleHeuristicAgent(BaseAgent):
    """
    Simple heuristic agent for 200bb deep stack play.
    Uses hand strength evaluation + pot odds + position.
    """

    def __init__(self):
        # High card rankings
        self.rank_values = {r: i for i, r in enumerate('23456789TJQKA')}

    def get_action(self, hole_cards, board, action_history, client_pos, state):
        hand_strength = self._evaluate_hand(hole_cards, board)
        pot_odds = state.to_call / (state.pot + state.to_call) if state.to_call > 0 else 0
        street = len(board) // 2  # 0=preflop, 1-2=flop, 3=turn, 4+=river

        # Get position (IP = in position = acting last)
        in_position = client_pos == 1  # Button position

        # Decision logic
        if state.to_call == 0:
            # Can check or bet
            if hand_strength > 0.7:
                # Strong hand - bet 75% pot
                bet_size = int(state.pot * 0.75)
                bet_size = max(bet_size, BIG_BLIND)
                return f'b{state.opp_bet_this_street + bet_size}'
            elif hand_strength > 0.5 and in_position:
                # Medium hand IP - bet 50% sometimes
                if random.random() < 0.5:
                    bet_size = int(state.pot * 0.5)
                    bet_size = max(bet_size, BIG_BLIND)
                    return f'b{state.opp_bet_this_street + bet_size}'
            return 'k'
        else:
            # Facing a bet - call, fold, or raise
            # Required equity = pot odds
            if hand_strength > 0.8:
                # Very strong - raise
                if state.can_raise:
                    raise_size = int(state.pot * 0.8)
                    total_bet = state.opp_bet_this_street + state.to_call + raise_size
                    return f'b{total_bet}'
                return 'c'
            elif hand_strength > pot_odds + 0.1:
                # Have enough equity to call
                return 'c'
            elif hand_strength > pot_odds and random.random() < 0.3:
                # Borderline - sometimes call as bluff catcher
                return 'c'
            else:
                return 'f'

    def _evaluate_hand(self, hole_cards: List[str], board: List[str]) -> float:
        """
        Estimate hand strength (0-1 scale).
        Simple heuristic - not true hand strength calculation.
        """
        if not hole_cards or len(hole_cards) < 2:
            return 0.5

        c1, c2 = hole_cards[0], hole_cards[1]
        r1 = self.rank_values.get(c1[0], 0)
        r2 = self.rank_values.get(c2[0], 0)
        suited = c1[1] == c2[1]
        paired = r1 == r2

        # Preflop hand strength
        high = max(r1, r2)
        low = min(r1, r2)
        gap = high - low

        preflop_strength = 0.0

        if paired:
            # Pairs: 22=0.5, AA=0.95
            preflop_strength = 0.5 + (r1 / 12) * 0.45
        else:
            # Unpaired: based on high card + low card + suited + connectedness
            preflop_strength = (high / 12) * 0.4 + (low / 12) * 0.2
            if suited:
                preflop_strength += 0.08
            if gap <= 3:
                preflop_strength += 0.05

        if not board:
            return preflop_strength

        # Postflop - check for made hands
        all_cards = hole_cards + board
        ranks = [self.rank_values.get(c[0], 0) for c in all_cards]
        suits = [c[1] for c in all_cards]

        # Check for pairs/trips/etc
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        max_count = max(rank_counts.values())
        pairs = sum(1 for c in rank_counts.values() if c >= 2)

        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suit = max(suit_counts.values())

        # Check for straights
        sorted_ranks = sorted(set(ranks))
        has_straight = False
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i+4] - sorted_ranks[i] == 4:
                has_straight = True
                break

        # Score hand
        if has_straight and max_suit >= 5:
            return 0.99  # Straight flush
        if max_count >= 4:
            return 0.97  # Quads
        if max_count >= 3 and pairs >= 2:
            return 0.95  # Full house
        if max_suit >= 5:
            return 0.90  # Flush
        if has_straight:
            return 0.85  # Straight
        if max_count >= 3:
            return 0.75  # Trips
        if pairs >= 2:
            return 0.65  # Two pair
        if pairs >= 1:
            # One pair - stronger if top pair
            pair_rank = max(r for r, c in rank_counts.items() if c >= 2)
            board_ranks = [self.rank_values.get(c[0], 0) for c in board]
            if board_ranks and pair_rank >= max(board_ranks):
                return 0.55 + (pair_rank / 12) * 0.1  # Top pair
            return 0.45 + (pair_rank / 12) * 0.1  # Lower pair
        else:
            # High card
            return 0.2 + (high / 12) * 0.15


class ARESSolverAgent(BaseAgent):
    """Agent using our CFR solver."""

    def __init__(self, strategy_path: str = 'blueprints/cpp_1M/strategy_1M.bin'):
        try:
            import ares_solver
            self.solver = ares_solver.Solver()
            self.solver.load(strategy_path)
            print(f"Loaded ARES solver with {self.solver.num_info_sets():,} info sets")
        except ImportError:
            print("WARNING: ares_solver not available, using random fallback")
            self.solver = None

    def get_action(self, hole_cards, board, action_history, client_pos, state):
        if self.solver is None:
            return RandomAgent().get_action(hole_cards, board, action_history, client_pos, state)

        try:
            # Convert to BB for our solver (Slumbot uses chips, 100 chips = 1bb)
            hero_stack = state.my_stack / 100.0
            villain_stack = state.opp_stack / 100.0
            pot = state.pot / 100.0

            # Get strategy from solver
            strategy = self.solver.get_strategy(
                hole_cards, board,
                hero_stack, villain_stack, pot
            )

            if not strategy or '_debug_key' in strategy:
                # Fallback to call/check if state not found
                return 'c' if state.to_call > 0 else 'k'

            # Sample action from strategy
            actions = []
            probs = []
            for action_name, prob in strategy.items():
                if action_name.startswith('_'):
                    continue
                actions.append(action_name)
                probs.append(prob)

            if not actions:
                return 'c' if state.to_call > 0 else 'k'

            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            # Sample
            r = random.random()
            cumsum = 0
            chosen = actions[0]
            for action, prob in zip(actions, probs):
                cumsum += prob
                if r < cumsum:
                    chosen = action
                    break

            # Convert action to Slumbot format
            return self._convert_action(chosen, state)

        except Exception as e:
            print(f"Solver error: {e}")
            return 'c' if state.to_call > 0 else 'k'

    def _convert_action(self, action: str, state: GameState) -> str:
        """Convert our action format to Slumbot format."""
        action = action.lower()

        if action == 'fold':
            return 'f'
        elif action in ('check', 'call'):
            return 'c' if state.to_call > 0 else 'k'
        elif action.startswith('allin'):
            return f'b{STACK_SIZE}'
        elif action.startswith(('bet_', 'raise_')):
            # Extract amount and convert
            try:
                amount_bb = float(action.split('_')[1])
                amount_chips = int(amount_bb * 100)
                # Slumbot wants total bet this street
                total_bet = state.opp_bet_this_street + amount_chips
                total_bet = max(total_bet, state.opp_bet_this_street + BIG_BLIND)
                total_bet = min(total_bet, STACK_SIZE)
                return f'b{total_bet}'
            except (ValueError, IndexError):
                return 'c' if state.to_call > 0 else 'k'
        else:
            return 'c' if state.to_call > 0 else 'k'


class RealtimeSearchAgent(BaseAgent):
    """
    Agent using C++ real-time CFR search.

    This is the KEY component that enables playing at any stack depth.
    For each decision, it runs CFR iterations from the current state.
    """

    def __init__(self, iterations: int = 500, qre_tau: float = 1.0, time_limit: float = 5.0,
                 neural_model_path: str = None):
        try:
            import ares_solver
            self.solver = ares_solver.Solver()
            self.has_solver = True

            # Try to load neural model for leaf evaluation
            if neural_model_path is None:
                # Default path
                neural_model_path = os.path.join(
                    os.path.dirname(__file__), 'models', 'value_net.torchscript'
                )

            if hasattr(self.solver, 'load_neural_model') and os.path.exists(neural_model_path):
                if self.solver.load_neural_model(neural_model_path):
                    print(f"  Neural model loaded: {neural_model_path}")
                else:
                    print("  Neural model failed to load, using heuristic evaluation")
            else:
                print("  Neural model not available, using heuristic evaluation")

        except ImportError:
            print("WARNING: ares_solver not available, using heuristic fallback")
            self.solver = None
            self.has_solver = False

        self.iterations = iterations
        self.qre_tau = qre_tau
        self.time_limit = time_limit

        print(f"C++ Realtime Search Agent: {iterations} iters, tau={qre_tau}, time_limit={time_limit}s")

    def get_action(self, hole_cards, board, action_history, client_pos, slumbot_state):
        if not self.has_solver:
            return SimpleHeuristicAgent().get_action(hole_cards, board, action_history, client_pos, slumbot_state)

        try:
            # Convert to bb (Slumbot uses chips, 100 chips = 1bb)
            pot_bb = slumbot_state.pot / 100.0
            my_stack_bb = slumbot_state.my_stack / 100.0
            opp_stack_bb = slumbot_state.opp_stack / 100.0
            my_bet_bb = slumbot_state.my_bet_this_street / 100.0
            opp_bet_bb = slumbot_state.opp_bet_this_street / 100.0

            # Street: 0=preflop, 1=flop, 2=turn, 3=river
            street = slumbot_state.street

            # Hero position: 0=OOP (BB), 1=IP (BTN)
            hero_position = 1 - client_pos  # Slumbot client_pos: 0=BB, 1=SB

            # Run C++ search
            result = self.solver.search(
                hero_cards=hole_cards,
                board_cards=board if board else [],
                pot=pot_bb,
                hero_stack=my_stack_bb,
                villain_stack=opp_stack_bb,
                hero_bet=my_bet_bb,
                villain_bet=opp_bet_bb,
                street=street,
                hero_position=hero_position,
                iterations=self.iterations,
                time_limit=self.time_limit,
                qre_tau=self.qre_tau
            )

            # Sample from strategy
            actions = result['actions']
            probs = result['strategy']

            if not actions:
                return 'c' if slumbot_state.to_call > 0 else 'k'

            # Sample action according to strategy
            r = random.random()
            cumsum = 0.0
            chosen_idx = 0
            for i, prob in enumerate(probs):
                cumsum += prob
                if r < cumsum:
                    chosen_idx = i
                    break

            chosen_action = actions[chosen_idx]
            return self._convert_action(chosen_action, slumbot_state)

        except Exception as e:
            print(f"Search error: {e}")
            return 'c' if slumbot_state.to_call > 0 else 'k'

    def _convert_action(self, action: str, state) -> str:
        """Convert C++ action string to Slumbot format."""
        action_lower = action.lower()

        if action_lower == 'fold':
            # Can only fold if facing a bet
            if state.to_call > 0:
                return 'f'
            else:
                return 'k'  # Check instead
        elif action_lower == 'check':
            # Can only check if not facing a bet
            if state.to_call == 0:
                return 'k'
            else:
                return 'c'  # Call instead
        elif action_lower == 'call':
            # Can only call if facing a bet
            if state.to_call > 0:
                return 'c'
            else:
                return 'k'  # Check instead
        elif action_lower.startswith('allin'):
            # All-in: extract amount from "allin_X.XXXX"
            try:
                allin_bb = float(action_lower.split('_')[1])
                allin_chips = int(allin_bb * 100)
                # Cap at our remaining stack + current bet this street
                max_bet = state.my_stack + state.my_bet_this_street
                allin_chips = min(allin_chips, max_bet)
                # Also cap at total stack size
                allin_chips = min(allin_chips, STACK_SIZE)
                # If our all-in doesn't raise the bet, it's just a call
                if allin_chips <= state.opp_bet_this_street:
                    return 'c' if state.to_call > 0 else 'k'
                return f'b{allin_chips}'
            except (ValueError, IndexError):
                # Fallback to actual stack
                actual_allin = state.my_stack + state.my_bet_this_street
                if actual_allin <= state.opp_bet_this_street:
                    return 'c' if state.to_call > 0 else 'k'
                return f'b{actual_allin}'
        elif action_lower.startswith(('bet_', 'raise_')):
            # C++ solver returns "bet_X" or "raise_X" where X is the TOTAL bet amount in bb
            # Slumbot wants the total bet this street in chips
            try:
                total_bet_bb = float(action_lower.split('_')[1])
                total_bet_chips = int(total_bet_bb * 100)

                # Minimum legal raise = opponent's bet + max(last_bet_size, BIG_BLIND)
                min_raise_size = max(state.last_bet_size, BIG_BLIND)
                min_total_bet = state.opp_bet_this_street + min_raise_size

                # Ensure we're at least at minimum
                total_bet_chips = max(total_bet_chips, min_total_bet)

                # Maximum bet = our stack + what we've already bet this street
                # Be conservative: use the lower of solver amount or stack-based max
                max_bet = state.my_stack + state.my_bet_this_street

                # Also can't bet more than what opponent can call
                # (effective stack logic)
                opp_can_call = state.opp_stack + state.opp_bet_this_street
                max_bet = min(max_bet, opp_can_call)

                total_bet_chips = min(total_bet_chips, max_bet)

                # Final sanity check: raise must be STRICTLY greater than opponent's bet
                if total_bet_chips <= state.opp_bet_this_street:
                    # Can't make a valid raise - just call
                    return 'c' if state.to_call > 0 else 'k'

                if total_bet_chips < min_total_bet:
                    # Can't meet minimum raise - just call
                    return 'c' if state.to_call > 0 else 'k'

                if total_bet_chips > STACK_SIZE:
                    total_bet_chips = STACK_SIZE

                return f'b{total_bet_chips}'
            except (ValueError, IndexError):
                return 'c' if state.to_call > 0 else 'k'
        else:
            return 'c' if state.to_call > 0 else 'k'


class NeuralNetworkAgent(BaseAgent):
    """
    Agent using trained AlphaHoldem-style neural network checkpoint.

    Loads an ActorCritic model from a training checkpoint and uses it
    to play against Slumbot. Supports both old (38-channel) and new (50-channel)
    checkpoint formats.
    """

    # Action space mapping (matching C++ FastEnv action_to_index):
    # 0: Fold
    # 1: Check/Call
    # 2: <0.4x pot  -> use 0.3x
    # 3: 0.4-0.6x   -> use 0.5x
    # 4: 0.6-0.8x   -> use 0.7x
    # 5: 0.8-1.0x   -> use 0.9x
    # 6: 1.0-1.5x   -> use 1.25x
    # 7: 1.5-2.0x   -> use 1.75x
    # 8: 2.0-3.0x   -> use 2.5x
    # 9-12: 3.0+x   -> use increasing sizes
    # 13: All-in
    NUM_ACTIONS = 14
    BET_SIZES = [0.3, 0.5, 0.7, 0.9, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5]
    ACTION_NAMES = ['Fold', 'Check/Call', 'Bet 0.3x', 'Bet 0.5x', 'Bet 0.7x', 'Bet 0.9x',
                    'Bet 1.25x', 'Bet 1.75x', 'Bet 2.5x', 'Bet 3.5x', 'Bet 4.5x',
                    'Bet 5.5x', 'Bet 6.5x', 'All-in']

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        try:
            import torch
            import torch.nn.functional as F

            self.torch = torch
            self.F = F

            # Load checkpoint first to detect architecture
            self.device = device
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict']

            # Handle compiled model prefix
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            # Detect architecture from checkpoint
            input_conv_shape = state_dict['backbone.input_conv.weight'].shape
            self.input_channels = input_conv_shape[1]

            # Detect head structure (old vs new)
            has_old_heads = 'policy_head.fc1.weight' in state_dict
            has_new_heads = 'policy_head.fc_layers.0.weight' in state_dict

            print(f"Detected architecture: {self.input_channels} input channels")

            if has_old_heads:
                print("  Using legacy head structure (fc1/fc2)")
                # Use legacy model definition
                self.model = self._create_legacy_model(
                    input_channels=self.input_channels,
                    hidden_dim=checkpoint.get('config', {}).get('hidden_dim', 256)
                ).to(device)
            elif has_new_heads:
                print("  Using new head structure (fc_layers)")
                from alphaholdem.src.network import ActorCritic

                # Get config from checkpoint
                config = checkpoint.get('config', {})
                fc_hidden_dim = config.get('fc_hidden_dim', checkpoint.get('fc_hidden_dim', 1024))
                fc_num_layers = config.get('fc_num_layers', checkpoint.get('fc_num_layers', 3))

                print(f"  FC hidden dim: {fc_hidden_dim}, FC layers: {fc_num_layers}")

                self.model = ActorCritic(
                    input_channels=self.input_channels,
                    use_cnn=True,
                    num_actions=self.NUM_ACTIONS,
                    fc_hidden_dim=fc_hidden_dim,
                    fc_num_layers=fc_num_layers
                ).to(device)
            else:
                raise ValueError("Unknown model architecture in checkpoint")

            # Load weights
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Get update count for info
            update_count = checkpoint.get('update_count', 0)
            print(f"Neural Network Agent loaded: {checkpoint_path}")
            print(f"  Update count: {update_count}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"  Device: {device}")

            # Action distribution tracking
            self.action_counts = [0] * self.NUM_ACTIONS
            self.action_by_street = [[0] * self.NUM_ACTIONS for _ in range(4)]  # preflop, flop, turn, river
            self.total_decisions = 0

        except Exception as e:
            print(f"ERROR loading checkpoint: {e}")
            raise

    def _create_legacy_model(self, input_channels: int, hidden_dim: int):
        """Create model with legacy head structure (fc1/fc2)."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from alphaholdem.src.network import CNNBackbone

        class LegacyPolicyHead(nn.Module):
            """Legacy policy head with fc1/fc2."""
            def __init__(self, hidden_dim, num_actions):
                super().__init__()
                self.fc1 = nn.Linear(hidden_dim, 128)
                self.fc2 = nn.Linear(128, num_actions)

            def forward(self, x, action_mask=None):
                x = F.relu(self.fc1(x))
                logits = self.fc2(x)
                if action_mask is not None:
                    logits = logits.masked_fill(~action_mask.bool(), float('-inf'))
                return logits

        class LegacyValueHead(nn.Module):
            """Legacy value head with fc1/fc2."""
            def __init__(self, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(hidden_dim, 128)
                self.fc2 = nn.Linear(128, 1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                return self.fc2(x).squeeze(-1)

        class LegacyActorCritic(nn.Module):
            """Legacy model structure for old checkpoints."""

            def __init__(self, input_channels, hidden_dim, num_actions):
                super().__init__()
                self.backbone = CNNBackbone(
                    input_channels=input_channels,
                    hidden_channels=128,
                    num_residual_blocks=4,
                    output_dim=hidden_dim
                )
                self.policy_head = LegacyPolicyHead(hidden_dim, num_actions)
                self.value_head = LegacyValueHead(hidden_dim)
                self.num_actions = num_actions

            def forward(self, x, action_mask=None):
                features = self.backbone(x)
                logits = self.policy_head(features, action_mask)
                value = self.value_head(features)
                return logits, value

        return LegacyActorCritic(input_channels, hidden_dim, self.NUM_ACTIONS)

    def get_action(self, hole_cards, board, action_history, client_pos, state):
        """Get action from neural network."""
        # Encode observation (use appropriate encoding based on detected architecture)
        if self.input_channels == 38:
            obs = self._encode_observation_38ch(hole_cards, board, action_history, client_pos, state)
        else:
            obs = self._encode_observation(hole_cards, board, action_history, client_pos, state)
        obs_tensor = self.torch.tensor(obs, dtype=self.torch.float32, device=self.device).unsqueeze(0)

        # Get action mask
        action_mask = self._get_action_mask(state)
        mask_tensor = self.torch.tensor(action_mask, dtype=self.torch.bool, device=self.device).unsqueeze(0)

        # Forward pass
        with self.torch.no_grad():
            logits, _ = self.model(obs_tensor, mask_tensor)
            probs = self.F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Sample action (with small exploration noise for robustness)
        probs = probs * 0.95 + 0.05 / self.NUM_ACTIONS
        probs = probs / probs.sum()

        # Mask invalid actions
        probs = probs * action_mask
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback
            return 'c' if state.to_call > 0 else 'k'

        action_idx = random.choices(range(self.NUM_ACTIONS), weights=probs)[0]

        # Track action distribution
        self.action_counts[action_idx] += 1
        self.action_by_street[state.street][action_idx] += 1
        self.total_decisions += 1

        return self._convert_action_idx(action_idx, state)

    def _encode_observation(self, hole_cards, board, action_history, client_pos, state) -> 'np.ndarray':
        """
        Encode game state as 50x4x13 tensor (matching C++ FastEnv encoding).
        """
        import numpy as np

        obs = np.zeros((50, 4, 13), dtype=np.float32)

        # Parse cards
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                    '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

        def parse_card(c):
            """Parse card string like 'As' -> (rank=12, suit=0)"""
            return rank_map.get(c[0].upper(), 0), suit_map.get(c[1].lower(), 0)

        all_cards = []

        # [0-1] Hole cards
        for i, card in enumerate(hole_cards[:2]):
            rank, suit = parse_card(card)
            obs[i, suit, rank] = 1.0
            all_cards.append((rank, suit))

        # [2-6] Board cards
        for i, card in enumerate(board[:5]):
            rank, suit = parse_card(card)
            obs[2 + i, suit, rank] = 1.0
            all_cards.append((rank, suit))

        # [7] All known cards combined
        for rank, suit in all_cards:
            obs[7, suit, rank] = 1.0

        # [8-11] Suit counts
        suit_counts = [0, 0, 0, 0]
        for rank, suit in all_cards:
            suit_counts[suit] += 1
        for s in range(4):
            obs[8 + s, :, :] = suit_counts[s] / 7.0

        # [12-15] Rank counts (pair/trips/quads indicators)
        rank_counts = [0] * 13
        for rank, suit in all_cards:
            rank_counts[rank] += 1
        for r in range(13):
            count = rank_counts[r]
            if count >= 1:
                obs[12, :, r] = 1.0
            if count >= 2:
                obs[13, :, r] = 1.0
            if count >= 3:
                obs[14, :, r] = 1.0
            if count >= 4:
                obs[15, :, r] = 1.0

        # [16-19] Street indicators
        obs[16 + state.street, :, :] = 1.0

        # [20-21] Position (hero is BTN or BB)
        # client_pos: 0 = BB, 1 = SB/BTN
        is_button = (client_pos == 1)
        if is_button:
            obs[20, :, :] = 1.0
        else:
            obs[21, :, :] = 1.0

        # Convert from chips to BB (Slumbot uses 100 chips = 1bb)
        pot_bb = state.pot / 100.0
        to_call_bb = state.to_call / 100.0
        my_stack_bb = state.my_stack / 100.0
        opp_stack_bb = state.opp_stack / 100.0
        starting_stack_bb = 200.0  # Slumbot uses 200bb stacks, same as training

        # [22] To-call amount (normalized)
        pot_for_odds = max(pot_bb, 1.0)
        obs[22, :, :] = min(to_call_bb / pot_for_odds, 2.0) / 2.0

        # [23] Facing all-in indicator
        facing_allin = (opp_stack_bb < 0.01 and to_call_bb > 0)
        obs[23, :, :] = 1.0 if facing_allin else 0.0

        # [24-27] Stack/pot info (normalized by starting stack in BB)
        obs[24, :, :] = my_stack_bb / starting_stack_bb
        obs[25, :, :] = opp_stack_bb / starting_stack_bb

        # SPR (Stack-to-Pot Ratio)
        effective_stack_bb = min(my_stack_bb, opp_stack_bb)
        spr = effective_stack_bb / max(pot_bb, 1.0)
        obs[26, :, :] = min(spr / 10.0, 1.0)

        # Pot as fraction of total chips
        total_bb = pot_bb + my_stack_bb + opp_stack_bb
        if total_bb > 0:
            obs[27, :, :] = pot_bb / total_bb

        # [28-49] Action history (simplified - just encode recent actions)
        # Parse action string and encode last 22 actions
        parsed_actions = self._parse_action_history(action_history)
        for i, (action_type, amount, player, street_num) in enumerate(parsed_actions[-22:]):
            channel = 28 + i
            if channel >= 50:
                break

            # Row 0: action type
            if action_type < 13:
                obs[channel, 0, action_type] = 1.0

            # Row 1: amount normalized (convert chips to BB)
            amount_bb = amount / 100.0
            obs[channel, 1, :] = min(amount_bb / starting_stack_bb, 1.0)

            # Row 2: player indicator
            # Convert to hero perspective
            is_hero = (player == client_pos)
            obs[channel, 2, :] = 1.0 if is_hero else 0.0

            # Row 3: street
            obs[channel, 3, :] = street_num / 3.0

        return obs

    def _encode_observation_38ch(self, hole_cards, board, action_history, client_pos, state) -> 'np.ndarray':
        """
        Encode game state as 38x4x13 tensor (legacy format for older checkpoints).

        Channel layout:
        [0-1]   : Hole cards (2 channels)
        [2-6]   : Board cards (5 channels)
        [7]     : All known cards combined
        [8-11]  : Suit counts
        [12-15] : Rank counts
        [16-19] : Street indicators
        [20-21] : Position indicators
        [22]    : To-call amount
        [23]    : Facing all-in
        [24-27] : Pot/stack ratios
        [28-37] : Betting history (10 actions)
        """
        import numpy as np

        obs = np.zeros((38, 4, 13), dtype=np.float32)

        # Parse cards
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                    '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

        def parse_card(c):
            return rank_map.get(c[0].upper(), 0), suit_map.get(c[1].lower(), 0)

        all_cards = []

        # [0-1] Hole cards
        for i, card in enumerate(hole_cards[:2]):
            rank, suit = parse_card(card)
            obs[i, suit, rank] = 1.0
            all_cards.append((rank, suit))

        # [2-6] Board cards
        for i, card in enumerate(board[:5]):
            rank, suit = parse_card(card)
            obs[2 + i, suit, rank] = 1.0
            all_cards.append((rank, suit))

        # [7] All known cards combined
        for rank, suit in all_cards:
            obs[7, suit, rank] = 1.0

        # [8-11] Suit counts
        suit_counts = [0, 0, 0, 0]
        for rank, suit in all_cards:
            suit_counts[suit] += 1
        for s in range(4):
            obs[8 + s, :, :] = suit_counts[s] / 7.0

        # [12-15] Rank counts
        rank_counts = [0] * 13
        for rank, suit in all_cards:
            rank_counts[rank] += 1
        for r in range(13):
            count = rank_counts[r]
            if count >= 1:
                obs[12, :, r] = 1.0
            if count >= 2:
                obs[13, :, r] = 1.0
            if count >= 3:
                obs[14, :, r] = 1.0
            if count >= 4:
                obs[15, :, r] = 1.0

        # [16-19] Street indicators
        obs[16 + state.street, :, :] = 1.0

        # [20-21] Position
        is_button = (client_pos == 1)
        if is_button:
            obs[20, :, :] = 1.0
        else:
            obs[21, :, :] = 1.0

        # [22] To-call amount
        pot = max(state.pot, 1.0)
        obs[22, :, :] = min(state.to_call / pot, 2.0) / 2.0

        # [23] Facing all-in
        facing_allin = (state.opp_stack < 1 and state.to_call > 0)
        obs[23, :, :] = 1.0 if facing_allin else 0.0

        # [24-27] Stack/pot info
        starting_stack = STACK_SIZE
        obs[24, :, :] = state.my_stack / starting_stack
        obs[25, :, :] = state.opp_stack / starting_stack

        effective_stack = min(state.my_stack, state.opp_stack)
        spr = effective_stack / max(pot, 1.0)
        obs[26, :, :] = min(spr / 10.0, 1.0)

        total_chips = state.pot + state.my_stack + state.opp_stack
        if total_chips > 0:
            obs[27, :, :] = state.pot / total_chips

        # [28-37] Betting history (10 actions for legacy format)
        parsed_actions = self._parse_action_history(action_history)
        for i, (action_type, amount, player, street_num) in enumerate(parsed_actions[-10:]):
            channel = 28 + i
            if channel >= 38:
                break
            if action_type < 13:
                obs[channel, 0, action_type] = 1.0
            obs[channel, 1, :] = min(amount / starting_stack, 1.0)
            is_hero = (player == client_pos)
            obs[channel, 2, :] = 1.0 if is_hero else 0.0

        return obs

    def _parse_action_history(self, action: str) -> List[Tuple[int, float, int, int]]:
        """Parse Slumbot action string to list of (action_type, amount, player, street)."""
        # Action types: 0=fold, 1=check, 2=call, 3=bet, 4=raise, 5=all-in
        actions = []
        street = 0
        current_player = 0  # SB acts first preflop

        i = 0
        while i < len(action):
            c = action[i]

            if c == '/':
                street += 1
                current_player = 1  # BB acts first postflop
                i += 1
            elif c == 'k':
                actions.append((1, 0.0, current_player, street))  # check
                current_player = 1 - current_player
                i += 1
            elif c == 'c':
                actions.append((2, 0.0, current_player, street))  # call
                current_player = 1 - current_player
                i += 1
            elif c == 'f':
                actions.append((0, 0.0, current_player, street))  # fold
                i += 1
            elif c == 'b':
                j = i + 1
                while j < len(action) and action[j].isdigit():
                    j += 1
                amount = float(action[i+1:j]) if j > i + 1 else 0.0
                # Check if this might be all-in
                action_type = 4 if actions else 3  # raise if there was a bet, else bet
                actions.append((action_type, amount, current_player, street))
                current_player = 1 - current_player
                i = j
            else:
                i += 1

        return actions

    def _get_action_mask(self, state) -> 'np.ndarray':
        """Get valid action mask."""
        import numpy as np

        mask = np.zeros(self.NUM_ACTIONS, dtype=bool)

        # Fold valid if facing a bet
        mask[0] = state.to_call > 0

        # Check/Call always valid
        mask[1] = True

        # Bet/raise sizes
        pot = state.pot
        for i, size in enumerate(self.BET_SIZES):
            bet_amount = int(pot * size)
            # Minimum raise check
            min_raise = max(state.last_bet_size, BIG_BLIND)
            if bet_amount >= min_raise and bet_amount <= state.my_stack:
                mask[2 + i] = True

        # All-in always valid if we have chips
        mask[13] = state.my_stack > 0

        return mask

    def _convert_action_idx(self, action_idx: int, state) -> str:
        """Convert action index to Slumbot format."""
        if action_idx == 0:  # Fold
            return 'f' if state.to_call > 0 else 'k'

        elif action_idx == 1:  # Check/Call
            return 'c' if state.to_call > 0 else 'k'

        elif action_idx == 13:  # All-in
            total_bet = state.my_stack + state.my_bet_this_street

            # If our all-in doesn't exceed opponent's bet, it's just a call
            if total_bet <= state.opp_bet_this_street:
                return 'c' if state.to_call > 0 else 'k'

            # If opponent is already all-in (can't raise), just call
            if not state.can_raise or state.opp_stack == 0:
                return 'c' if state.to_call > 0 else 'k'

            # Check minimum raise requirement
            min_raise_size = max(state.last_bet_size, BIG_BLIND)
            min_total_bet = state.opp_bet_this_street + min_raise_size

            # If we can't make a legal raise (not enough chips), just call
            if total_bet < min_total_bet:
                return 'c' if state.to_call > 0 else 'k'

            return f'b{int(total_bet)}'

        else:  # Bet/Raise (indices 2-12)
            # If opponent is all-in or we can't raise, just call/check
            if not state.can_raise or state.opp_stack == 0:
                return 'c' if state.to_call > 0 else 'k'

            size_idx = action_idx - 2
            if size_idx < len(self.BET_SIZES):
                pot_mult = self.BET_SIZES[size_idx]
                bet_amount = int(state.pot * pot_mult)

                # Minimum raise size (the INCREMENT, not total)
                min_raise_size = max(state.last_bet_size, BIG_BLIND)

                if state.to_call > 0:
                    # Raising - minimum total bet is opponent's bet + min raise size
                    min_total_bet = state.opp_bet_this_street + min_raise_size
                    # Our desired bet = opponent's bet + our raise amount
                    total_bet = state.opp_bet_this_street + max(bet_amount, min_raise_size)
                else:
                    # Betting (no prior bet) - minimum is BB
                    min_total_bet = BIG_BLIND
                    total_bet = max(bet_amount, BIG_BLIND)

                # Cap at our maximum (stack + what we've already put in)
                max_bet = state.my_stack + state.my_bet_this_street
                total_bet = min(total_bet, max_bet)

                # If we can't make a legal raise, just call/check
                if total_bet < min_total_bet:
                    return 'c' if state.to_call > 0 else 'k'

                # If our bet doesn't exceed opponent's, just call/check
                if total_bet <= state.opp_bet_this_street:
                    return 'c' if state.to_call > 0 else 'k'

                return f'b{int(total_bet)}'

            return 'c' if state.to_call > 0 else 'k'

    def print_action_distribution(self):
        """Print summary of action distribution."""
        if self.total_decisions == 0:
            print("No decisions recorded yet.")
            return

        print("\n" + "=" * 60)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("=" * 60)

        # Overall distribution
        print(f"\nTotal decisions: {self.total_decisions}")
        print("\nOverall Action Distribution:")
        print("-" * 40)

        for i, name in enumerate(self.ACTION_NAMES):
            count = self.action_counts[i]
            pct = count / self.total_decisions * 100
            bar = '#' * int(pct / 2)
            print(f"  {name:12s}: {count:4d} ({pct:5.1f}%) {bar}")

        # Group into categories
        fold_pct = self.action_counts[0] / self.total_decisions * 100
        check_call_pct = self.action_counts[1] / self.total_decisions * 100
        small_bet_pct = sum(self.action_counts[2:6]) / self.total_decisions * 100  # 0.3x-0.9x
        medium_bet_pct = sum(self.action_counts[6:9]) / self.total_decisions * 100  # 1.25x-2.5x
        large_bet_pct = sum(self.action_counts[9:13]) / self.total_decisions * 100  # 3.5x+
        allin_pct = self.action_counts[13] / self.total_decisions * 100

        print("\nAction Categories:")
        print("-" * 40)
        print(f"  Fold:           {fold_pct:5.1f}%")
        print(f"  Check/Call:     {check_call_pct:5.1f}%")
        print(f"  Small bet:      {small_bet_pct:5.1f}%  (0.3x-0.9x pot)")
        print(f"  Medium bet:     {medium_bet_pct:5.1f}%  (1.25x-2.5x pot)")
        print(f"  Large bet:      {large_bet_pct:5.1f}%  (3.5x+ pot)")
        print(f"  All-in:         {allin_pct:5.1f}%")

        # Per-street breakdown
        street_names = ['Preflop', 'Flop', 'Turn', 'River']
        print("\nPer-Street Breakdown:")
        print("-" * 40)

        for s, street_name in enumerate(street_names):
            street_total = sum(self.action_by_street[s])
            if street_total == 0:
                continue

            print(f"\n  {street_name} ({street_total} decisions):")
            fold = self.action_by_street[s][0] / street_total * 100
            chk_call = self.action_by_street[s][1] / street_total * 100
            bets = sum(self.action_by_street[s][2:13]) / street_total * 100
            allin = self.action_by_street[s][13] / street_total * 100
            print(f"    Fold: {fold:5.1f}%  Check/Call: {chk_call:5.1f}%  Bet/Raise: {bets:5.1f}%  All-in: {allin:5.1f}%")

        print("=" * 60)


def play_hand(client: SlumbotClient, agent: BaseAgent, verbose: bool = False) -> int:
    """Play a single hand against Slumbot. Returns winnings in chips."""
    r = client.new_hand()

    while True:
        action = r.get('action', '')
        client_pos = r.get('client_pos', 0)
        hole_cards = r.get('hole_cards', [])
        board = r.get('board', [])
        winnings = r.get('winnings')

        if verbose:
            print(f"  Action: {action}, Cards: {hole_cards}, Board: {board}")

        if winnings is not None:
            if verbose:
                print(f"  Result: {winnings:+d} chips ({winnings/100:+.2f} bb)")
            return winnings

        # Parse state and get our action
        state = parse_action(action, client_pos)

        if state.position == -1:
            # Hand is over
            continue

        our_action = agent.get_action(hole_cards, board, action, client_pos, state)

        if verbose:
            print(f"  Our action: {our_action}")
            print(f"    State: pot={state.pot}, to_call={state.to_call}, "
                  f"my_stack={state.my_stack}, opp_stack={state.opp_stack}, "
                  f"my_bet={state.my_bet_this_street}, opp_bet={state.opp_bet_this_street}, "
                  f"last_bet_size={state.last_bet_size}")

        try:
            r = client.act(our_action)
        except Exception as e:
            if "Illegal bet" in str(e):
                print(f"  ILLEGAL BET DEBUG:")
                print(f"    Action history: {action}")
                print(f"    Our action: {our_action}")
                print(f"    State: pot={state.pot}, to_call={state.to_call}")
                print(f"    my_stack={state.my_stack}, opp_stack={state.opp_stack}")
                print(f"    my_bet={state.my_bet_this_street}, opp_bet={state.opp_bet_this_street}")
                print(f"    last_bet_size={state.last_bet_size}, can_raise={state.can_raise}")
            raise


def run_benchmark(
    agent: BaseAgent,
    num_hands: int = 1000,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """Run benchmark against Slumbot."""
    client = SlumbotClient(username, password)

    total_winnings = 0
    hands_played = 0
    wins = 0
    losses = 0

    print(f"\nPlaying {num_hands} hands against Slumbot...")
    print("-" * 50)

    start_time = time.time()

    for i in range(num_hands):
        try:
            winnings = play_hand(client, agent, verbose)
            total_winnings += winnings
            hands_played += 1

            if winnings > 0:
                wins += 1
            elif winnings < 0:
                losses += 1

            if (i + 1) % 100 == 0:
                rate = total_winnings / hands_played / 100 * 1000  # mbb/hand
                elapsed = time.time() - start_time
                print(f"  Hands: {hands_played}, "
                      f"Winnings: {total_winnings/100:+.1f} bb, "
                      f"Rate: {rate:+.1f} mbb/hand, "
                      f"Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"Error on hand {i+1}: {e}")
            time.sleep(1)  # Rate limit on errors

    elapsed = time.time() - start_time

    # Calculate statistics
    bb_won = total_winnings / 100
    mbb_per_hand = total_winnings / hands_played / 100 * 1000
    bb_per_100 = bb_won / hands_played * 100

    results = {
        'hands_played': hands_played,
        'total_bb_won': bb_won,
        'mbb_per_hand': mbb_per_hand,
        'bb_per_100': bb_per_100,
        'wins': wins,
        'losses': losses,
        'elapsed_seconds': elapsed
    }

    print("-" * 50)
    print(f"\nResults against Slumbot:")
    print(f"  Hands played: {hands_played}")
    print(f"  Total won: {bb_won:+.1f} bb")
    print(f"  Win rate: {mbb_per_hand:+.1f} mbb/hand ({bb_per_100:+.1f} bb/100)")
    print(f"  Wins/Losses: {wins}/{losses}")
    print(f"  Time: {elapsed:.1f}s ({hands_played/elapsed:.1f} hands/sec)")

    print(f"\nReference rates:")
    print(f"  ReBeL: +45 mbb/hand")
    print(f"  Ruse: +194 mbb/hand")
    print(f"  Call-station: ~-500 mbb/hand")

    # Print action distribution if agent supports it
    if hasattr(agent, 'print_action_distribution'):
        agent.print_action_distribution()

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark against Slumbot')
    parser.add_argument('--hands', type=int, default=100, help='Number of hands to play')
    parser.add_argument('--agent', type=str, default='ares',
                        choices=['ares', 'call', 'random', 'heuristic', 'search', 'neural'],
                        help='Agent to use')
    parser.add_argument('--strategy', type=str, default='blueprints/cpp_1M/strategy_1M.bin',
                        help='Path to ARES strategy file')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to neural network checkpoint (for --agent neural)')
    parser.add_argument('--iterations', type=int, default=200,
                        help='CFR iterations for search agent (default: 200)')
    parser.add_argument('--username', type=str, help='Slumbot username for leaderboard')
    parser.add_argument('--password', type=str, help='Slumbot password')
    parser.add_argument('--verbose', action='store_true', help='Print each hand')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for neural network (cpu/cuda/mps)')

    args = parser.parse_args()

    print("=" * 60)
    print("ARES-HU Slumbot Benchmark")
    print("=" * 60)

    # Create agent
    if args.agent == 'neural':
        if not args.checkpoint:
            print("ERROR: --checkpoint required for neural agent")
            sys.exit(1)
        agent = NeuralNetworkAgent(args.checkpoint, device=args.device)
    elif args.agent == 'ares':
        agent = ARESSolverAgent(args.strategy)
    elif args.agent == 'call':
        agent = CallStationAgent()
        print("Using call-station agent (baseline)")
    elif args.agent == 'heuristic':
        agent = SimpleHeuristicAgent()
        print("Using simple heuristic agent (baseline)")
    elif args.agent == 'search':
        agent = RealtimeSearchAgent(iterations=args.iterations, qre_tau=1.0, time_limit=30.0)
    else:
        agent = RandomAgent()
        print("Using random agent (baseline)")

    # Run benchmark
    results = run_benchmark(
        agent,
        num_hands=args.hands,
        username=args.username,
        password=args.password,
        verbose=args.verbose
    )

    return results


if __name__ == '__main__':
    main()
