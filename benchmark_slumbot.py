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

        r = client.act(our_action)


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

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark against Slumbot')
    parser.add_argument('--hands', type=int, default=100, help='Number of hands to play')
    parser.add_argument('--agent', type=str, default='ares',
                        choices=['ares', 'call', 'random', 'heuristic', 'search'],
                        help='Agent to use')
    parser.add_argument('--strategy', type=str, default='blueprints/cpp_1M/strategy_1M.bin',
                        help='Path to ARES strategy file')
    parser.add_argument('--iterations', type=int, default=200,
                        help='CFR iterations for search agent (default: 200)')
    parser.add_argument('--username', type=str, help='Slumbot username for leaderboard')
    parser.add_argument('--password', type=str, help='Slumbot password')
    parser.add_argument('--verbose', action='store_true', help='Print each hand')

    args = parser.parse_args()

    print("=" * 60)
    print("ARES-HU Slumbot Benchmark")
    print("=" * 60)

    # Create agent
    if args.agent == 'ares':
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
