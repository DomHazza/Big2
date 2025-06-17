from simulator import Big2Game, Player, Card, is_valid_move
from typing import List, Optional, Tuple, Dict, Any
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import random


class Big2Env(gym.Env):
    def __init__(self):
        super(Big2Env, self).__init__()
        self.game = Big2Game()
        self.agent_id = 0
        self.action_map = []  # Maps index to legal moves
        self.observation_space = spaces.Dict({
            "hand": spaces.Box(low=0, high=1, shape=(52,), dtype=np.int8),
            "last_play": spaces.Box(low=0, high=1, shape=(52,), dtype=np.int8),
            "player_id": spaces.Discrete(4),
            "hand_sizes": spaces.Box(low=0, high=13, shape=(4,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(1000)
        self.max_steps = 500
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.game = Big2Game()
        self.game.deal_cards()
        self.step_count = 0

        # Advance until it's the agent's turn
        while self.game.current_player != self.agent_id:
            self._play_random_turn()

        return self._get_obs(), {}

    def _get_obs(self):
        player = self.game.players[self.agent_id]

        def cards_to_binary(cards: List[Card]) -> np.ndarray:
            arr = np.zeros(52, dtype=np.int8)
            for card in cards:
                idx = (card.rank - 3) * 4 + card.suit.value
                arr[idx] = 1
            return arr

        hand_binary = cards_to_binary(player.hand)
        last_play_binary = cards_to_binary(self.game.last_play or [])
        hand_sizes = np.array([len(p.hand) for p in self.game.players], dtype=np.int8)

        return {
            "hand": hand_binary,
            "last_play": last_play_binary,
            "player_id": self.agent_id,
            "hand_sizes": hand_sizes
        }

    def _get_legal_moves(self, player_id: int) -> List[Optional[List[Card]]]:
        return self.game.players[player_id].get_legal_moves(self.game.last_play)

    def _play_move(self, player_id: int, move: Optional[List[Card]]):
        player = self.game.players[player_id]
        if move:
            player.remove_cards(move)
            self.game.last_play = move
            self.game.passes = [False] * 4
        else:
            self.game.passes[player_id] = True

        if self.game.is_round_reset():
            self.game.reset_round()

    def _play_random_turn(self):
        player_id = self.game.current_player
        legal_moves = self._get_legal_moves(player_id)
        move = random.choice(legal_moves) if legal_moves else None
        self._play_move(player_id, move)

        # Check for winner
        if len(self.game.players[player_id].hand) == 0:
            self.winner = player_id
            return

        self.game.current_player = (player_id + 1) % 4

    def step(self, action_idx: int):
        self.step_count += 1
        done = False
        reward = 0.0
        self.winner = None

        # === Agent plays ===
        legal_moves = self._get_legal_moves(self.agent_id)
        move = None if action_idx >= len(legal_moves) else legal_moves[action_idx]
        self._play_move(self.agent_id, move)

        if len(self.game.players[self.agent_id].hand) == 0:
            self.winner = self.agent_id
            done = True
            reward = 1.0
            return self._get_obs(), reward, done, False, {"winner": self.winner}

        # === Random opponents play until it's agent's turn or game over ===
        while self.game.current_player != self.agent_id:
            self._play_random_turn()
            if hasattr(self, "winner") and self.winner is not None:
                done = True
                reward = 0.0  # agent loses
                return self._get_obs(), reward, done, False, {"winner": self.winner}

            self.game.current_player = (self.game.current_player + 1) % 4

        if self.step_count >= self.max_steps:
            done = True  # failsafe
            reward = 0.0

        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f"Agent (Player {self.agent_id}) Hand: {self.game.players[self.agent_id].hand}")
        print(f"Last Play: {self.game.last_play}")
        print(f"Hand sizes: {[len(p.hand) for p in self.game.players]}")
