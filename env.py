from simulator import Big2Game, Player, Card, is_valid_move, generate_all_possible_moves
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
        self.step_count = 0
        self.max_steps = 500

        # Fixed action space from precomputed legal moves
        self.all_moves = generate_all_possible_moves()
        self.action_space = spaces.Discrete(len(self.all_moves))

        self.observation_space = spaces.Dict({
            "hand": spaces.Box(low=0, high=1, shape=(52,), dtype=np.int8),
            "last_play": spaces.Box(low=0, high=1, shape=(52,), dtype=np.int8),
            "player_id": spaces.Discrete(4),
            "hand_sizes": spaces.Box(low=0, high=13, shape=(4,), dtype=np.int8),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Big2Game()
        self.game.deal_cards()
        self.step_count = 0

        while self.game.current_player != self.agent_id:
            self._play_random_turn()

        return self._get_obs(), {}

    def _card_to_index(self, card: Card) -> int:
        return (card.rank - 3) * 4 + card.suit.value

    def _cards_to_binary(self, cards: List[Card]) -> np.ndarray:
        arr = np.zeros(52, dtype=np.int8)
        for card in cards:
            idx = self._card_to_index(card)
            arr[idx] = 1
        return arr

    def _get_obs(self):
        player = self.game.players[self.agent_id]
        return {
            "hand": self._cards_to_binary(player.hand),
            "last_play": self._cards_to_binary(self.game.last_play or []),
            "player_id": self.agent_id,
            "hand_sizes": np.array([len(p.hand) for p in self.game.players], dtype=np.int8)
        }

    def _get_legal_moves(self, player_id: int) -> List[Optional[List[Card]]]:
        return self.game.players[player_id].get_legal_moves(self.game.last_play)

    def _normalize_move(self, move: Optional[List[Card]]) -> Optional[Tuple[Tuple[int, int], ...]]:
        if move is None:
            return None
        return tuple(sorted((card.rank, card.suit.value) for card in move))

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

        if len(self.game.players[player_id].hand) == 0:
            self.winner = player_id
            return

        self.game.current_player = (player_id + 1) % 4

    def step(self, action_idx: int):
        self.step_count += 1
        done = False
        reward = 0.0
        self.winner = None

        legal_moves = self._get_legal_moves(self.agent_id)
        normalized_legal = [self._normalize_move(m) for m in legal_moves]

        chosen_move = self.all_moves[action_idx]
        normalized_chosen = self._normalize_move(chosen_move)

        if normalized_chosen in normalized_legal:
            move = legal_moves[normalized_legal.index(normalized_chosen)]
        else:
            # ILLEGAL move: punish and end
            reward = -1.0
            done = True
            return self._get_obs(), reward, done, False, {"illegal_action": True}

        self._play_move(self.agent_id, move)

        if len(self.game.players[self.agent_id].hand) == 0:
            reward = 1.0
            done = True
            self.winner = self.agent_id
            return self._get_obs(), reward, done, False, {"winner": self.winner}

        # Let opponents play
        while self.game.current_player != self.agent_id:
            self._play_random_turn()
            if hasattr(self, "winner") and self.winner is not None:
                done = True
                reward = 0.0
                return self._get_obs(), reward, done, False, {"winner": self.winner}
            self.game.current_player = (self.game.current_player + 1) % 4

        if self.step_count >= self.max_steps:
            done = True
            reward = 0.0

        print(reward)
        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f"Agent (Player {self.agent_id}) Hand: {self.game.players[self.agent_id].hand}")
        print(f"Last Play: {self.game.last_play}")
        print(f"Hand sizes: {[len(p.hand) for p in self.game.players]}")
