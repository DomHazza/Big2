from typing import List, Optional, Tuple
from itertools import combinations
from collections import Counter
from enum import Enum, auto
import random


class Suit(Enum):
    DIAMOND = 0
    CLUB = 1
    HEART = 2
    SPADE = 3


class HandType(Enum):
    INVALID = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


class Card:
    def __init__(self, rank: int, suit: Suit):
        self.rank = rank  # 3 = lowest, 2 = highest
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit.name[0]}"

    def __lt__(self, other):
        if self.rank == other.rank:
            return self.suit.value < other.suit.value
        return self.rank < other.rank

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))


class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand: List[Card] = []

    def receive_cards(self, cards: List[Card]):
        self.hand = sorted(cards)

    def remove_cards(self, cards: List[Card]):
        for card in cards:
            self.hand.remove(card)

    def get_legal_moves(self, last_play: Optional[List[Card]]) -> List[Optional[List[Card]]]:
        legal_moves = []

        # Try group sizes 1, 2, 3, and 5
        for k in [1, 2, 3, 5]:
            for combo in combinations(self.hand, k):
                move = list(combo)
                if is_valid_move(move, last_play):
                    legal_moves.append(move)

        # Allow pass if not leading
        if last_play is not None:
            legal_moves.append(None)

        return legal_moves

    def play_random_move(self, last_play: Optional[List[Card]]) -> Optional[List[Card]]:
        legal_moves = self.get_legal_moves(last_play)
        if not legal_moves:
            print(f"Player {self.player_id} has no valid moves. Forced pass.")
            return None
        return random.choice(legal_moves)

    def compare_moves(self, move1: List[Card], move2: List[Card]) -> int:
        # Compare single card plays only for now
        card1, card2 = move1[0], move2[0]
        if card1.rank != card2.rank:
            return card1.rank - card2.rank
        return card1.suit.value - card2.suit.value


class Big2Game:
    def __init__(self):
        self.players = [Player(i) for i in range(4)]
        self.current_player = 0
        self.last_play: Optional[List[Card]] = None
        self.passes = [False] * 4
        self.starting_player = None
        self.scores = [0] * 4

    def deal_cards(self):
        deck = [Card(rank, suit) for rank in range(3, 15) for suit in Suit]
        random.shuffle(deck)
        for i in range(4):
            hand = deck[i * 13:(i + 1) * 13]
            self.players[i].receive_cards(hand)
            for card in hand:
                if card.rank == 3 and card.suit == Suit.DIAMOND:
                    self.starting_player = i
        self.current_player = self.starting_player

    def play_turn(self):
        player = self.players[self.current_player]
        move = player.play_random_move(self.last_play)

        if move is None:
            self.passes[self.current_player] = True
            print(f"Player {self.current_player} passes.")
        else:
            print(f"Player {self.current_player} plays: {move}")
            player.remove_cards(move)
            self.last_play = move
            self.passes = [False] * 4  # Reset passes

        self.current_player = (self.current_player + 1) % 4

    def is_round_reset(self):
        return sum(self.passes) == 3

    def reset_round(self):
        self.last_play = None
        self.passes = [False] * 4

    def is_game_over(self):
        return any(len(p.hand) == 0 for p in self.players)

    def compute_scores(self):
        for i, player in enumerate(self.players):
            self.scores[i] = len(player.hand)
        print("Final Scores (lower is better):")
        for i, score in enumerate(self.scores):
            print(f"Player {i}: {score} cards remaining")

    def play_game(self):
        self.deal_cards()
        print(f"Starting Player: {self.current_player}")
        while not self.is_game_over():
            self.play_turn()
            if self.is_round_reset():
                print("Round reset.")
                self.reset_round()
        self.compute_scores()





def classify_hand(cards: List[Card]) -> Tuple[HandType, int, int]:
    if not cards:
        return HandType.INVALID, 0, 0

    cards = sorted(cards)
    ranks = [card.rank for card in cards]
    suits = [card.suit for card in cards]
    rank_counts = Counter(ranks)
    count_values = sorted(rank_counts.values(), reverse=True)
    
    is_flush = len(set(suits)) == 1
    is_straight = (
        len(cards) == 5 and 
        sorted(ranks) == list(range(ranks[0], ranks[0] + 5))
    )

    # SINGLE
    if len(cards) == 1:
        return HandType.SINGLE, cards[0].rank, cards[0].suit.value

    # PAIR
    if len(cards) == 2 and len(rank_counts) == 1:
        return HandType.PAIR, ranks[0], max(s.value for s in suits)

    # TRIPLE
    if len(cards) == 3 and len(rank_counts) == 1:
        return HandType.TRIPLE, ranks[0], max(s.value for s in suits)

    # FIVE-CARD HANDS
    if len(cards) == 5:
        if is_flush and is_straight:
            return HandType.STRAIGHT_FLUSH, max(ranks), max(s.value for s in suits)
        elif 4 in count_values:
            # Four of a kind + 1
            main_rank = [rank for rank, cnt in rank_counts.items() if cnt == 4][0]
            return HandType.FOUR_OF_A_KIND, main_rank, 0
        elif sorted(count_values) == [2, 3]:
            # Full house
            main_rank = [rank for rank, cnt in rank_counts.items() if cnt == 3][0]
            return HandType.FULL_HOUSE, main_rank, 0
        elif is_flush:
            return HandType.FLUSH, max(ranks), max(s.value for s in suits)
        elif is_straight:
            return HandType.STRAIGHT, max(ranks), max(s.value for s in suits)
    return HandType.INVALID, 0, 0


def is_valid_move(move: Optional[List[Card]], last_play: Optional[List[Card]]) -> bool:
    if move is None:
        return last_play is not None  # Cannot pass if leading
    htype, hrank, _ = classify_hand(move)
    if htype == HandType.INVALID:
        return False
    if last_play is None:
        return True
    ltype, lrank, _ = classify_hand(last_play)
    if htype != ltype:
        return False
    return compare_hands(move, last_play) > 0


def compare_hands(move1: List[Card], move2: List[Card]) -> int:
    t1, r1, s1 = classify_hand(move1)
    t2, r2, s2 = classify_hand(move2)
    if t1.value != t2.value:
        return t1.value - t2.value
    if r1 != r2:
        return r1 - r2
    return s1 - s2


def generate_all_possible_moves() -> List[Optional[List[Card]]]:
    all_cards = [Card(rank, suit) for rank in range(3, 15) for suit in Suit]
    move_set = set()

    # Try group sizes 1, 2, 3, and 5 (singles, pairs, triples, 5-card hands)
    for k in [1, 2, 3, 5]:
        for combo in combinations(all_cards, k):
            move = list(combo)
            if classify_hand(move)[0] != HandType.INVALID:
                move_set.add(tuple(sorted((card.rank, card.suit.value) for card in move)))

    # Add None for pass
    move_set.add(None)
    return [None if m is None else [Card(rank, Suit(suit)) for rank, suit in m] for m in move_set]


if __name__ == "__main__":
    game = Big2Game()
    game.play_game()