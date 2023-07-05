from copy import deepcopy, copy
import numpy as np

from rlcard.games.newlimitholdem import Dealer
from rlcard.games.newlimitholdem import Player, PlayerStatus
from rlcard.games.newlimitholdem import Judger
from rlcard.games.newlimitholdem import Round


class NewLimitHoldemGame:
    def __init__(self, allow_step_back=False, num_players=2):
        """Initialize the class limit holdem game"""
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()

        # Some configurations of the game
        # These arguments can be specified for creating new games

        # Small blind and big blind
        self.ante = 1
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind

        # Raise amount and allowed times
        self.raise_amount = self.ante
        self.allowed_raise_num = 2
        self.allowed_action_num = 2

        self.num_players = num_players

        # Save betting history
        self.history_raise_nums = [0 for _ in range(2)]

        self.dealer = None
        self.players = None
        self.judger = None
        self.public_cards = None
        self.game_pointer = None
        self.round = None
        self.round_counter = None
        self.history = None
        self.history_raises_nums = None
        self.first = None
        self.pcards = None

    def configure(self, game_config):
        """Specify some game specific parameters, such as number of players"""
        self.num_players = game_config['game_num_players']

    def init_game(self, starter=None, agent=None, hcard=None, pcard1=None, pcard2=None, opcard=None):
        """
        Initialize the game of limit texas holdem

        This version supports two-player limit texas holdem

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        """
        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize two players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        if pcard1 is not None and pcard2 is not None:
            self.pcards = []
            self.pcards.append(pcard1)
            self.pcards.append(pcard2)


        # Deal cards to each  player to prepare for the first round
        # 1 card per player for our mode
        if hcard is None:
            for i in range(self.num_players):
                self.players[i % self.num_players].hand.append(self.dealer.deal_card())
        else:
            self.players[agent].hand.append(hcard)

        if opcard is not None:
            self.players[(agent + 1) % self.num_players].hand.append(opcard)


        # Initialize public cards
        self.public_cards = []

        # Randomly choose a small blind and a big blind
        s = self.np_random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].in_chips = self.ante
        self.players[s].in_chips = self.ante

        # The player that plays the first
        if starter is None:
            self.game_pointer = (b + 1) % self.num_players
        else:
            self.game_pointer = starter
        self.first = self.game_pointer

        # Initialize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(raise_amount=self.raise_amount,
                           allowed_action_num=self.allowed_action_num,
                           num_players=self.num_players,
                           np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 3 rounds in each game.
        self.round_counter = 0

        # Save the history for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        # Save betting history
        self.history_raise_nums = [0 for _ in range(2)]

        return state, self.game_pointer

    def step(self, action):
        """
        Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next player id
        """
        if self.allow_step_back:
            # First snapshot the current state
            r = deepcopy(self.round)
            b = self.game_pointer
            r_c = self.round_counter
            d = deepcopy(self.dealer)
            p = deepcopy(self.public_cards)
            ps = deepcopy(self.players)
            rn = copy(self.history_raise_nums)
            self.history.append((r, b, r_c, d, p, ps, rn))

        # Then we proceed to the next round
        self.game_pointer = self.round.proceed_round(self.players, action)

        # Save the current raise num to history
        self.history_raise_nums[self.round_counter] = self.round.have_raised

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # For the first round, we deal 2 cards
            if self.round_counter == 0:
                if self.pcards is None:
                    self.public_cards.append(self.dealer.deal_card())
                    self.public_cards.append(self.dealer.deal_card())
                else:
                    self.public_cards.append(self.pcards[0])
                    self.public_cards.append(self.pcards[1])

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def change_hand(self, card, player_id):
        if not self.players[player_id].hand:
            self.players[player_id].hand.append(card)
        else:
            self.players[player_id].hand[0] = card

    def op_hand(self, player):
        if not self.players[player].hand:
            return None
        else:
            return self.players[player].hand[0]

    def step_back(self):
        """
        Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        """
        if len(self.history) > 0:
            self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, \
                self.players, self.history_raises_nums = self.history.pop()
            return True
        return False

    def get_num_players(self):
        """
        Return the number of players in limit texas holdem

        Returns:
            (int): The number of players in the game
        """
        return self.num_players

    @staticmethod
    def get_num_actions():
        """
        Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 4 actions (call, raise, check and fold)
        """
        return 4

    def get_player_id(self):
        """
        Return the current player's id

        Returns:
            (int): current player's id
        """
        return self.game_pointer

    def get_state(self, player):
        """
        Return player's state

        Args:
            player (int): player id

        Returns:
            (dict): The state of the player
        """

        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.get_legal_actions()
        state = self.players[player].get_state(self.public_cards, chips, legal_actions, self.first)
        state['raise_nums'] = self.history_raise_nums

        return state

    def is_over(self):
        """
        Check if the game is over

        Returns:
            (boolean): True if the game is over
        """
        alive_players = [1 if p.status in (PlayerStatus.ALIVE, PlayerStatus.ALLIN) else 0 for p in self.players]
        # If only one player is alive, the game is over.
        if sum(alive_players) == 1:
            return True

        # If all rounds are finished
        if self.round_counter >= 2:
            return True
        return False

    def get_payoffs(self):
        """
        Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        """
        hands = [p.hand + self.public_cards if p.status == PlayerStatus.ALIVE else None for p in self.players]
        chips_payoffs = self.judger.judge_game(self.players, hands)
        payoffs = np.array(chips_payoffs) / self.ante
        return payoffs

    def get_legal_actions(self):
        """
        Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        """
        return self.round.get_legal_actions()
