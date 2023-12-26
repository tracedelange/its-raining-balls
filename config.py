class Config(object):

    def __init__(self) -> None:
        # Constants
        self.WIDTH = 800
        self.HEIGHT = 600
        # self.WIDTH = 1600
        # self.HEIGHT = 800
        self.PLAYER_SIZE = 50
        self.PLAYER_SPEED = 5
        self.BALL_SIZE = 20
        self.BALL_SPEED = 5
        self.BALL_SPAWN_RATE = 0.1

        self.INPUT_NODE_SIZE = 32

        # Optional Game Features
        self.INITIAL_MIDDLE_BALL = True
        self.WALL_HUG_PENALTY = True

        # Visual elements
        self.GRAPH_ACTIVE = False
        self.VISUAL_FIELD_ACTIVE = False

        # TODO Add idle penalty

        # Adjust this value based on how fast you want the balls to get.
        self.BALL_SPEED_SCALE = 0.0000001

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Show sim of best agent every x generations
        self.preview_mod = 1
        self.preview_active = False

        # GA parameters
        self.population_size = 250
        self.num_generations = 500
        self.mutation_rate = 0.5
        self.mutation_strength = 0.5
        self.elitism = True
        self.num_elites_rate = 0.05
        self.tournament_size_rate = 0.5
        self.num_parent_rate = 0.15
        self.crossover_rate = 1
        self.crossover_type = "single_point"

        self.same_seed = True

        # Derived parameters
        self.num_elites = int((self.population_size *
                               self.num_elites_rate))
        self.num_parents = int((self.population_size *
                               self.num_parent_rate))
        self.tournament_size = int((self.population_size *
                                   self.tournament_size_rate))
