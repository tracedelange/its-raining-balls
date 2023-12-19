class Config(object):

    def __init__(self) -> None:    
        # Constants
        self.WIDTH = 800
        self.HEIGHT = 600
        self.PLAYER_SIZE = 50
        self.PLAYER_SPEED = 5
        self.BALL_SIZE = 20
        self.BALL_SPEED = 5

        self.INPUT_NODE_SIZE = 32
        
        self.BALL_SPEED_SCALE = 0.0000001  # Adjust this value based on how fast you want the balls to get.
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # GA parameters
        self.population_size = 250
        self.num_generations = 500
        self.mutation_rate = 0.01
        self.mutation_strength = 0.2
        self.elitism = True
        self.num_elites = 1
        self.num_parents = 8
        self.tournament_size = 50
        self.crossover_type = "single_point"
        self.crossover_rate = 0.4

    