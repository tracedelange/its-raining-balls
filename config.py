class Config(object):

    def __init__(self) -> None:    
        # Constants
        self.WIDTH = 800
        self.HEIGHT = 600
        self.PLAYER_SIZE = 50
        self.PLAYER_SPEED = 5
        self.BALL_SIZE = 20
        self.BALL_SPEED = 5
        
        self.BALL_SPEED_SCALE = 0.0000001  # Adjust this value based on how fast you want the balls to get.
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)