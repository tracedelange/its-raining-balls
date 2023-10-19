import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import pdb
from brain import Brain
from config import Config


#convert the above code into a function
def get_input_array(balls):
    N = 8
    DIM = 800

    input_array = np.zeros(N)
    for ball in balls:
        try:
            x, y = ball
            index = min(int(x // (DIM / N)), N-1)
            input_array[index] = max(input_array[index], y) / DIM
        except IndexError:
            print('Index error', ball)
    return input_array

con = Config()


def evaluate_network(mode):

    # player = con.WIDTH // 2 - con.PLAYER_SIZE // 2
    player = random.randint(0, con.WIDTH - con.PLAYER_SIZE)  # Randomly initialize player position

    initial_player = player

    balls = []
    score = 0
    brain = Brain()

    running = True

    while running:
        ia = get_input_array(balls)

        input_tensor = torch.tensor(ia, dtype=torch.float32)

        # Pass the input tensor through the model
        action = brain(input_tensor)

        highest_index = torch.argmax(action).item()

        if highest_index == 0:
            player -= con.PLAYER_SPEED
        elif highest_index == 2:
            player += con.PLAYER_SPEED

        if player < 0:
            player = con.WIDTH - con.PLAYER_SIZE  # Wrap to the right side
        elif player > con.WIDTH - con.PLAYER_SIZE:
            player = 0  # Wrap to the left side

        # Generate balls with random x and y positions
        if random.randint(1, 100) < 5:
            x = random.randint(0, con.WIDTH - con.BALL_SIZE)
            y = 0
            balls.append((x, y))

        # Move balls
        balls = [(x, y + con.BALL_SPEED) for x, y in balls]

        # Remove balls that go out of the screen
        balls = [(x, y) for x, y in balls if y < con.HEIGHT]

        # Check for collisions with the player
        for x, y in balls:
            if player <= x <= player + con.PLAYER_SIZE:
                running = False
        
        # Increase score
        score += 1
        con.BALL_SPEED += score * con.BALL_SPEED_SCALE

        if score > 300:
            torch.save(brain.state_dict(), './models/output.txt')

    # Game over
    print("Game Over! Your Score:", score)
    return score


if __name__ == "__main__":
    evaluate_network(mode="model")