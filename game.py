import pygame
import random

import torch

import pdb
import numpy as np
from brain import Brain
from config import Config
from config import Config


con = Config()


def get_input_array(balls, player):
    N = con.INPUT_NODE_SIZE
    DIM = con.WIDTH

    input_array = np.zeros(N + 2)
    for ball in balls:
        if len(ball) == 2:
            x, y = ball
            start_index = int(x // (DIM / N))
            end_index = int((x) // (DIM / N))
            for index in range(start_index, end_index + 1):
                input_array[index] = max(input_array[index], y) / DIM
        else:
            print('Invalid ball format:', ball)

    input_array[N] = player.right / DIM
    input_array[N + 1] = player.left / DIM

    return input_array


def _draw_input_array(ia, screen):

    # Divide the screen into columns of n = con.INPUT_NODE_SIZE, with lines in between
    for i in range(con.INPUT_NODE_SIZE):
        pygame.draw.line(screen, (0, 0, 0, 0.5), (i * con.WIDTH // con.INPUT_NODE_SIZE,
                                                  0), (i * con.WIDTH // con.INPUT_NODE_SIZE, con.HEIGHT), 1)

        # Also render text of the value of the IA at that index
        font = pygame.font.Font(None, 20)
        text = font.render(str(round(ia[i], 2)), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (i * con.WIDTH // con.INPUT_NODE_SIZE, 40)
        screen.blit(text, text_rect)


# random.seed(0)

def loop_game(model=None, generation=None, individual_index=None, virtual=False, random_seed=0):

    if not virtual:
        print(random_seed)

    # Initialize Pygame
    if not virtual:
        pygame.init()
        # Create the game window
        screen = pygame.display.set_mode((con.WIDTH, con.HEIGHT))
        pygame.display.set_caption("It's Raining Balls!")

    # Assign the random seed
    random.seed(random_seed)

    # Player
    player = pygame.Rect(con.WIDTH // 2 - con.PLAYER_SIZE // 2,
                         con.HEIGHT - con.PLAYER_SIZE, con.PLAYER_SIZE, con.PLAYER_SIZE)

    # Ball list
    if con.INITIAL_MIDDLE_BALL:
        # Start with one ball in the middle of the screen
        balls = [(pygame.Rect(con.WIDTH // 2 - con.BALL_SIZE //
                              2, 0, con.BALL_SIZE, con.BALL_SIZE), (0, 0, 0))]
    else:
        balls = []

    # Clock
    clock = pygame.time.Clock()

    # Score
    if not virtual:
        # You can specify the font and font size (None means a default font).
        font = pygame.font.Font(None, 36)

    brain = model

    # Game loop
    running = True
    score = 0

    BALL_SPEED = con.BALL_SPEED

    while running:
        if not virtual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        if brain is None:
            if not virtual:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT] and player.left > 0:
                    player.x -= con.PLAYER_SPEED
                if keys[pygame.K_RIGHT] and player.right < con.WIDTH:
                    player.x += con.PLAYER_SPEED
        else:
            # Get input array
            ia = get_input_array([(ball[0].x, ball[0].y)
                                 for ball in balls], player)

            # Draw input array as columns on the screen
            if not virtual and con.VISUAL_FIELD_ACTIVE:
                _draw_input_array(ia, screen)

            input_tensor = torch.tensor(ia, dtype=torch.float32)

            # Pass the input tensor through the model
            action = brain(input_tensor)

            highest_index = torch.argmax(action).item()

            if highest_index == 0 and player.left > 0:
                player.left -= con.PLAYER_SPEED
            elif highest_index == 2 and player.right < con.WIDTH:
                player.left += con.PLAYER_SPEED

        # Generate balls with random colors
        if random.randint(1, 100) < con.BALL_SPAWN_RATE * 100:
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))
            ball = pygame.Rect(random.randint(
                0, con.WIDTH - con.BALL_SIZE), 0, con.BALL_SIZE, con.BALL_SIZE)
            balls.append((ball, color))

        # Move balls
        for ball, color in balls:
            ball.y += BALL_SPEED
        # Remove balls that go out of the screen
        balls = [ball for ball in balls if ball[0].y < con.HEIGHT]

        # Check for collisions with the player
        for ball in balls:
            if player.colliderect(ball[0]):
                running = False

        # Clear the screen
        if not virtual:
            screen.fill(con.WHITE)

        # Draw player and balls
        if not virtual:
            pygame.draw.rect(screen, con.BLACK, player)

        if not virtual and con.VISUAL_FIELD_ACTIVE:
            # Get input array
            ia = get_input_array([(ball[0].x, ball[0].y)
                                 for ball in balls], player)
            _draw_input_array(ia, screen)

        # Draw balls with random colors
        if not virtual:
            for ball, color in balls:
                pygame.draw.ellipse(screen, color, ball)

        if not virtual:
            # Draw the score
            score_text = font.render("Score: " + str(score), True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topleft = (10, 10)
            screen.blit(score_text, score_rect)

            # if generation is not none, print value under score
            if generation is not None:
                gen_text = font.render(
                    "Generation: " + str(generation), True, (0, 0, 0))
                gen_rect = gen_text.get_rect()
                gen_rect.topleft = (10, 40)
                screen.blit(gen_text, gen_rect)

            # if individual_index is not none, print value under generation
            if individual_index is not None:
                ind_text = font.render(
                    "Individual: " + str(individual_index), True, (0, 0, 0))
                ind_rect = ind_text.get_rect()
                ind_rect.topleft = (10, 70)
                screen.blit(ind_text, ind_rect)

        # Update the display
        if not virtual:
            if con.GRAPH_ACTIVE:
                graph = pygame.image.load('./figures/training_graph.png')
                width = graph.get_rect().width
                height = graph.get_rect().height
                graph = pygame.transform.scale(
                    graph, (int(width * 0.6), int(height * 0.6)))
                # Make the image 90% transparent
                graph.set_alpha(0)
                # Draw graph in top right corner
                screen.blit(graph, (con.WIDTH - width * 0.5, 0))
            pygame.display.update()
            pygame.display.flip()

        if con.WALL_HUG_PENALTY:
            # Increase score if player is not on left or right edge and is moving
            if player.left > 0 and player.right < con.WIDTH:
                score += 1
        else:
            score += 1

        # Cap the frame rate
        if not virtual:
            # maximum fps when not rendering
            clock.tick(60)
        else:
            clock.tick(1000000)

    # Game over
    pygame.quit()
    if not virtual:
        print("Game Over! Your Score:", score)
    return score
