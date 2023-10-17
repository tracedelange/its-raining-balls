import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
PLAYER_SIZE = 50
PLAYER_SPEED = 5
BALL_SIZE = 20
BALL_SPEED = 5

BALL_SPEED_SCALE = 0.0000001  # Adjust this value based on how fast you want the balls to get.

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("It's Raining Balls!")

# Player
player = pygame.Rect(WIDTH // 2 - PLAYER_SIZE // 2, HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE)

# Ball list
balls = []

# Clock
clock = pygame.time.Clock()

# Score
score = 0

font = pygame.font.Font(None, 36)  # You can specify the font and font size (None means a default font).


# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player.left > 0:
        player.x -= PLAYER_SPEED
    if keys[pygame.K_RIGHT] and player.right < WIDTH:
        player.x += PLAYER_SPEED

    # Generate balls with random colors
    if random.randint(1, 100) < 5:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        ball = pygame.Rect(random.randint(0, WIDTH - BALL_SIZE), 0, BALL_SIZE, BALL_SIZE)
        balls.append((ball, color))

    # Move balls
    for ball, color in balls:
        ball.y += BALL_SPEED

    # Remove balls that go out of the screen
    balls = [ball for ball in balls if ball[0].y < HEIGHT]

    # Check for collisions with the player
    for ball in balls:
        if player.colliderect(ball[0]):
            running = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw player and balls
    pygame.draw.rect(screen, BLACK, player)
    
    # Draw balls with random colors
    for ball, color in balls:
        pygame.draw.ellipse(screen, color, ball)


    score_text = font.render("Score: " + str(score), True, (0, 0, 0))
    score_rect = score_text.get_rect()
    score_rect.topleft = (10, 10)
    screen.blit(score_text, score_rect)


    # Update the display
    pygame.display.flip()

    # Increase score
    score += 1

    BALL_SPEED += score * BALL_SPEED_SCALE

    # Cap the frame rate
    clock.tick(60)

# Game over
pygame.quit()
print("Game Over! Your Score:", score)
