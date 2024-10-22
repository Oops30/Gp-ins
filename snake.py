import pygame
import random

# Initialize pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 500, 400
BLOCK_SIZE = 10
FPS = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up font for score display
font = pygame.font.SysFont("Arial", 24)

# Snake and food
snake = [(100, 50), (90, 50), (80, 50)]
food = (random.randrange(0, WIDTH, BLOCK_SIZE), random.randrange(0, HEIGHT, BLOCK_SIZE))
dx, dy = BLOCK_SIZE, 0
score = 0  # Initialize score

# Game loop
running, clock = True, pygame.time.Clock()
game_over = False

while running:
    if not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and dy == 0: dx, dy = 0, -BLOCK_SIZE
                if event.key == pygame.K_DOWN and dy == 0: dx, dy = 0, BLOCK_SIZE
                if event.key == pygame.K_LEFT and dx == 0: dx, dy = -BLOCK_SIZE, 0
                if event.key == pygame.K_RIGHT and dx == 0: dx, dy = BLOCK_SIZE, 0

        # Move snake
        head = (snake[0][0] + dx, snake[0][1] + dy)
        snake = [head] + snake[:-1]

        # Check for collisions
        if head == food:
            snake.append(snake[-1])  # Grow snake
            score += 1  # Increase score when food is eaten
            food = (random.randrange(0, WIDTH, BLOCK_SIZE), random.randrange(0, HEIGHT, BLOCK_SIZE))
        if head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT or head in snake[1:]:
            game_over = True

        # Draw everything
        screen.fill(BLACK)
        for segment in snake:
            pygame.draw.rect(screen, GREEN, (*segment, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(screen, RED, (*food, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.flip()
        clock.tick(FPS)

    else:
        # Display Game Over and Score
        screen.fill(BLACK)
        game_over_text = font.render(f"Game Over! Score: {score}", True, WHITE)
        screen.blit(game_over_text, (WIDTH // 4, HEIGHT // 2))
        pygame.display.flip()

        # Wait for a few seconds before closing
        pygame.time.wait(2000)  # Wait for 2 seconds
        running = False

pygame.quit()