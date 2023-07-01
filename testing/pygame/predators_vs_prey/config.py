import pygame

# create a new window
pygame.init()
pygame.display.set_caption("Evolution")

# Variables
CLOCK_SPEED: int = 5
WIDTH: int = 1200
HEIGHT: int = 600
BACKGROUND_COLOR: tuple[int, int, int] = (210, 210, 210)
EPOCHS: int = 100

# Initialize the screen
SCREEN: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
SCREEN.fill(BACKGROUND_COLOR)
