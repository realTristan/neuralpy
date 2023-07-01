import pygame
from config import SCREEN, BACKGROUND_COLOR, CLOCK_SPEED, EPOCHS
from prey import Prey
from predator import Predator
from models import PREY_MODEL, PREDATOR_MODEL

# Infinite Loop
clock: pygame.time.Clock = pygame.time.Clock()

# Winners
predator_wins: int = 0
prey_wins: int = 0

# Run the program
prey: list[Prey] = [Prey.new() for _ in range(50)]
predators: list[Predator] = [Predator.new() for _ in range(50)]

while 1:
    SCREEN.fill(BACKGROUND_COLOR)
    
    # Move the prey
    for p in prey:
        data = p.move(predators, prey)
        prey, output = p.is_colliding_prey(predators, prey)
        PREY_MODEL._train(data, output)
    
    # Move the predators
    for p in predators:
        if not p.can_move():
            predators.remove(p)
            continue
        
        data = p.move()
        prey, predators, output = p.is_colliding(prey, predators)
        PREDATOR_MODEL._train(data, output)
    
    # If prey win
    if len(prey) > 120 or len(predators) <= 10:
        prey_wins += 1
        print("Prey win!")
        break
    
    # If predators win
    if len(predators) > 120 or len(prey) <= 10:
        predator_wins += 1
        print("Predators win!")
        break
    
    # Display how much prey and predators there are
    pygame.display.set_caption(
        f"Alive [Prey: {len(prey)} Predators: {len(predators)}] - Wins [Prey: {prey_wins} Predators: {predator_wins}]")

    # Update the display
    pygame.display.update()
    clock.tick(CLOCK_SPEED)
    
    # Game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
