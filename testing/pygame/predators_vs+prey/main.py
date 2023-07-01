import pygame
from config import SCREEN, BACKGROUND_COLOR, CLOCK_SPEED
from prey import Prey
from predator import Predator
from models import PREY_MODEL, PREDATOR_MODEL

prey: list[Prey] = []
for i in range(50):
  prey.append(Prey.new())
  
predators: list[Predator] = []
for i in range(50):
  predators.append(Predator.new())


# Infinite Loop
clock: pygame.time.Clock = pygame.time.Clock()

# Run the epochs
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
    if len(prey) > 150 or len(predators) <= 10:
        print("Prey win!")
        break
    
    # If predators win
    if len(predators) > 150 or len(prey) <= 10:
        print("Predators win!")
        break
    
    # Display how much prey and predators there are
    pygame.display.set_caption(f"Evolution - Prey: {len(prey)} Predators: {len(predators)}")

    # Update the display
    pygame.display.update()
    clock.tick(CLOCK_SPEED)
    
    # Game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    

