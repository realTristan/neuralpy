import pygame
from config import SCREEN, BACKGROUND_COLOR, CLOCK_SPEED, EPOCHS
from prey import Prey
from predator import Predator
from models import PREY_MODEL

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
        prey, made_baby = p.is_colliding_prey(predators, prey)
        PREY_MODEL._train(data, made_baby)
    
    # Move the predators
    for p in predators:
        if not p.can_move():
            predators.remove(p)
            continue
        
        p.move()
        prey = p.is_colliding_prey(prey)
        predators = p.multiply(predators)
    
    # If prey win
    if len(prey) > 100 or len(predators) <= 10:
        print("Prey win!")
        break
    
    # If predators win
    if len(predators) > 100 or len(prey) <= 10:
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
    

