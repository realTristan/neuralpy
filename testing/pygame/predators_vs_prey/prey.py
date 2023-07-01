import pygame
import random
import torch
from config import SCREEN, WIDTH, HEIGHT
from models import PREY_MODEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Prey:
    def __init__(self, x: int, y: int, speed: int):
        self.x: int = x
        self.y: int = y
        self.speed: int = speed
        self.x_multiplier: float = 1.0
        self.y_multiplier: float = 1.0
        self.color: tuple[int, int, int] = (100, 100, 255)
        self.last_output: float = 0.0
        self.size: int = 5

    def move(self, predators, prey):
        # Movement multiplier
        self.x_multiplier = self.speed * random.choice([-1, 1])
        self.y_multiplier = self.speed * random.choice([-1, 1])
        self.x, self.y = self.x + self.x_multiplier, self.y + self.y_multiplier

        # If the prey goes off the screen, bring it back
        if self.x > WIDTH - 50:
            self.x -= self.x_multiplier

        # If the prey goes off the screen, bring it back
        if self.y > HEIGHT - 50:
            self.y -= self.y_multiplier

        # Get the closest predators and prey
        closest_predator, closest_prey = self.closest(predators, prey)

        # Model prediction
        data: list = [self.x, self.y, self.last_output,
                      closest_prey, closest_predator]
        data_tensor: torch.Tensor = torch.tensor([data]).float()
        self.last_output = PREY_MODEL(data_tensor)
        
        # Use the last output
        self.x_multiplier *= self.last_output.item()
        self.y_multiplier *= self.last_output.item()

        # Print the output
        print(f"PREY: {self.last_output.item()}")

        # Draw the prey as a stick with width of self.size
        pygame.draw.line(SCREEN, self.color, (self.x, self.y),
                         (self.x + self.x_multiplier, self.y + self.y_multiplier),
                         self.size)

        # pygame.draw.circle(SCREEN, self.color, (self.x, self.y), self.size)

        # Return the data
        return data_tensor

    # Get the closest prey and predators positions
    def closest(self, predators, prey):
        # Get the closest prey and predators
        closest_prey: float = -1.0
        closest_predator: float = -1.0

        # Iterate over the predators
        for p in predators:
            if abs(self.x - p.x) + abs(self.y - p.y) < closest_predator or closest_predator == -1:
                closest_predator = abs(self.x - p.x) + abs(self.y - p.y)

        # Iterate over the prey
        for p in prey:
            if p == self:
                continue

            # Calculate closest prey
            if abs(self.x - p.x) + abs(self.y - p.y) < closest_prey or closest_prey == -1:
                closest_prey = abs(self.x - p.x) + abs(self.y - p.y)

        # Return the results
        return closest_predator, closest_prey

    # Check if the prey is colliding with another prey

    def is_colliding_prey(self, predators, prey):
        len0: int = len(prey)
        is_eaten: int = 0

        # Iterate over the predators
        for p in predators:
            if abs(self.x - p.x) < 10 and abs(self.y - p.y) < 10:
                is_eaten = 1
                break

        # Iterate over the prey
        for p in prey:
            if p == self:
                continue

            if abs(self.x - p.x) < 10 and abs(self.y - p.y) < 10:
                new_prey = Prey.new()
                prey.append(new_prey)

                # Random chance of the prey dying during birth
                if random.choice([1, 1, 0, 0, 0, 0]):
                    prey.remove(self)
                    new_prey.size += 1
                    break

        # Return the correct cboice
        output: int = (0 if len(prey) != len0 or is_eaten == 0 else 1)
        return prey, torch.tensor(
            [output + self.y_multiplier * self.x_multiplier]
        ).float()

    # Create a new prey

    @staticmethod
    def new():
        x: int = random.randint(50, WIDTH - 50)
        y: int = random.randint(50, HEIGHT - 50)
        speed: int = random.randint(1, 5)
        return Prey(x, y, speed)
