import pygame
import random
import torch
from config import SCREEN, WIDTH, HEIGHT
from models import PREDATOR_MODEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predator:
    def __init__(self, x: int, y: int, speed: int):
        self.x: int = x
        self.y: int = y
        self.x_multiplier: float = 1.0
        self.y_multiplier: float = 1.0
        self.speed: int = speed
        self.food: int = 0
        self.moves_left: int = 10**3
        self.color: tuple[int, int, int] = (255, 0, 0)

    def can_move(self):
        return self.moves_left >= 0

    def move(self):
        # Movement multiplier
        self.x_multiplier = self.speed * random.choice([-1, 1])
        self.y_multiplier = self.speed * random.choice([-1, 1])
        self.x, self.y = self.x + self.x_multiplier, self.y + self.y_multiplier

        # If the prey goes off the screen, bring it back
        if self.x > WIDTH - 50:
            self.x -= self.x_multiplier
            return

        # If the prey goes off the screen, bring it back
        if self.y > HEIGHT - 50:
            self.y -= self.y_multiplier
            return

        self.moves_left -= 1

        # Model prediction
        data = [self.x, self.y, self.x_multiplier,
                self.y_multiplier, self.moves_left]
        data_tensor = torch.tensor([data]).float()
        self.last_output = PREDATOR_MODEL(data_tensor)

        # Print the output
        print(f"PREDATOR: {self.last_output.item()}")

        # Check the last output
        self.x_multiplier /= self.last_output.item()
        self.y_multiplier /= self.last_output.item()

        # Draw the predator
        pygame.draw.circle(SCREEN, self.color, (self.x, self.y), 5)

        # Return the data
        return data_tensor

    # Chekc if the predator is colliding with anything
    def is_colliding(self, prey, predators):
        len_prey0: int = len(prey)
        len_pred0: int = len(predators)
        predators = self.multiply(predators)

        for p in prey:
            if abs(self.x - p.x) < 10 and abs(self.y - p.y) < 10:
                prey.remove(p)
                self.food += 1
                self.moves_left = 10**3

        output: int = (1 if len(prey) != len_prey0 or len(
            predators) != len_pred0 else 0)
        return prey, predators, torch.tensor([output]).float()


    # Check if the predator can multiply itself
    def multiply(self, predators):
        if self.food > 7:
            predators.append(Predator.new())
            self.food = 0
        return predators

    
    # Create a new predator
    @staticmethod
    def new():
        x: int = random.randint(50, 750)
        y: int = random.randint(50, 550)
        speed: int = random.randint(1, 5)
        return Predator(x, y, speed)
