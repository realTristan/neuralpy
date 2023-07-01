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
        self.color: tuple[int, int, int] = (0, 255, 0)
        self.last_output: float = 0.0

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
        
        closest_prey: float = -1.0
        closest_predator: float = -1.0
        
        for p in predators:
            if abs(self.x - p.x) + abs(self.y - p.y) < closest_predator or closest_predator == -1:
                closest_predator = abs(self.x - p.x) + abs(self.y - p.y)
        
        for p in prey:
            if p == self:
                continue
            
            # Calculate closest prey
            if abs(self.x - p.x) + abs(self.y - p.y) < closest_prey or closest_prey == -1:
                closest_prey = abs(self.x - p.x) + abs(self.y - p.y)


        # Model prediction
        data = [self.x, self.y, self.x_multiplier,
                self.y_multiplier, self.last_output, 
                closest_prey, closest_predator]
        data_tensor = torch.tensor([data]).float()
        self.last_output = PREY_MODEL(data_tensor)
        
        # Print the output
        print(self.last_output.item())
        
        # Check the last output
        if self.last_output >= 1:
            self.x_multiplier /= self.last_output.item()
            self.y_multiplier /= self.last_output.item()
            self.x -= self.x_multiplier
            self.y -= self.y_multiplier
        
        pygame.draw.circle(SCREEN, self.color, (self.x, self.y), 5)

        # Return the data
        return data_tensor

    def is_colliding_prey(self, predators, prey):
        len0: int = len(prey)
        is_eaten: int = 0
        
        for p in predators:
            if abs(self.x - p.x) < 10 and abs(self.y - p.y) < 10:
                is_eaten = 1
                break
                
        for p in prey:
            if p == self:
                continue
            
            if abs(self.x - p.x) < 10 and abs(self.y - p.y) < 10:
                prey.append(Prey.new())
                if random.choice([1, 1, 0, 0, 0, 0]):
                    if self not in prey:
                        continue
                    prey.remove(self)
                    break
        
        return prey, torch.tensor(
            0 if len(prey) != len0 or is_eaten == 0 else 1
        ).unsqueeze(0).float()

    @staticmethod
    def new():
        x: int = random.randint(50, 750)
        y: int = random.randint(50, 550)
        speed: int = random.randint(1, 5)
        return Prey(x, y, speed)
