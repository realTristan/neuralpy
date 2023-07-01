import pygame, random, torch
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
        #data = [self.x, self.y, self.x_multiplier, self.y_multiplier, self.moves_left]
        #data_tensor = torch.tensor(data).unsqueeze(0).float()
        #output: int = PREDATOR_MODEL(data_tensor)
        #if output == 1 or output == 2:
        pygame.draw.circle(SCREEN, self.color, (self.x, self.y), 5)
        #else:
        #    self.x -= self.x_multiplier
        #    self.y -= self.y_multiplier
        
        # Return the data
        #return data_tensor

    def is_colliding_prey(self, prey):
        for p in prey:
            if abs(self.x - p.x) < 10 and abs(self.y - p.y) < 10:
                prey.remove(p)
                self.food += 1
                self.moves_left = 10**3
        return prey

    def multiply(self, predators):
        if self.food > 7:
            predators.append(Predator.new())
            self.food = 0
        return predators

    @staticmethod
    def new():
        x: int = random.randint(50, 750)
        y: int = random.randint(50, 550)
        speed: int = random.randint(1, 5)
        return Predator(x, y, speed)
