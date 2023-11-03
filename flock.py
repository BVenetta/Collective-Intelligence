from distutils.command.config import config
from enum import Enum, auto
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation, probability, Window
from vi.config import Config, dataclass, deserialize
from random import uniform
from tabnanny import check
from numpy import fix
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

@deserialize
@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 2
    cohesion_weight: float = 4.5
    separation_weight: float = 4

    # weight for randomness
    randomWeight: float = 0.5

    # maximum velocity
    maxVelocity = 2

    delta_time: float = 3

    mass: int = 20

    def weights(self) -> tuple[float, float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight, self.randomWeight)


class Rabbit(Agent):
    config: FlockingConfig
    timer = 0
    reprod_prob = 0.5
    reprod_times = 0
    nr_rabbits = 15 

    def update(self):
        #check for reproduction every 5 seconds (300 frames)
        if self.timer < 300:
            self.timer += 1
        else:
            self.rabbit_reprod()
            self.timer = 0
        self.save_data("nr_rabbits", Rabbit.nr_rabbits)
        self.save_data("nr_foxes", Fox.nr_foxes)
        self.save_data("kind", 1)

    def rabbit_reprod(self): #use to reproduce
        if probability(self.reprod_prob) and self.reprod_times <= 1 :
            self.reprod_times += 1
            child = self.reproduce()
            child.move = Vector2(5,5)
            Rabbit.nr_rabbits += 1

    def change_position(self):
        # Pac-man-style teleport to the other end of the screen when trying to escape
        self.there_is_no_escape()
        
        alignmentWeight, separationWeight, cohesionWeight, randomWeight = self.config.weights()

        inProximity = []

        for agent, distance in self.in_proximity_accuracy():
            if agent.move != self.move:
                inProximity.append((agent, distance))
        
        #inProximity = list(self.in_proximity_accuracy())

        if len(inProximity) == 0:
            pass

        else:
            totalVeocity = Vector2()
            totalDifference = Vector2()
            totalPosition = Vector2()

            for boid, distance in inProximity:
                totalVeocity += boid.move
                totalDifference += (self.pos - boid.pos)
                totalPosition += boid.pos

            averageVelocity = totalVeocity/len(inProximity)
            averageDifference = totalDifference/len(inProximity)
            averagePosition = totalPosition/len(inProximity)
            cohesionForce = averagePosition - self.pos

            alignment = (averageVelocity - self.move).normalize()
            separation = averageDifference.normalize()
            cohesion = (cohesionForce - self.move).normalize()

            randomNoise = Vector2(uniform(-1,1), uniform(-1,1))

            totalForce = (alignmentWeight*alignment + separationWeight*separation + cohesionWeight*cohesion + randomWeight*randomNoise)/self.config.mass

            self.move += totalForce
        
        if self.move.length() > self.config.maxVelocity:
            self.move = self.move.normalize() * self.config.maxVelocity

        self.pos += self.move * self.config.delta_time


class Fox(Agent):
    config: Config

    timer = 0
    nr_foxes = 10
    energy = 10
    max_energy = 20
    chasing_radius = 15
    maxVelocity = 1.7
    minVelocity = 1.0
    reprod_times = 0

    def update(self): 
        #check energy level, if <=0, kill
        if self.move.length() >= self.maxVelocity:
            self.move = 0.53*(self.move)
        if self.move.length() <= self.minVelocity:
            if self.move.length() == 0:
                self.move = Vector2(0.3,0.4)
            else: 
                self.move = 1.53*(self.move)

        if self.energy <= 0:
            self.kill()
            Fox.nr_foxes -= 1
        elif self.energy < self.max_energy: #hungry 
            self.eat()
            self.chase()
        
        if self.energy< 5:
            self.chasing_radius = 30
        else: 
            self.chasing_radius = 15

        # energy decrease takes place every 60 frames (which is 1 second)
        if self.timer < 60: 
            self.timer += 1
        else:
            self.energy_decrease()
            self.timer = 0

        self.save_data("nr_rabbits", Rabbit.nr_rabbits)
        self.save_data("nr_foxes", Fox.nr_foxes)
        self.save_data("kind", 0)

    def chase(self):
        if self.in_proximity_accuracy().count() > 0:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Rabbit) and distance >7 and distance< self.chasing_radius:
                    x1, y1 = self.pos
                    x2 , y2 = agent.pos
                    s3, s4 = agent.move
                    loc_rabbit_x, loc_rabbit_y = Vector2(x2+s3,y2+s4)
                    self.move = 0.25*Vector2(loc_rabbit_x-x1,loc_rabbit_y-y1)
                    self.pos += self.move

    #if hungry check for food
    def eat(self):
        if self.in_proximity_accuracy().count() > 0:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Rabbit) and distance <= 20:
                    agent.kill()
                    x,y = self.move
                    self.move = 0.6*Vector2(x ,y)
                    self.replenish()
                    self.fox_reprod()
                    Rabbit.nr_rabbits -= 1
                    # print("ate")
                    
    def replenish(self):
        self.energy += 0.25

    def fox_reprod(self): #use to reproduce
        if probability(1/self.energy) and self.reprod_times <= 3 :
            self.reproduce()
            Fox.nr_foxes += 1
            self.reprod_times += 1

    def energy_decrease(self):
        self.energy -= 0.5



class Selection(Enum):
    ALIGNMENT = auto()
    COHESION = auto()
    SEPARATION = auto()


class FlockingLive(Simulation):
    selection: Selection = Selection.ALIGNMENT
    config: FlockingConfig

    def handle_event(self, by: float):
        if self.selection == Selection.ALIGNMENT:
            self.config.alignment_weight += by
        elif self.selection == Selection.COHESION:
            self.config.cohesion_weight += by
        elif self.selection == Selection.SEPARATION:
            self.config.separation_weight += by

    def before_update(self):
        super().before_update()

        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.handle_event(by=0.1)
                elif event.key == pg.K_DOWN:
                    self.handle_event(by=-0.1)
                elif event.key == pg.K_1:
                    self.selection = Selection.ALIGNMENT
                elif event.key == pg.K_2:
                    self.selection = Selection.COHESION
                elif event.key == pg.K_3:
                    self.selection = Selection.SEPARATION

        a, c, s, r = self.config.weights()
        # print(f"A: {a:.1f} - C: {c:.1f} - S: {s:.1f} - R: {r:.1f}")


df = (
    FlockingLive(
        FlockingConfig(
            image_rotation=True,
            movement_speed=1,
            radius=50,
            seed=1,
            fps_limit = 60
        )
    )
    .batch_spawn_agents(15, Rabbit, images=["images/white.png"])
    .batch_spawn_agents(10, Fox, images=["images/fox_face.png"])
    .run()
    .snapshots.rechunk()
    .groupby(['frame', 'kind', 'nr_rabbits', 'nr_foxes'])
    .agg(pl.count("id").alias('agents'))
    .sort(['frame', 'kind'])
)


df = df.to_pandas()

ax = plt.gca() 
df.plot(kind = 'line',x = 'frame', y = 'nr_rabbits', color = 'blue',ax = ax)
df.plot(kind = 'line',x = 'frame', y = 'nr_foxes', color = 'red',ax = ax)
plt.show()
df.plot()