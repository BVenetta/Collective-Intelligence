from tabnanny import check
from numpy import fix
import matplotlib.pyplot as plt
import random
from pygame.math import Vector2
from vi import Agent, Simulation, probability
from vi.config import Config
import polars as pl
import seaborn as sns
import math

class Fox(Agent):

    timer = 0
    energy = 18
    max_energy = 16
    age = 0
    max_age = 18

    reprod_prob = 0.685
    nr_foxes = 13
    eaten = 0
    maxVelocity = 0.7
    minVelocity = 0.4
    chasing_radius = 15

    def update(self): 
        #check energy level, if <=0, kill
        if self.move.length() >= self.maxVelocity:
            self.move = 0.53*(self.move)
        if self.move.length() <= self.minVelocity:
            if self.move.length() == 0:
                self.move = Vector2(0.3,0.4)
            else: 
                self.move = 1.53*(self.move)
        if self.energy <= 0 or self.age >= self.max_age:
            self.kill()
            Fox.nr_foxes -= 1
        elif self.energy < self.max_energy : #hungry 
            self.chase() 
            self.eat()
        if self.energy < 5:
            self.chasing_radius = 30
        else: 
            self.chasing_radius = 15
        # energy decrease takes place every 60 frames (which is 1 second)
        if self.timer < 60: 
            self.timer += 1
        else:
            self.age += 1
            self.energy_decrease()
            self.timer = 0

        # if Fox.nr_foxes > 1.5*Rabbit.nr_rabbits:
        #     Fox.reprod_prob = 0.1
        # if Fox.nr_foxes <= Rabbit.nr_rabbits:
        #     Fox.reprod_prob = 0.5

        self.save_data("nr_rabbits", Rabbit.nr_rabbits)
        self.save_data("nr_foxes", Fox.nr_foxes)
        self.save_data("kind", 0)

    #if hungry check for food
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

    def eat(self):
        if self.in_proximity_accuracy().count() > 0:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Rabbit) and distance < 7:
                    agent.kill()
                    x,y = self.move
                    self.move = 0.6*Vector2(x ,y)
                    self.eaten += 1
                    self.replenish()
                    self.fox_reprod()
                    Rabbit.nr_rabbits -= 1
                    #print((1-1/self.eaten) - (1/self.energy))
                    
    def replenish(self):
        self.energy += 1

    def fox_reprod(self): #use to reproduce
        if probability((1-1/self.eaten) - (1/self.energy)):
            self.reproduce()
            Fox.nr_foxes += 1

    def energy_decrease(self):
        self.energy -= 0.5

class Rabbit(Agent):

    timer = 0
    energy = 25 #for extras

    reprod_prob = 0.5
    reprod_times = 0
    nr_rabbits = 18
    age = 0 
    max_age = 15

    def update(self):
        #check for reproduction every 5 seconds (300 frames)
        
        if self.timer < 300:
            self.timer += 1

        else:
            self.age +=5
            self.rabbit_reprod()
            self.timer = 0

        if self.age >= self.max_age:
            self.kill()
            Rabbit.nr_rabbits -= 1

        # if Rabbit.nr_rabbits > 2*Fox.nr_foxes:
        #     Rabbit.reprod_prob = 0.1
        # if Rabbit.nr_rabbits < Fox.nr_foxes:
        #     Rabbit.reprod_prob = 0.5

        self.save_data("nr_rabbits", Rabbit.nr_rabbits)
        self.save_data("nr_foxes", Fox.nr_foxes)
        self.save_data("kind", 1)

    def rabbit_reprod(self): #use to reproduce
        if probability(self.reprod_prob) and self.reprod_times <= 1 :
            self.reproduce()
            Rabbit.nr_rabbits += 1
            self.reprod_times =+ 1
        

x, y = Config().window.as_tuple() 

df=(
    Simulation(Config(radius=30, movement_speed=0.5))
    .batch_spawn_agents(
        20,
        Rabbit,  # 👈 use our own MyAgent class
        images=["C:/Users/ohadd/Desktop/vu/2nd year/project collective int/rabbit_face.png"],
    )
    .batch_spawn_agents(
        20,
        Fox,  # 👈 use our own MyAgent class
        images=["C:/Users/ohadd/Desktop/vu/2nd year/project collective int/fox_face.png"],
    )
    .run()
    .snapshots.rechunk()
    #.groupby(['frame', 'nr_rabbits', 'nr_foxes'])
    .groupby(['frame', 'kind'])
    .agg(pl.count("id").alias('agents'))
    .sort(['frame', 'kind'])
)

#print(df)

plot = sns.relplot(x=df["frame"], y=df["agents"], hue=df["kind"],kind="line",legend=False)
plt.legend(labels=["Foxes", "Rabbits"], title="Population sizes", loc = 2, bbox_to_anchor = (1,1))
plot.savefig('plot_chaser_with_age_5.png', dpi=300)