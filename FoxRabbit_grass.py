from tabnanny import check
import pygame 
import random
from pygame.math import Vector2
from vi import Agent, Simulation, probability
from vi.config import Config
import polars as pl
import seaborn as sns
from scipy import stats
from scipy.stats import ranksums
from numpy import fix
import matplotlib.pyplot as plt

class Fox(Agent):
    timer = 0
    energy = 10
    max_energy = 20
    reprod_prob = 0.1
    nr_foxes = 10


    def update(self): 
        if self.energy <= 0:
            self.kill()
            Fox.nr_foxes -= 1
        elif self.energy < self.max_energy: #hungry 
            self.eat()
        
        # energy decrease takes place every 60 frames (which is 1 second)
        if self.timer < 60: 
            self.timer += 1
        else:
            self.energy_decrease()
            self.timer = 0

        self.save_data("nr_rabbits", Rabbit.nr_rabbits)
        self.save_data("nr_foxes", Fox.nr_foxes)
        self.save_data("kind", 0)

    #if hungry check for food
    def eat(self):
        if self.in_proximity_accuracy().count() > 0:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Rabbit):
                    agent.kill()
                    self.replenish()
                    self.fox_reprod()
                    Rabbit.nr_rabbits -= 1

    def replenish(self):
        self.energy = self.max_energy

    def fox_reprod(self): #use to reproduce
        if probability(self.reprod_prob):
            self.reproduce()
            Fox.nr_foxes += 1
    
    def energy_decrease(self):
        self.energy -= 0.5

class Grass(Agent):
    timer = 0 
    fully_grown = True
       
    def update(self):
            
        if self.fully_grown == False:
            self.timer+=1
            if self.timer >= 180:
                self.fully_grown == True
                self.change_image(0)
                self.timer=0
            
        self.save_data("nr_rabbits", Rabbit.nr_rabbits)
        self.save_data("nr_foxes", Fox.nr_foxes)
        self.save_data("kind", 2)

        
    def eaten(self):
        self.fully_grown = False
        self.change_image(1)

    def on_spawn(self):
        self.pos = Vector2(random.uniform(50, 700),random.uniform(50, 700)) 
        self.freeze_movement()
          
class Rabbit(Agent):
    timer1 = 0
    timer2 = 0
    
    energy = 15
    hungry_threshold = 20
    max_energy = 60

    reprod_prob = 0.4
    nr_rabbits = 15

    def update(self):
        #check for reproduction every 5 seconds (300 frames)
        if self.timer1 < 300:
            self.timer1 += 1
        else:
            self.rabbit_reprod()
            self.timer1 = 0
        
        self.save_data("nr_rabbits", Rabbit.nr_rabbits)
        self.save_data("nr_foxes", Fox.nr_foxes)
        self.save_data("kind", 1)

        if self.energy <= 0:
            self.kill()
            Rabbit.nr_rabbits -= 1
        elif self.energy < self.hungry_threshold: #hungry 
            self.eat()
            
        # energy decrease takes place every 60 frames (which is 1 second)
        if self.timer2 < 60: 
            self.timer2 += 1
        else:
            self.energy_decrease()
            self.timer2 = 0


    def rabbit_reprod(self): #use to reproduce
        if probability(self.reprod_prob):
            self.reproduce()
            Rabbit.nr_rabbits += 1

    def replenish(self):
        self.energy = self.max_energy
        
    def eat(self):
        if self.in_proximity_accuracy().count() > 0:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Grass):
                    if agent.fully_grown == True:
                        agent.eaten()
                        self.replenish()
                        self.rabbit_reprod()
    
    def energy_decrease(self):
        self.energy -= 0.5
    
x, y = Config().window.as_tuple() 
df=(
    Simulation(Config(radius=10, movement_speed=1.5))
    .batch_spawn_agents(   
        15,
        Rabbit,  # 👈 use our own MyAgent class
        images=["images/rabbit_face.png"],
        
    )
    .batch_spawn_agents(
        10,
        Fox,  # 👈 use our own MyAgent class
        images=["images/fox_face.png"],
    )
    .batch_spawn_agents(
        15,
        Grass,  # 👈 use our own MyAgent class
        images=["images/grass.png", "images/grass_eaten.png"],    
    )
    .run()
    .snapshots.rechunk()
    .groupby(['frame', 'kind', 'nr_rabbits', 'nr_foxes'])
    .agg(pl.count("id").alias('agents'))
    .sort(['frame'])
)

df = df.to_pandas()
df["agents"] = df["agents"]-15 #subtract number of grass agents because we do not regard them as "agents"
   
# plot = sns.relplot(x=df["frame"], y=df["agents"], hue=df["kind"],kind="line",legend=False)
# plt.legend(labels=["Foxes", "Rabbits"], title="Population sizes", loc = 2, bbox_to_anchor = (1,1))
# plot.savefig('grass.png', dpi=300)
ax = plt.gca() 
df.plot(kind = 'line',x = 'frame', y = 'nr_rabbits', color = 'blue',ax = ax)
df.plot(kind = 'line',x = 'frame', y = 'nr_foxes', color = 'red',ax = ax)
plt.show()
df.plot()
# plot = sns.relplot(x=df['frame'], y=df['agents'], hue=df["image_index"], kind='line')
# plot.savefig('convergence1.png', dpi=300)