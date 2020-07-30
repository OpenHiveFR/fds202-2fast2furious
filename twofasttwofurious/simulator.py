#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame
import esper
from math import *
from pygame.draw import polygon
from typing import *

import enum

FPS = 60
RESOLUTION = 720, 480
CAR_POLYGON = [
    (-1,-1),
    (1,-1),
    (0,2)
]

CHECKPOINT_POLYGON = [
    (-3,-1),
    (-3,1),
    (3,1),
    (2,-1),
]

class Direction(enum.Enum):
    EAST = 0/360*2*3.14
    NORTH = 90/360*2*3.14
    WEST = 180/360*2*3.14
    SOUTH = 270/360*2*3.14

MAP_SAVE ={
    "checkpoint":
    [
        {
            "x":100,
            "y":100, 
            "scale":10, 
            "angle": Direction.NORTH.value,
            "UUID": 0
        },
        {
            "x":200,
            "y":100, 
            "scale":10,
            "angle": Direction.EAST.value,
            "UUID": 1
        },
        {
            "x":300,
            "y":100, 
            "scale":10,
            "angle": Direction.SOUTH.value,
            "UUID": 2
        },
        {
            "x":400,
            "y":100, 
            "scale":10,
            "angle": Direction.WEST.value,
            "UUID": 3
        }
    ],
    "driver":
    [
        {
            "x":100,
            "y":100, 
            "scale":5,
            "angle": Direction.NORTH.value,
            "start": 0
        }
    ]
}

CONFIG = {
    "driver":{
        "ACC":0.1,
        "DEC":0.5,
        "TURN":0.1
    },
    "slowdown":{
        "LOW":0.1,
        "MEDIUM":0.4,
        "HARD":0.5,
        "EXTREME":0.7,
        "NONE":0
    },
    "SPEED":{
        "ALPHA":1.006,
        "MAX":345
    },
    "RACE":{
        "LAPS":1,
        "MAXTIME":30
    }
}

##################################
#  Define some Components:
##################################
class Velocity:
    def __init__(self, x=0.0, y=0.0, polar=True):
        self.x = 0
        self.y = 0
        self.speed = 0 
        self.angle = 0
        if polar:
            self.setPolar(x, y)
        else:
            self.setCoord(x, y)

    def polarToCoord(self):
        return (self.speed*cos(self.angle), self.speed*sin(self.angle))

    def coordToPolar(self):
        return ((x**2 + y**2)**0.5, atan2(y, x))

    def setPolar(self, r, angle):
        self.speed = r
        self.angle = angle % (2*3.14)
        self.x, self.y = self.polarToCoord()        
        return self

    def setCoord(self, x, y):
        self.x = x
        self.y = y 
        self.speed, self.angle = self.coordToPolar()
        return self
    
    def addSpeed(self, acc):
        self.setPolar(min(self.speed+acc, CONFIG["SPEED"]["MAX"]), self.angle)
        return self

    def addAngle(self, acc):
        self.setPolar(self.speed, self.angle+acc)
        return self

    def addX(self, acc):
        self.setCoord(self.x+acc, self.y)
        return self

    def addY(self, acc):
        self.setCoord(self.x, self.y+acc)
        return self

    def actualSpeed(self):
        if self.speed == 0:
            return 0
        return CONFIG["SPEED"]["MAX"] * (1-1/(CONFIG["SPEED"]["ALPHA"]**abs(self.speed))) * (self.speed/abs(self.speed))


class Model:
    def __init__(self, points, gravityCenter=(0,0), angle=0, scale=10, width=0, x=0, y=0):
        self.points = points
        self.gravityCenter = gravityCenter
        self.angle = angle
        self.width = width
        self.scale = scale
        self.x = x
        self.y = y
    
    def actual_model(self):
        dist = lambda coord: ((coord[0]-self.gravityCenter[0])**2 + (coord[1]-self.gravityCenter[1])**2)**0.5
        theta = lambda coord: atan2(coord[1], coord[0])
        # scaling on map
        scaled = list(map(lambda coord: (coord[0]*self.scale, coord[1]*self.scale),self.points))
        # rotation on map
        rotated = list(map(lambda coord: (dist(coord)*sin(self.angle-theta(coord)),dist(coord)*cos(self.angle-theta(coord))), scaled)) # rotation
        # translation on map
        translated = list(map(lambda coords: (coords[0]+self.x, coords[1]+self.y), rotated)) # translation
        return translated

    def render(self, window, color=(255, 0, 0)):
        # rotation for camera

        # translation for camera

        # actual render            
        polygon(window, color, self.actual_model(), self.width)

class Driver:
    def __init__(self):
        super().__init__()
        self.isAccelerating = False
        self.isDecelerating = False
        self.isTurningLeft = False
        self.isTurningRight = False

class Slowdown:
    def __init__(self):
        super().__init__()
        self.cycleSteps:List[str] = list(CONFIG["slowdown"].keys())
        self.currentStep:int = 0
        self.power = 0 
        self.setCycle(self.currentStep)
    
    def setCycle(self, step:int):
        self.power = CONFIG["slowdown"][self.cycleSteps[step]]
        self.currentStep = step
        
    def nextCycle(self):
        self.setCycle((self.currentStep+1) % len(self.cycleSteps))

class Renderable:
    def __init__(self, posx, posy, depth=0, width=0):
        self.depth = depth
        self.width = width
        self.x = posx
        self.y = posy


class Camera:
    def __init__(self):
        self.x = x
        self.y = y 
        self.FOV = FOV 

class Checkpoint:
    def __init__(self, UUID):
        self.UUID = UUID

class Named:
    def __init__(self, name):
        self.name = name

class Racer:
    def __init__(self, goal):
        self.goal = goal
        self.onCheckpoint = None
################################
#  Define some Processors:
################################
class MovementProcessor(esper.Processor):
    def __init__(self, minx, maxx, miny, maxy):
        super().__init__()
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

    def process(self):
        # This will iterate over every Entity that has BOTH of these components:
        for ent, (vel, model, slow) in self.world.get_components(Velocity, Model, Slowdown):
            newx = -vel.actualSpeed()*cos(vel.angle)
            newy = vel.actualSpeed()*sin(vel.angle)
            # Update the Renderable Component's position by it's Velocity:
            model.x += newx
            model.y += newy
            model.angle = vel.angle
            model.angle %= 2* 3.14
            # An example of keeping the sprite inside screen boundaries. Basically,
            # adjust the position back inside screen boundaries if it tries to go outside:
            model.x = max(self.minx, model.x)
            model.y = max(self.miny, model.y)
            model.x = min(self.maxx, model.x)
            model.y = min(self.maxy, model.y)

class RenderProcessor(esper.Processor):
    def __init__(self, window, clear_color=(0, 0, 0)):
        super().__init__()
        self.window = window
        self.clear_color = clear_color
        self.font = pygame.font.Font(None, 35)

    def process(self):
        # Clear the window:
        self.window.fill(self.clear_color)
        # Entities
        for ent, (model, checkpoint) in self.world.get_components(Model, Checkpoint): # checkpoints
            model.render(self.window, (100, 100, 255))
            self.window.blit(self.font.render(f"{checkpoint.UUID}", True, (220,220,255)), (model.x, model.y))
        for ent, (model, driver) in self.world.get_components(Model, Driver): # player
            model.render(self.window, (255, 255, 255))
        # UX
        i = 0
        for ent, vel in self.world.get_component(Velocity):
            self.window.blit(self.font.render(f"entity {ent} speed: {round(vel.actualSpeed(),2)} km/h", True, (255,255,255)), (0, i*100))
            i+=1
        # Flip the framebuffers
        pygame.display.flip()

class DriverProcessor(esper.Processor):
    def __init__(self):
        super().__init__()
        self.config = CONFIG["driver"]
    
    def process(self):
        for ent, (vel, driver) in self.world.get_components(Velocity, Driver):
            # speed formula when accelerating: (1 - (1 / (alpha * speed))) * maxspeed
            if driver.isAccelerating and not driver.isDecelerating:
                vel.addSpeed(self.config["ACC"])
            if driver.isDecelerating and not driver.isAccelerating:
                if vel.speed > 0:
                    vel.addSpeed(-self.config["DEC"])
                else:
                    vel.addSpeed(-self.config["ACC"])
            if driver.isTurningLeft and not driver.isTurningRight:
                vel.addAngle(self.config["TURN"])
            elif driver.isTurningRight and not driver.isTurningLeft:
                vel.addAngle(-self.config["TURN"])

class SlowdownProcessor(esper.Processor):
    def __init__(self):
        super().__init__()
        self.config = CONFIG["slowdown"]
    
    def process(self):
        for ent, (vel, slowdown) in self.world.get_components(Velocity, Slowdown):
            if vel.speed != 0:
                speed_percentage = abs(vel.speed)/CONFIG["SPEED"]["MAX"]
                speed_sign = (abs(vel.speed)/vel.speed)
                acceleration_value = CONFIG["driver"]["ACC"]
                slowdown_value = slowdown.power * max(0.8, speed_percentage) * acceleration_value
                vel.addSpeed(min(abs(vel.speed), slowdown_value)*-speed_sign)

class Collision2DProcessor(esper.Processor):
    def __init__(self):
        super().__init__()
    
    def process(self):
        for ent, (vel, model, named) in self.world.get_components(Velocity, Model, Named):
            for ent2, (model2, named2) in self.world.get_components(Model, Named):
                if ent == ent2:
                    continue
                if collidePolyPoly(model.actual_model(), model2.actual_model()):
                    #print(f"[collision] {named.name} | {named2.name}")
                    pass
        # RACING COLLISION: RACER x Checkpoint du brute force pour le moment, la flemme d opti
        for ent, (vel, model, racer) in self.world.get_components(Velocity, Model, Racer):
            onCheckpoint = None
            for ent2, (model2, checkpoint) in self.world.get_components(Model, Checkpoint):
                if ent == ent2:
                    continue
                if not collidePolyPoly(model.actual_model(), model2.actual_model()):
                    continue
                onCheckpoint = checkpoint

                if racer.onCheckpoint is not None:
                    break

                if checkpoint.UUID == racer.goal:
                    #trigger checkpoint reached.
                    self.world.get_processor(GoalProcessor).updateGoal(ent, racer)
            if onCheckpoint is None:
                racer.onCheckpoint = None
                

class GoalProcessor(esper.Processor):
    def __init__(self, goalOrder, lapGoal):
        super().__init__()
        self.goalOrder:Dict[int, int] = goalOrder
        self.lapGoal:int = lapGoal

    def process(self):
        pass
    
    def nextGoal(self, goal):
        return self.goalOrder[goal]

    def updateGoal(self, ent, racer):

        if racer.goal == self.lapGoal:
            # END GOAL
            if self.world.get_processor(LapProcessor).hasCurrentLap(ent): #on the start of any non first lap, we register the ending time of each lap before starting a new one
                self.world.get_processor(LapProcessor).addGoalTime(ent)
            # START GOAL
            self.world.get_processor(LapProcessor).newLap(ent)        
        
        self.world.get_processor(LapProcessor).addGoalTime(ent)
        racer.goal = self.nextGoal(racer.goal)

from collections import *
from time import time

class LapProcessor(esper.Processor):
    def __init__(self):
        self.lapTimes = defaultdict(list)

    def process(self):
        pass
    
    def hasCurrentLap(self, ent):
        return len(self.lapTimes[ent]) > 0

    def newLap(self, ent):
        #on marque le nouveau temps de tour
        self.lapTimes[ent].append([])
        if len(self.lapTimes[ent]) > CONFIG["RACE"]["LAPS"]:
            # end of race for entity
            self.world.get_processor(GameProcessor).finishedRace(ent, self.lapTimes[ent])
    
    def addGoalTime(self, ent):
        self.lapTimes[ent][-1].append(time())

def prettyLapTimes(lapsTimes):
    startTime = lapsTimes[0][0]
    return list(map(lambda goalTimes: list(map(lambda x: x-startTime, goalTimes)), lapsTimes))

class GameProcessor(esper.Processor):
    def __init__(self):
        self.startTime = time()

    def process(self):
        if self.hasGameEnded():
            lapProcessor = self.world.get_processor(LapProcessor) 
            for ent, racer in self.world.get_component(Racer):
                self.endRace(ent, lapProcessor.lapTimes[ent])

    def startRace(self):
        self.startTime = time()

    def finishedRace(self, ent, lapTimes):
        print(f"entity {ent} ended the race with the following times: {prettyLapTimes(lapTimes)}")
    
    def endRace(self, ent, lapTimes):
        print(f"entity {ent} ended up with {len(lapTimes)} laps with the following times : {prettyLapTimes(lapTimes)}")

    def hasGameEnded(self):
        time_elapsed = time() - self.startTime
        return time_elapsed > CONFIG["RACE"]["MAXTIME"]

################################
#  The main core of the program:
################################


def loadProcessors(world, window, goalOrder, lapGoal):
    # Render processing
    world.add_processor(RenderProcessor(window=window))
    # Movement processing
    world.add_processor(MovementProcessor(minx=0, maxx=RESOLUTION[0]*0.8, miny=0, maxy=RESOLUTION[1]*0.8))
    # Player driving processing    
    world.add_processor(DriverProcessor())
    # Terrain slowdown processing
    world.add_processor(SlowdownProcessor())
    # Collision detection
    world.add_processor(Collision2DProcessor())
    world.add_processor(GoalProcessor(goalOrder, lapGoal))
    world.add_processor(LapProcessor())
    world.add_processor(GameProcessor())


def loadPlayer(world, data):
    player = world.create_entity()
    world.add_component(player, Velocity())
    world.add_component(player, Model(CAR_POLYGON, x=data["x"], y=data["y"], angle=data["angle"], scale=data["scale"]))
    world.add_component(player, Driver())
    world.add_component(player, Slowdown())
    world.add_component(player, Named(f"PLAYER (ent {player})"))
    world.add_component(player, Racer(data["start"]))

def loadCheckpoint(world, data):
    checkpoint = world.create_entity()
    world.add_component(checkpoint, Model(CHECKPOINT_POLYGON, x=data["x"], y=data["y"], angle=data["angle"], scale=data["scale"]))
    world.add_component(checkpoint, Checkpoint(data["UUID"]))
    world.add_component(checkpoint, Named(f"CHECKPOINT {data['UUID']} (ent {checkpoint})"))

def loadMap(world, save:Dict[str,Iterable]):
    goalOrder = {}
    goals = []
    for element in save:
        for data in save[element]:
            if element == "driver":
                loadPlayer(world, data)
            elif element == "checkpoint":
                loadCheckpoint(world, data)
                goals.append(data["UUID"])
    sorted(goals)
    for i in range(len(goals)-1):
        goalOrder[goals[i]] = goals[i+1]
    goalOrder[goals[-1]] = goals[0] 
    return goalOrder, goals[0]


def run():
    # Initialize Pygame stuff
    pygame.init()
    window = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("Esper Pygame example")
    clock = pygame.time.Clock()
    pygame.key.set_repeat(1, 1)

    # Initialize Esper world, and create a "player" Entity with a few Components.
    world = esper.World()
    goalOrder, lapGoal = loadMap(world, MAP_SAVE)
    loadProcessors(world, window, goalOrder, lapGoal)
    players = world.get_component(Driver)
    player, _ = players[0]
    running = True
    game_processor = world.get_processor(GameProcessor)
    game_processor.startRace()
    while not game_processor.hasGameEnded() and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    world.component_for_entity(player, Driver).isTurningLeft = True
                elif event.key == pygame.K_RIGHT:
                    world.component_for_entity(player, Driver).isTurningRight = True
                elif event.key == pygame.K_UP:
                    world.component_for_entity(player, Driver).isAccelerating = True
                elif event.key == pygame.K_DOWN:
                    world.component_for_entity(player, Driver).isDecelerating = True
                elif event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    world.component_for_entity(player, Driver).isTurningLeft = False
                elif event.key == pygame.K_RIGHT:
                    world.component_for_entity(player, Driver).isTurningRight = False
                elif event.key == pygame.K_UP:
                    world.component_for_entity(player, Driver).isAccelerating = False
                elif event.key == pygame.K_DOWN:
                    world.component_for_entity(player, Driver).isDecelerating = False
                elif event.key == pygame.K_s:
                    world.component_for_entity(player, Slowdown).nextCycle()
        # A single call to world.process() will update all Processors:
        world.process()

        clock.tick(FPS)


##
# source: https://github.com/Wopple/GJK/blob/master/python/gjk.py

from math import sqrt

def add(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1])

def sub(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1])

def neg(v):
    return (-v[0], -v[1])

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def aXbXa(v1, v2):
    """
    Performs v1 X v2 X v1 where X is the cross product. The
    input vectors are (x, y) and the cross products are
    performed in 3D with z=0. The output is the (x, y)
    component of the 3D cross product.
    """

    x0 = v1[0]
    x1 = v1[1]
    x1y0 = x1 * v2[0]
    x0y1 = x0 * v2[1]
    return (x1 * (x1y0 - x0y1), x0 * (x0y1 - x1y0))

def supportPoly(polygon, direction):
    bestPoint = polygon[0]
    bestDot = dot(bestPoint, direction)

    for i in range(1, len(polygon)):
        p = polygon[i]
        d = dot(p, direction)

        if d > bestDot:
            bestDot = d
            bestPoint = p

    return bestPoint

def supportCircle(circle, direction):
    mag = sqrt(dot(direction, direction))
    s = circle[1] / mag
    center = circle[0]
    return (center[0] + s * direction[0], center[1] + s * direction[1])

def support(poly1, poly2, support1, support2, direction):
    return sub(support1(poly1, direction), support2(poly2, neg(direction)))

def collidePolyPoly(poly1, poly2):
    return collide(poly1, poly2, supportPoly, supportPoly)

def collidePolyCircle(poly, circle):
    return collide(poly, circle, supportPoly, supportCircle)

def collide(shape1, shape2, support1, support2):
    s = support(shape1, shape2, support1, support2, (-1, -1))
    simplex = [s]
    d = list(neg(s))

    for i in range(100):
        a = support(shape1, shape2, support1, support2, d)

        if dot(a, d) < 0:
            return False

        simplex.append(a)

        if doSimplex(simplex, d):
            return True

    raise RuntimeError("infinite loop in GJK algorithm")

def doSimplex(simplex, d):
    l = len(simplex)

    if l == 2:
        b = simplex[0]
        a = simplex[1]
        a0 = neg(a)
        ab = sub(b, a)

        if dot(ab, a0) >= 0:
            cross = aXbXa(ab, a0)
            d[0] = cross[0]
            d[1] = cross[1]
        else:
            simplex.pop(0)
            d[0] = a0[0]
            d[1] = a0[1]
    else:
        c = simplex[0]
        b = simplex[1]
        a = simplex[2]
        a0 = neg(a)
        ab = sub(b, a)
        ac = sub(c, a)

        if dot(ab, a0) >= 0:
            cross = aXbXa(ab, a0)

            if dot(ac, cross) >= 0:
                cross = aXbXa(ac, a0)

                if dot(ab, cross) >= 0:
                    return True
                else:
                    simplex.pop(1)
                    d[0] = cross[0]
                    d[1] = cross[1]
            else:
                simplex.pop(0)
                d[0] = cross[0]
                d[1] = cross[1]
        else:
            if dot(ac, a0) >= 0:
                cross = aXbXa(ac, a0)

                if dot(ab, cross) >= 0:
                    return True
                else:
                    simplex.pop(1)
                    d[0] = cross[0]
                    d[1] = cross[1]
            else:
                simplex.pop(1)
                simplex.pop(0)
                d[0] = a0[0]
                d[1] = a0[1]

    return False


##

if __name__ == "__main__":
    run()
    pygame.quit()
