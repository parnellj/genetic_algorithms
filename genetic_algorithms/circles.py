import pygame
import os, sys
import random as rnd
import numpy as np





POPULATION = 100
MUTATION_RATE = 0.003
CROSSOVER_RATE = 0.7
STATIC_CIRCLES = 50

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

WIDTH, HEIGHT = 1600, 900
RAD_MIN, RAD_MAX = 10, 450
os.environ['SDL_VIDEO_CENTERED'] = '1'
screen = pygame.display.set_mode((WIDTH, HEIGHT))

class Circle():
    def __init__(self, mother=None, father=None, r_min=RAD_MIN, r_max=RAD_MAX):
        self.r_min=r_min
        self.r_max=r_max
        self.r, self.x, self.y = 0, 0, 0
        if mother is None or father is None:
            self.r = rnd.randint(self.r_min, self.r_max)
            self.x = rnd.randint(self.r, WIDTH - self.r)
            self.y = rnd.randint(self.r, WIDTH - self.r)
        elif rnd.random() <= CROSSOVER_RATE:
            self.x, self.y, self.r  = mother.crossover(father)
        else:
            inheritance = rnd.choice([mother, father])
            self.x, self.y, self.r = inheritance.x, inheritance.y, inheritance.r

        viable = False
        while not viable:
            self.mutate()
            self.pos = [self.x, self.y]
            self.bounding_box = [self.x - self.r, self.y - self.r,
                                 self.r * 2, self.r * 2]
            if self.bounding_box[0] < 0 or self.bounding_box[1] < 0:
                continue
            if self.bounding_box[0] + self.bounding_box[2] > WIDTH or \
                self.bounding_box[1] + self.bounding_box[3] > HEIGHT:
                continue
            viable = True

        self.area = np.pi * self.r * self.r
        self.collision_area = 0
        self.collision_count = 0
        self.fitness = None

    def crossover(self, other):
        x = rnd.choice([self.x, other.x])
        y = rnd.choice([self.y, other.y])
        r = rnd.choice([self.r, other.r])
        return x, y, r

    def mutate(self, force=False):
        if rnd.random() <= MUTATION_RATE or force:
            self.r = rnd.randint(self.r_min, self.r_max)
        if rnd.random() <= MUTATION_RATE or force:
            self.x = rnd.randint(self.r, WIDTH - self.r)
        if rnd.random() <= MUTATION_RATE or force:
            self.y = rnd.randint(self.r, HEIGHT - self.r)

class Core(object):
    def __init__(self, surface, name):
        pygame.display.set_caption(name)
        self.screen = surface
        self.clock = pygame.time.Clock()
        self.advance = False

    def dispatch(self, event):
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.advance = True

    def run(self):
        static_circles = generate_circles(STATIC_CIRCLES, False, r_min=10, r_max=50)
        cur_gen = None
        gen_number = 0
        while True:
            for event in pygame.event.get():
                self.dispatch(event)
            self.screen.fill([0xFF, 0xFF, 0xFF])
            for c in static_circles:
                pygame.draw.circle(self.screen, BLACK, c.pos, c.r)

            if cur_gen is None:
                next_gen = [Circle() for _ in xrange(0, POPULATION)]
            else:
                next_gen = reproduce(cur_gen, POPULATION)
                gen_number += 1
            for c in next_gen:
                c.collision_area = 0
                collisions = find_collisions(c, static_circles)
                if collisions:
                    c.collision_area = sum([a['collision_area'] for a in collisions])
                    c.collision_count = len(collisions)
            calculate_fitness(next_gen)
            cur_gen = next_gen
            mean_fitness = np.mean([c.fitness for c in cur_gen])
            max_fitness = np.max([c.fitness for c in cur_gen])
            best = cur_gen[[c.fitness for c in cur_gen].index(max_fitness)]

            # s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            # s.fill((255, 255, 255, 128))
            # self.screen.blit(s, (0, 0))

            s = pygame.Surface((WIDTH, HEIGHT))
            s.set_alpha(128)
            for i in cur_gen:
                pygame.draw.circle(s, RED, i.pos, i.r, 1)
            pygame.draw.circle(s, RED, best.pos, best.r)
            self.screen.blit(s, (0, 0))
            pygame.display.flip()

def calculate_fitness(population):
    for p in population:
        size_score = p.area / (np.pi * RAD_MAX ** 2)
        collision_score = 1 - p.collision_area / (np.pi * RAD_MAX ** 2)
        collision_count_score = (STATIC_CIRCLES + 1) / (p.collision_count + 1)
        p.fitness = (size_score * 10) +\
                    (collision_score * 0.0) + \
                    (collision_count_score * 1)
    return

def reproduce(generation, size):
    raw_fitness = np.array([c.fitness for c in generation])
    rel_fitness = raw_fitness / raw_fitness.sum()
    new_generation = []
    for _ in xrange(0, size):
        mother = None
        father = None
        while mother == father:
            mother = np.random.choice(generation, p=rel_fitness)
            father = np.random.choice(generation, p=rel_fitness)
        new_generation.append(Circle(mother, father))
    return new_generation

def show_collisions(screen, collisions):
    """
    A nice  testing function that demonstrates my approach to measuring the amount of
    overlap between two circles: find the clipping area for the bounding boxes
    describing each circle.
    :param screen:
    :type screen:
    :param objects:
    :type objects:
    :return:
    :rtype:
    """
    for c in collisions:
        pygame.draw.rect(screen, BLUE, c['clip_mask'], 1)

def get_clip_mask(a, b):
    rect_a = pygame.Rect(a.bounding_box)
    rect_b = pygame.Rect (b.bounding_box)
    return rect_a.clip(rect_b)

def find_collisions(target, population):
    """
    Find all collisions between a target circle and a population of circles.
    :param target:
    :type target:
    :param population:
    :type population:
    :return:
    :rtype:
    """
    collisions = []
    for o in population:
        center_distance = np.linalg.norm(np.array(o.pos) - np.array(target.pos))
        radii_sum = o.r + target.r
        radii_difference = abs(o.r - target.r)
        if center_distance <= radii_difference:
            collisions.append({'a': o, 'b': target,
                               'clip_mask': get_clip_mask(o, target)})
        if radii_difference < center_distance < radii_sum:
            collisions.append({'a': o, 'b': target,
                               'clip_mask': get_clip_mask(o,target)})
        o_bb = np.array(o.bounding_box)
        target_bb = np.array(target.bounding_box)

    for c in collisions:
        c['collision_area'] = c['clip_mask'].size[0] * c['clip_mask'].size[1]
    return collisions

def generate_circles(count=10, allow_collision=False, r_min=RAD_MIN, r_max=RAD_MAX):
    circles = []
    while len(circles) < count:
        new = Circle(r_min=r_min, r_max=r_max)
        if allow_collision:
            circles.append(new)
            continue
        elif not find_collisions(new, circles):
            circles.append(new)

    return circles

if __name__ == '__main__':
    main = Core(screen, 'Node')
    main.run()
