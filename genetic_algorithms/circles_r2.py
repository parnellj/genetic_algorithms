import os
import random
import sys

import numpy as np
import pygame

"""
POPULATION
    The number of members in each generation

MUTATION_RATE
    The rate at which chromosomes are selected for mutation

CROSSOVER_RATE
    The rate at which chromosomes are reproduced by "crossing-over" the genes of its
    parents; if not selected for crossover, the chromosome is a duplicate of either
    the mother or father.

CIRCLE_GENE_KEY
    A dictionary table mapping the name, length, and type of each gene in the genome
"""

POPULATION = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

WIDTH, HEIGHT = 1600, 900

CIRCLE_GENE_KEY = {'r': {'len': 8, 'type': int},
                   'x': {'len': 11, 'type': int},
                   'y': {'len': 11, 'type': int}}

CIRCLE_GENOME_LENGTH = sum([g['len'] for g in CIRCLE_GENE_KEY.values()])

os.environ['SDL_VIDEO_CENTERED'] = '1'
screen = pygame.display.set_mode((WIDTH, HEIGHT))

def decode_genes(chromosome, gene_key=CIRCLE_GENE_KEY):
    """
    Chunk the genes in a chromosomeaccording to the specification of gene_key.
    :param chromosome: any chromsome with genes matching those specified in gene_key
    :type chromosome: Chromsome
    :param gene_key: a name-mapped dictionary specifying gene lengths and types
    :type gene_key: dict
    :return: A name-mapped dictionary of gene values
    :rtype: dict
    """
    offset = 0
    genes = {}
    for k, v in gene_key.iteritems():
        chunk = ((1 << v['len']) - 1) << offset
        gene = (chromosome.genes & chunk) >> offset
        genes[k] = gene
        offset += v['len']
    return genes


class Chromosome:
    def __init__(self, mother=None, father=None):
        """
        Initialize the chromsome in a few ways:
            1.  If neither a mother nor father are given, randomly create a genome
            2.  If crossover is triggered, cross the mother's and father's genomes
            3.  If no crossover, make this Chromosome a copy of the mother or father
        :param mother:
        :type mother: Chromosome
        :param father:
        :type father:
        :return:
        :rtype:
        """
        if mother is None or father is None:
            self.genes = random.getrandbits(CIRCLE_GENOME_LENGTH)
        elif random.random() <= CROSSOVER_RATE:
            self.genes = random.choice([mother.crossover(father),
                                        father.crossover(mother)])
        else:
            self.genes = random.choice([mother.genes, father.genes])

        self.mutate()
        self.symbols = decode_genes(self)
        self.fitness = None
        return

    def crossover(self, other):
        """
        Construct a genome from those of this Chromsome and another partner Chromsome
        by selecting a random position along the length of the genome, before which
        this Chromsome's genes will be contributed, after which the partner's genes.
        :param other:The other Chromsome contributing genes to the crossover offspring
        :type other:Chromsome
        :return: A gene sequence
        :rtype: int
        """
        place = random.randint(1, CIRCLE_GENOME_LENGTH)
        mask = ((1 << CIRCLE_GENOME_LENGTH) - 1)
        high = mask ^ ((1 << place) - 1)
        low = mask ^ high
        return (self.genes & high) + (other.genes & low)

    def mutate(self):
        """
        Flip the value of each gene along the length of this chromsome's genome at a
        rate defined by the MUTATION_RATE constant.
        :return: A gene sequence
        :rtype: int
        """
        for i in xrange(0, CIRCLE_GENOME_LENGTH):
            if random.random() <= MUTATION_RATE:
                self.genes ^= (1 << i)
        return self.genes

    def print_genes(self):
        """
        Print a binary representation of the genome integer.
        :return: A string formatted binary representation of the genome integer.
        :rtype: str
        """
        return ('{0:0' + str(CIRCLE_GENOME_LENGTH) + 'b}').format(self.genes)


class Circle():
    def __init__(self, chromosome, color=(0x80, 0x0, 0x0)):
        """
        Initialize the circle object according to the genes in the passed chromsome,
        optionally applying a custom color.
        :param chromosome: A chromosome object containing the genes of a circle.
        :type chromosome: Chromsome
        :param color: A color expressed as a tuple of R, G, B hex values.
        :type color: tuple
        :return: Nothing
        :rtype: NoneType
        """
        self.chromosome = chromosome
        self.r = chromosome.symbols['r']
        self.x = chromosome.symbols['x']
        self.y = chromosome.symbols['y']
        self.color = color
        self.pos = [self.x, self.y]
        self.area = np.pi * self.r * self.r
        self.bounding_box = [self.x - self.r, self.y - self.r, self.r * 2, self.r * 2]
        self.fitness = 0
        self.most_fit = False
        return


def find_collisions(c, population):
    """
    Find and return all collisions (intersections or enclosures) between a circle and a
    population of circles.

    NOTE: In this implementation of the GA, the population will always be distinct from
    the tested circle, so there is no handling of identity cases.
    :param c: A circle
    :type c: Circle
    :param population: A list of circles
    :type population: list
    :return: A list of collisions, expressed as dict objects including a clip_mask
    from PyGame and an expression of the square-approximated collision area.
    :rtype: list
    """
    collisions = []
    for o in population:
        center_d = np.linalg.norm(np.array(o.pos) - np.array(c.pos))
        radii_sum = o.r + c.r
        radii_diff = abs(o.r - c.r)
        if (center_d <= radii_diff or radii_diff < center_d < radii_sum):
            clip_mask = pygame.Rect(o.bounding_box).clip(pygame.Rect(c.bounding_box))
            collisions.append({'clip_mask': clip_mask,
                               'collision_area': clip_mask.size[0] * clip_mask.size[1]})

    return collisions


def check_viability(c, population):
    """
    Check if a Circle is viable.  A circle is non-viable if any of these are true:
    Chromsome are designated non-viable if either:
        The circle's radius is below a certain level
        Any part of the circle lies out of bounds of the display area
        The circle collides with an existing circle
    :param c: The circle whose viability will be tested
    :type c: Circle
    :param population: A population of circles for collision testing
    :type population: list
    :return: Whether the Circle is viable
    :rtype: bool
    """
    if c.r < 2:
        return False
    elif (c.x - c.r < 0
          or c.y - c.r < 0
          or c.x + c.r > WIDTH
          or c.y + c.r > HEIGHT):
        return False
    elif find_collisions(c, population):
        return False
    else:
        return True


def calculate_fitness(population, attribute='area'):
    """
    Calculate the fitness of each member in a population, based on a given attribute.
    :param population: A population of objects
    :type population: list
    :param attribute: The name of the attribute which will determine fitness
    :type attribute: str
    :return: None
    :rtype: NoneType
    """
    values = np.array([getattr(p, attribute) for p in population])
    for p in population:
        p.fitness = getattr(p, attribute) / max(values)
        if getattr(p, attribute) == max(values):
            p.most_fit = True
    return

def reproduce(generation, elite_count=1):
    """
    Create a new generation based on a list of objects as follows:
        1.  Extract the raw fitness of each list member and scale it from 0 to 1.
        2.  Reserve the specified number of  'elites': members who are guaranteed to
            pass on to the next generation without mutation or crossover.
        3.  Begin the generative process:
            A.  Given a mother and father (selected from the current generation
                proportionally based on fitness), generate a child
            B.  If the child is viable, add it to the new generation
            C.  If the child is not viable, repeat until a viable child is created
        4.  Once the population has been generated, calculate fitness for each member.
    :param generation: The current generation of reproducible objects
    :type generation: list
    :return: A new generation of objects
    :rtype: list
    """
    raw_fitness = np.array([c.fitness for c in generation])
    rel_fitness = raw_fitness / raw_fitness.sum()

    generation.sort(key=lambda x: x.fitness, reverse=True)
    new_generation = []
    new_generation += generation[0:elite_count]

    for _ in xrange(0, POPULATION - elite_count):
        viable = False
        while not viable:
            mother = None
            father = None
            while mother == father:
                mother = np.random.choice(generation, p=rel_fitness)
                father = np.random.choice(generation, p=rel_fitness)
            new_chromosome = Chromosome(mother.chromosome, father.chromosome)
            new_circle = Circle(new_chromosome)
            viable = check_viability(new_circle, static_circles)
        new_generation.append(new_circle)
    calculate_fitness(new_generation)
    return new_generation

class Core(object):
    def __init__(self, surface, name):
        pygame.display.set_caption(name)
        self.screen = surface
        self.clock = pygame.time.Clock()
        self.cur_gen = []
        self.initialize()

    def dispatch(self, event):
        """
        Dispatcher that emits pygame commands based on input events.
        :param event: A PyGame event.
        :type event: event
        :return: None
        :rtype: NoneType
        """
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.initialize()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pass

    def run(self):
        generation = 0
        while True:
            for event in pygame.event.get():
                self.dispatch(event)
            self.screen.fill([0xFF, 0xFF, 0xFF])
            for sc in static_circles:
                pygame.draw.circle(self.screen, sc.color, sc.pos, sc.r)
            for cg in self.cur_gen:
                if cg.most_fit:
                    pygame.draw.circle(self.screen, cg.color, cg.pos, cg.r)
            pygame.display.flip()
            self.cur_gen = reproduce(self.cur_gen)
            generation += 1
            if generation % 100 == 0:
                print generation

    def initialize(self):
        """
        Initialize (or re-initialize) a set of static circles and population circles.
        Static circles are the obstacles around which the population circles seek to
        maximize their size.
        :return: None
        :rtype: NoneType
        """
        global static_circles
        static_circles = []
        for _ in xrange(0, 50):
            viable = False
            while not viable:
                chromosome = Chromosome()
                static_circle = Circle(chromosome, (0, 0, 0))
                viable = check_viability(static_circle, static_circles)
            static_circles.append(static_circle)

        self.cur_gen = []
        for _ in xrange(0, POPULATION):
            viable = False
            while not viable:
                chromosome = Chromosome()
                circle = Circle(chromosome, (128, 0, 0))
                viable = check_viability(circle, static_circles + self.cur_gen)
            self.cur_gen.append(circle)
        calculate_fitness(self.cur_gen)
        cur_gen = reproduce(self.cur_gen)

if __name__ == '__main__':
    static_circles = []
    main = Core(screen, 'Node')
    main.run()
    print 'done'
