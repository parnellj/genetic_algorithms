"""
Given the digits 0 through 9 and the operators +, -, * and /,  find a sequence that will represent a given target
number. The operators will be applied sequentially from left to right as you read.
 
So, given the target number 23, the sequence 6+5*4/2+1 would be one possible solution.
 
If  75.5 is the chosen number then 5/2+9*7-5 would be a possible solution.
"""

from __future__ import division
import random
import math
from operator import add, sub, mul, div
import numpy as np

FUN_TYPE = type(add)
CHROMOSOME_LENGTH = 48
POPULATION = 20
MUTATION_RATE = 0.001
CROSSOVER_RATE = 0.7
GENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '+', '-', '*', '/', None, None]
GENE_SIZE = int(math.floor(math.log(len(GENES), 2)))

def contains(list, filter):
    # https://stackoverflow.com/a/598407/4272935
    for x in list:
        if filter(x):
            return True
    return False

def genes_to_symbols(c):
    chunk = (1 << GENE_SIZE) - 1
    symbols = []
    genes = c.genes
    while genes != 0:
        gene = GENES[genes & chunk]
        if gene is not None:
            symbols.append(GENES[genes & chunk])
        genes >>= GENE_SIZE
    symbols = symbols[::-1]

    filtered_symbols = []
    next_type = int
    for s in symbols:
        if isinstance(s, next_type):
            filtered_symbols.append(s)
            next_type = str if next_type is int else int

    if isinstance(filtered_symbols[-1], str):
        filtered_symbols = filtered_symbols[:-1]
    return filtered_symbols

def symbols_to_result(s, verbose=False):
    result = s[0]

    for op, val in zip(s[1:], s[2:]):
        if op == '+':
            result += val
        if op == '-':
            result -= val
        if op == '*':
            result *= val
        if op == '/':
            try:
                result /= val
            except ZeroDivisionError:
                result = 0
                break

    return result

def calculate_fitness(chromosomes, target):
    results = np.array([c.result for c in chromosomes])
    fitness = np.abs(1 / (target - results))

    for i, c in enumerate(chromosomes):
        c.fitness = fitness[i]

class Chromosome:
    def __init__(self, mother=None, father=None):
        if mother is None or father is None:
            self.genes = random.getrandbits(CHROMOSOME_LENGTH)
        elif random.random() <= CROSSOVER_RATE:
            self.genes = mother.crossover(father)
        else:
            self.genes = random.choice([mother.genes, father.genes])

        self.length = len(bin(self.genes)[2:])
        self.mutate()
        self.symbols = genes_to_symbols(self)
        self.result = symbols_to_result(self.symbols)
        self.fitness = None
        return

    def crossover(self, other):
        longest = max(self.length, other.length)
        place = random.randint(1, longest)
        mask = ((1 << longest) - 1)
        high = mask ^ ((1 << place) - 1)
        low = mask ^ high
        return (self.genes & high) + (other.genes & low)

    def mutate(self):
        for i in xrange(0, self.length):
            if random.random() <= MUTATION_RATE:
                self.genes ^= (1 << i)
        return self.genes

    def print_genes(self):
        return ('{0:0' + str(CHROMOSOME_LENGTH) + 'b}').format(self.genes)

mother = Chromosome()
father = Chromosome()
child = Chromosome(mother, father)

print 'mother: ' + mother.print_genes()
print 'father: ' + father.print_genes()
print 'child:  ' + child.print_genes()

def reproduce(generation, size):
    # First drop some non-viable members of the generation
    raw_fitness = np.array([c.fitness for c in generation])
    rel_fitness = raw_fitness / raw_fitness.sum()
    new_generation = []
    for _ in xrange(0, size):
        mother = None
        father = None
        while mother == father:
            mother = np.random.choice(generation, p=rel_fitness)
            father = np.random.choice(generation, p=rel_fitness)
        new_generation.append(Chromosome(mother, father))
    print [genes_to_symbols(c) for  c in new_generation]
    return new_generation

target = 100
gen_size = 100
gen_number = 0
found_solution = False
cur_gen = None
report_max = 'max fitness in gen {}: {}'
report_mean = 'mean fitness in gen {}: {}'

while not found_solution:
    if cur_gen is None:
        next_gen = [Chromosome() for _ in xrange(0, gen_size)]
    else:
        next_gen = reproduce(cur_gen, gen_size)
        gen_number += 1
    matches = [x for x in next_gen if x.result == target]
    if any(matches):
        print 'match found'
        for m in matches:
            print genes_to_symbols(m)
        break

    calculate_fitness(next_gen, target)
    cur_gen = next_gen
    mean_fitness = np.mean([c.fitness for c in cur_gen])
    print report_mean.format(gen_number, mean_fitness)


