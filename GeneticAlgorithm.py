import random
import bisect

class GA:
  def __init__(self, population, fn_fitness, gene_pool):
    self.population = population
    self.fn_fitness = fn_fitness
    self.gene_pool = gene_pool
    self.fitnesses = [] # fitness for each individual in population
    self.fit_dist = [] # distribuition of probability proportional to fitness
    self.roulette = None # sampler of individuals following the fitness distribuition

  def evolve(self, ngen=1000, pmut=0.1):
    self.fit_population()
    for _ in range(ngen):
      new_population = []
      for _ in range(len(self.population)):
        p1, p2 = self.select(2)
        child = self.recombine(p1, p2)
        child = self.mutate(child, pmut)
        new_population.append(child)
      self.population = new_population
      self.fit_population()
    best = min(self.fitnesses)
    return self.population[self.fitnesses.index(best)]

  def select(self, r):
    return [self.roulette() for i in range(r)] if r > 1 else self.roulette()

  def selectUniform(self, r):
    return [self.population[random.randrange(0, len(self.population))] for i in range(r)] if r > 1 else self.population[random.randrange(0, len(self.population))]

  def recombine(self, x, y):
    c = random.randrange(0, len(x))
    return x[:c] + y[c:]

  def mutate(self, x, pmut):
    if random.uniform(0, 1) >= pmut:
      return x

    c = random.randrange(0, len(x))
    r = random.randrange(0, len(self.gene_pool))

    new_gene = self.gene_pool[r]
    return x[:c] + [new_gene] + x[c+1:]

  def fit_population(self):
    self.fitnesses = list(map(lambda x: self.fn_fitness(x), self.population))
    
    # flip roulette logic, the lower the better
    total_fit = sum(self.fitnesses)
    tmp_fit = list(map(lambda x: total_fit / x, self.fitnesses))

    weight_dist = []
    for w in tmp_fit:
      weight_dist.append(w + weight_dist[-1] if weight_dist else w)
    self.fit_dist = weight_dist

    self.roulette = lambda: self.population[bisect.bisect(self.fit_dist, random.uniform(0, self.fit_dist[-1]))]

def fn_evaluate(array):
  return sum([(i * x) for i, x in enumerate(array)])

population_size = 100
individual_size = 100
gene_pool = list(range(0, individual_size))
population = []

for _ in range(0, population_size):
  population.append(random.sample(range(0, individual_size), individual_size))

ga = GA(
  population=population, 
  fn_fitness=fn_evaluate, 
  gene_pool=gene_pool
)

solution = ga.evolve(ngen=1000, pmut=0.1)
print(solution)
print(fn_evaluate(solution))
