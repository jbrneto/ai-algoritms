import math
import numpy as np
import numpy.random as rn

class SA:
  def __init__(self, initial, fn_fitness, fn_cooler, fn_next_state, fn_acceptance, maxsteps=1000):
    self.initial = initial
    self.fn_fitness = fn_fitness
    self.fn_cooler = fn_cooler
    self.fn_next_state = fn_next_state
    self.fn_acceptance = fn_acceptance
    self.maxsteps = maxsteps

  def anneal(self):
    state = self.initial
    cost = self.fn_fitness(state)
    for step in range(self.maxsteps):
      fraction = step / float(self.maxsteps)
      temperature = self.fn_cooler(fraction)
      new_state = self.fn_next_state(state, fraction)
      new_cost = self.fn_fitness(new_state)
      if self.fn_acceptance(cost, new_cost, temperature) > rn.random():
        state, cost = new_state, new_cost
    return state

class SAAckley:
  def __init__(self, interval):
    self.interval = interval
    self.start = (self.randomize(), self.randomize())

  def randomize(self):
    a, b = self.interval
    return a + (b - a) * rn.random_sample()

  def fitness(self, state):
    x, y = state
    # Ackley's Function
    return -20.0*math.exp(-0.2*math.sqrt(0.5*((x**2)+(y**2)))) - math.exp(0.5*(math.cos(2.0*math.pi*x)+math.cos(2.0*math.pi*y))) + math.e + 20

  def cooler(self, fraction):
    return max(0.01, min(1, 1 - fraction))

  def next_state(self, state, fraction):
    spread = (max(self.interval) - min(self.interval)) * fraction / 10
    delta1 = (-spread/2.) + spread * rn.random_sample()
    delta2 = (-spread/2.) + spread * rn.random_sample()
    return (self.clip(state[0] + delta1), self.clip(state[1] + delta2))

  def clip(self, x):
    a, b = self.interval
    return max(min(x, b), a) # Force x into the interval

  def acceptance(self, cost, new_cost, temperature):
    return 1 if new_cost < cost else np.exp(- (new_cost - cost) / temperature)


problem = SAAckley(interval=(-5, 5))
sa = SA(
  initial = problem.start,
  fn_fitness = problem.fitness,
  fn_cooler = problem.cooler,
  fn_next_state = problem.next_state,
  fn_acceptance = problem.acceptance,
  maxsteps=1000
  )
solution = sa.anneal();
print(solution, problem.fitness(solution))