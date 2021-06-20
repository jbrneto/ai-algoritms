import random

class KOpt:
  def __init__(self, array, fn_fitness):
    self.array = array
    self.fn_fitness = fn_fitness

  def runClassic2Opt(self):
    indv = self.array
    fit_indv = self.fn_fitness(self.array)
    changed = True

    while changed:
      changed = False
      for i in range(len(indv)):
        for k in range(i+1, len(indv)-1):
          new_indv = indv[0:i] + list(reversed(indv[i:k+1])) + indv[k+1:]
          new_fit = self.fn_fitness(new_indv)
          if new_fit < fit_indv:
            indv = new_indv
            fit_indv = new_fit
            changed = True
  
    return indv

  def runStochasticKOpt(self, kopt=2, permuts=100):
      indv = self.array
      fit_indv = self.fn_fitness(self.array)
      useds = {}

      for _ in range(0, permuts):
        valid = False
        while not valid:
          i = random.randrange(0, len(indv))
          j = i

          while j == i: j = random.randrange(0, len(indv))

          k = j
          if kopt == 3:
            while k == i or k == j: k = random.randrange(0, len(indv))

          if (i, j, k) not in useds:
            valid = True

        useds[(i, j, k)] = True

        if k < j:
          temp = j
          j = k
          k = temp

        if j < i:
          temp = i
          i = j
          j = temp

        # 2 opt I - J
        new_indv = indv[0:i] + list(reversed(indv[i:j+1])) + indv[j+1:]
        if self.fn_fitness(new_indv) < self.fn_fitness(indv):
          indv = new_indv
          
        # 3 opt
        elif kopt == 3:

          # J - K
          new_indv = indv[0:j] + list(reversed(indv[j:k+1])) + indv[k+1:]
          new_indv2 = indv[0:i] + list(reversed(indv[i:k+1])) + indv[k+1:]
          new_indv3 = indv[0:i] + indv[j:k+1] + indv[i:j+1] + indv[k+1:]
          if self.fn_fitness(new_indv) < self.fn_fitness(indv):
            indv = new_indv

          # I - K
          elif self.fn_fitness(new_indv2) < self.fn_fitness(indv):
            indv = new_indv2

          # J - K + I - J
          elif self.fn_fitness(new_indv3) < self.fn_fitness(indv):
            indv = new_indv3

      return indv

def fn_evaluate(array):
  return sum([(i * x) for i, x in enumerate(array)])

size = 100
array = random.sample(range(0, size), size)
kopt = KOpt(array, fn_evaluate)
result1 = kopt.runClassic2Opt()
result2 = kopt.runStochasticKOpt(3, 100)

print(result1)
print(fn_evaluate(result1))
print(result2)
print(fn_evaluate(result2))
