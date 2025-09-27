!pip install scikit-opt

import numpy as np
from sko.GA import GA
from sko.operators import ranking, selection, crossover, mutation

# دالة الهدف
demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + (x[2] - 0.5) ** 2

# تعريف Selection Operator مخصص (Tournament Selection)
def selection_tournament(algorithm, tourn_size=3):
    FitV = algorithm.FitV
    sel_index = []
    for i in range(algorithm.size_pop):
        aspirants_index = np.random.choice(range(algorithm.size_pop), size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    algorithm.Chrom = algorithm.Chrom[sel_index, :]  # الجيل التالي
    return algorithm.Chrom

# تهيئة GA
ga = GA(func=demo_func, n_dim=3, size_pop=100, max_iter=500,
        prob_mut=0.001, lb=[-1, -10, -5], ub=[2, 10, 2],
        precision=[1e-7, 1e-7, 1])

# تسجيل المشغلات
ga.register(operator_name='selection', operator=selection_tournament, tourn_size=3)
ga.register(operator_name='ranking', operator=ranking.ranking)
ga.register(operator_name='crossover', operator=crossover.crossover_2point)
ga.register(operator_name='mutation', operator=mutation.mutation)

# تشغيل الخوارزمية
best_x, best_y = ga.run()
print('GA - best_x:', best_x, '\nGA - best_y:', best_y)
