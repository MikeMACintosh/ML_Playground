"""
Имеются данные по среднему годовому доходу в некотором городе,
с помощью линейной регрессии построим линию тренда,
аппроксимирующую наблюдения, и найдём предсказанное
значение дохода на следующий год.
"""

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

income_list = np.array([1133, 1222, 1354, 1369, 1362, 1377, 1465, 1584])

years = np.arange(1, len(income_list)+1)
sum_i = len(income_list)*(len(income_list)+1)/2
sum_i_q = len(income_list)*(len(income_list)+1)*(2*len(income_list)+1)/6
sum_element = sum(income_list)
mult_list_years = income_list*years
sum_list_years = sum(mult_list_years)

# Получим систему уравнений и найдём её решение:
M = np.array([[len(income_list), sum_i], [sum_i, sum_i_q]])
A = np.array([sum_element, sum_list_years])
answer = np.linalg.solve(M, A)

print('-----------------')
print('Линия тренда: ', 'y =', answer[0], '+', answer[1], '* t')
x = np.arange(1, len(income_list)+1)
y = answer[0] + answer[1] * x
print('Коэффициенты уравнения: ', answer)

next_year = years[-1] + 1

# Построим график:
fig, ax = plt.subplots()
ax.plot(x, y, color='red', label='Уравнение тренда')
ax.scatter(x, income_list, color='g', label='Наблюдения')
ax.scatter(next_year, answer[0] + answer[1] * next_year, label='Прогноз на следующий год')
ax.set_xlabel('Год')
ax.set_ylabel('Доход на душу населения, день. ед')
ax.legend()
plt.show()


