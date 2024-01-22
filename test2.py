import numpy as np
# Вихід з другого шару (до активації)
layer2_output = np.array([[-0.11107764, -0.1520794]])

# Обчислення експоненти кожного елемента
exp_values = np.exp(layer2_output)
print(exp_values)