import matplotlib.pyplot as plt
# Створення двовимірної системи координат
fig, ax = plt.subplots()

# Візуалізація кількох векторів у двовимірному просторі
ax.quiver(0, 0, 1, 2, angles='xy', scale_units='xy', scale=1)
ax.quiver(0, 0, -1, -1, angles='xy', scale_units='xy', scale=1)
ax.quiver(0, 0, 2, 0.5, angles='xy', scale_units='xy', scale=1)

# Налаштування меж графіка
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Двовимірний Векторний Простір')
plt.show()