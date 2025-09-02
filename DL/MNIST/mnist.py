from keras import Sequential
from keras.activations import (
    relu, 
    sigmoid,
    softmax,
)
from keras.layers import Dense, Input
from keras.losses import CategoricalCrossentropy
from keras.metrics import (
    CategoricalAccuracy,
    CategoricalCrossentropy as CategoricalCrossentropyMetric,
    F1Score
)
from keras.optimizers import Adam

from matplotlib import pyplot as plt
from numpy import (
    array, 
    load as load_npz, 
)
from numpy.random import default_rng
from PIL import Image
from sklearn.metrics import accuracy_score

from functools import partial
from pathlib import Path
from sys import path


def one_hot_encoder(x: int):
    """
    Функция, которая кодирует категориальные переменные
    """
    return (0,) * x + (1,) + (0,) * (9 - x)


def one_hot_decoder(x):
    ...



script_dir = Path(path[0])
data_path = script_dir / 'mnist.npz'

with open(data_path, 'rb') as fileout:
    x_test, x_train, y_train, y_test = load_npz(fileout).values() # извлекаем значения 
    
# Значения - это матрицы 28 на 28, в которых содержатся числа от 0 до 255 - это яркость пикселей в монохромном изображении.

# графический вывод подвыборки из тренировочной
# m, n = 10, 16
# randgen = default_rng()
# sample = randgen.choice(x_train, m*n)
# 
# fig = plt.figure(figsize=(17, 11), layout='tight')
# axs = fig.subplots(m, n)
# 
# for i in range(m):
#     for j in range(n):
#         axs[i][j].imshow(sample[i*n+j], cmap='gray', vmin=0, vmax=255)
#         axs[i][j].set_axis_off()
# 
# fig.show()


input_vector_len = x_train.shape[1] * x_train.shape[2]

# изменение размерности
x_train = x_train.reshape((x_train.shape[0], input_vector_len))
x_test = x_test.reshape((x_test.shape[0], input_vector_len))
# масштабирование — приведение к диапазону [0; 1]
x_train = x_train / 255
x_test = x_test / 255
# перекодирование
y_train = array([one_hot_encoder(y) for y in y_train])
y_test = array([one_hot_encoder(y) for y in y_test])

# Выходной слой со своей кодировкой
# one-hot encoding
# 0 -> 1 0 0 0 0 0 0 0 0 0
# 1 -> 0 1 0 0 0 0 0 0 0 0
# 2 -> 0 0 1 0 0 0 0 0 0 0
# 3 -> 0 0 0 1 0 0 0 0 0 0
# 4 -> 0 0 0 0 1 0 0 0 0 0
# 5 -> 0 0 0 0 0 1 0 0 0 0
# 6 -> 0 0 0 0 0 0 1 0 0 0
# 7 -> 0 0 0 0 0 0 0 1 0 0
# 8 -> 0 0 0 0 0 0 0 0 1 0
# 9 -> 0 0 0 0 0 0 0 0 0 1

# Инициализация модели, подбор гиперпараметров для каждого слоя 
model = Sequential(name='MNIST_digits_reckognition')

model.add(Input(shape=(input_vector_len,), dtype='float64'))
model.add(Dense(200, activation=partial(relu, threshold=0.4)), )
model.add(Dense(100, activation=sigmoid), )
model.add(Dense(10, activation=softmax, name='output'))

model.summary()
# Model: "MNIST_digits_reckognition"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 200)                 │         157,000 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 100)                 │          20,100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ output (Dense)                       │ (None, 10)                  │           1,010 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 178,110 (695.74 KB)
#  Trainable params: 178,110 (695.74 KB)
#  Non-trainable params: 0 (0.00 B)

# Собираем функцию потерь, оптимизатор (стохастический градиентный спуск) и метрики
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.01),
    metrics=[
        CategoricalAccuracy(name='acc'),
        # F1Score(name='f1'),
    ]
)

print('\nОБУЧЕНИЕ\n')

# Количество эпох 
epochs = 15
# Обучение 
training_results = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    validation_split=0.25,
    verbose=1
)

print('\nТЕСТИРОВАНИЕ\n')

scores = model.evaluate(
    x_test,
    y_test,
    return_dict=True,
    verbose=1
)
print(
    f'\nloss = {scores["loss"]:.3f}',
    f'\naccuracy = {scores["acc"]:.1%}\n',
)
# loss = 0.149
# accuracy = 96.3%

fig = plt.figure(figsize=(11, 5))
axs = fig.subplots(1, 2)

# Функция потерь на обучающих выборках 
axs[0].plot(
    range(1, epochs+1), 
    training_results.history['loss'], 
    label='loss'
)
# Функция потерь на валидационных подвыборках
axs[0].plot(
    range(1, epochs+1), 
    training_results.history['val_loss'],
    label='val_loss'
)
# Функция потерь на тестовых данных 
axs[0].scatter(
    epochs+1, 
    scores['loss'], 
    s=30, 
    c='#c80608', 
    label='test_loss'
)
axs[0].set_xticks(range(1, epochs+2))
axs[0].legend()

# Точность на обучающих данных
axs[1].plot(
    range(1, epochs+1), 
    training_results.history['acc'], 
    label='accuracy'
)
# Точность на валидационных подвыборках 
axs[1].plot(
    range(1, epochs+1), 
    training_results.history['val_acc'], 
    label='val_accuracy'
)
# Окончательная точность на тестовых данных 
axs[1].scatter(
    epochs+1, 
    scores['acc'], 
    s=30, 
    c='#c80608', 
    label='test_accuracy'
)
axs[1].set_xticks(range(1, epochs+2))
axs[1].legend()

fig.show()


test_images_dir = script_dir / 'mnist_test/28'
# Тестовые изображения
test_images = array([
    array(Image.open(img_path).convert('L'))
    for img_path in test_images_dir.iterdir()
    if img_path.is_file()
])
# Изменение размера
test_images = test_images.reshape(
    test_images.shape[0], 
    test_images.shape[1] * test_images.shape[2]
)
# Масштабирование
test_images = test_images / 255

# >>> test_images.shape
# (13, 28, 28)

predictions = model.predict(test_images, verbose=1)

# >>> predictions.round(2)
# array([[0.2 , 0.  , 0.  , 0.  , 0.  , 0.02, 0.59, 0.  , 0.05, 0.13],
#        [1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
#        [0.99, 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  , 0.  , 0.  ],
#        [0.73, 0.  , 0.26, 0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
#        [0.  , 0.99, 0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  ],
#        [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.27, 0.  , 0.73, 0.  ],
#        [0.  , 0.99, 0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.01, 0.  ],
#        [0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
#        [0.  , 0.  , 0.98, 0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.  ],
#        [0.  , 0.  , 0.97, 0.03, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
#        [0.01, 0.  , 0.37, 0.11, 0.  , 0.01, 0.  , 0.  , 0.26, 0.24],
#        [0.  , 0.  , 0.  , 0.  , 0.89, 0.  , 0.  , 0.  , 0.  , 0.1 ],
#        [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  ]],
#       dtype=float32)
# >>> 
# >>> predictions.argmax(axis=1) 
# array([6, 0, 0, 0, 1, 8, 1, 1, 2, 2, 2, 4, 8]) # предсказанные значения 

test_true = array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 8]) # истинные значения 

# >>> accuracy_score(test_true, predictions.argmax(axis=1))
# 0.8461538461538461

