from keras import Sequential
from keras.activations import (
    relu, 
    sigmoid,
    softmax,
)
from keras.layers import (
    Conv2D, 
    Dense, 
    Dropout,
    Flatten, 
    Input,
    MaxPooling2D, 
)
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
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
    x_test, x_train, y_train, y_test = load_npz(fileout).values()


# масштабирование — приведение к диапазону [0; 1]
x_train = x_train / 255
x_test = x_test / 255
# перекодирование
y_train = array([one_hot_encoder(y) for y in y_train])
y_test = array([one_hot_encoder(y) for y in y_test])

# Инициализация модели, подбор гиперпараметров для каждого слоя
model = Sequential(name='MNIST_digits_reckognition')

model.add(Input(shape=(28, 28, 1)))
# свёрточная часть
model.add(Conv2D(
    filters=32,
    kernel_size=3,
    # то же самое:
    # kernel_size=(3, 3),
    activation=relu,
    name='low_lvl_feature_extraction'
)) # размер карты признаков (26, 26, 32) 
model.add(MaxPooling2D()) # после подвыборки размер карты - (13, 13, 32)
model.add(Conv2D(
    filters=64,
    kernel_size=2,
    activation=relu,
    name='high_lvl_feature_extraction'
)) # свёртка по карте признаков. Размер новой карты признаков (12, 12, 64)
model.add(MaxPooling2D()) # после подвыборки размер карты - (6, 6, 64)

model.add(Flatten()) # псевдо-слой меняет размерность. Плоский вектор с размерностью 2304 элементов.
# классификационная часть
# model.add(Dropout(rate=0.5))
model.add(Dense(units=10, activation=softmax, name='output'))

model.summary()
# Model: "MNIST_digits_reckognition"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ low_lvl_feature_extraction (Conv2D)  │ (None, 26, 26, 32)          │             320 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ high_lvl_feature_extraction (Conv2D) │ (None, 12, 12, 64)          │           8,256 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 6, 6, 64)            │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 2304)                │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ output (Dense)                       │ (None, 10)                  │          23,050 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 31,626 (123.54 KB)
#  Trainable params: 31,626 (123.54 KB)
#  Non-trainable params: 0 (0.00 B)

# Собираем функцию потерь, оптимизатор (стохастический градиентный спуск) и метрики
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.01),
    metrics=[
        CategoricalAccuracy(name='acc'),
    ]
)
print('\nОБУЧЕНИЕ\n')
epochs = 15
training_results = model.fit(
    x_train,
    y_train,
    batch_size=64,
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
# loss = 0.128
# accuracy = 98.5%

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
]) / 255
test_true = array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 8]) # истинные значения 

predictions = model.predict(test_images, verbose=1)

test_acc = accuracy_score(test_true, predictions.argmax(axis=1))
print(f'additional test accuracy = {test_acc:.1%}\n')
# additional test accuracy = 92.3%

