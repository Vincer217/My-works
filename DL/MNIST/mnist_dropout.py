from keras import Sequential
from keras.activations import (
    relu, 
    sigmoid,
    softmax,
)
from keras.layers import Dense, Dropout, Input
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
    x_test, x_train, y_train, y_test = load_npz(fileout).values()


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


# Инициализация модели, подбор гиперпараметров для каждого слоя 
model = Sequential(name='MNIST_digits_reckognition')

model.add(Input(shape=(input_vector_len,), dtype='float64'))
model.add(Dense(400, activation=partial(relu, threshold=0.4)), )
model.add(Dropout(0.5)) # прореживание 
model.add(Dense(100, activation=sigmoid), )
model.add(Dense(10, activation=softmax, name='output'))

model.summary()
# Model: "MNIST_digits_reckognition"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 400)                 │         314,000 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 400)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 100)                 │          40,100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ output (Dense)                       │ (None, 10)                  │           1,010 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 355,110 (1.35 MB)
#  Trainable params: 355,110 (1.35 MB)
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

axs[0].plot(
    range(1, epochs+1), 
    training_results.history['loss'], 
    label='loss'
)
axs[0].plot(
    range(1, epochs+1), 
    training_results.history['val_loss'],
    label='val_loss'
)
axs[0].scatter(
    epochs+1, 
    scores['loss'], 
    s=30, 
    c='#c80608', 
    label='test_loss'
)
axs[0].set_xticks(range(1, epochs+2))
axs[0].legend()

axs[1].plot(
    range(1, epochs+1), 
    training_results.history['acc'], 
    label='accuracy'
)
axs[1].plot(
    range(1, epochs+1), 
    training_results.history['val_acc'], 
    label='val_accuracy'
)
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

test_images = array([
    array(Image.open(img_path).convert('L'))
    for img_path in test_images_dir.iterdir()
    if img_path.is_file()
])
test_images = test_images.reshape(
    test_images.shape[0], 
    test_images.shape[1] * test_images.shape[2]
)
test_images = test_images / 255


predictions = model.predict(test_images, verbose=1)

test_true = array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 8])

test_acc = accuracy_score(test_true, predictions.argmax(axis=1))
print(f'additional test accuracy = {test_acc:.1%}\n')

