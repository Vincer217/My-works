from keras.activations import relu, softmax
from keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout,
    Input, MaxPooling2D, ReLU, GlobalAveragePooling2D,
)
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2

from matplotlib import pyplot as plt
from numpy import load
from pathlib import Path
from sys import path


# ДАННЫЕ
# Загружаем заранее подготовленный датасет (уже аугментированный)
# В файле лежат: обучающая выборка (x_train, y_train) и тестовая (x_test, y_test)
x_train, y_train, x_test, y_test = *load(
    Path(path[0]) / 'fmr_aug.npz', allow_pickle=True
).values(),


# МОДЕЛЬ
# Определяем сверточную нейросеть для классификации картинок на 3 класса (Ferrari, Mercedes, Renault)
model = Sequential([
    Input(shape=(x_train.shape[1:])),  # входной слой (размерность изображения из данных)

    # Первый блок: извлечение низкоуровневых признаков
    Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0005), name='low_lvl_features'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D((2, 2)),

    # Второй блок: признаки среднего уровня
    Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0005), name='mid_lvl_features'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D((2, 2)),

    Dropout(0.25),  # регуляризация — случайно отключаем нейроны

    # Третий блок: более глубокие признаки
    Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0005), name='high_lvl_features'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D((2, 2)),
    
    Dropout(0.3),

    # Четвертый блок: самые глубокие признаки
    Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0005), name='deeper_features'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D((2, 2)),
    
    Dropout(0.35),

    # Переход к полносвязным слоям
    GlobalAveragePooling2D(),  # усредняем карты признаков в один вектор
    
    Dense(256, activation=relu, name='hidden'),  # скрытый слой
    Dropout(0.4),  # сильная регуляризация перед финальным выходом

    Dense(3, activation=softmax, name='output'),  # выход: 3 класса
], name='Ferrari_Mercedes_Renault_recognition')

model.summary()  # вывод архитектуры


# КОМПИЛЯЦИЯ
# Настраиваем процесс обучения
model.compile(
    loss=CategoricalCrossentropy(label_smoothing=0.1),  # кросс-энтропия с мягкими метками
    optimizer=Adam(learning_rate=1e-4),  # оптимизатор Adam с небольшим lr
    metrics=[CategoricalAccuracy(name='acc')]  # метрика — точность классификации
)


# CALLBACKS (колбэки управления обучением)
callbacks = [
    EarlyStopping(
        monitor="val_loss",       # следим за качеством на валидации
        patience=10,              # если 10 эпох подряд нет улучшений - стоп
        restore_best_weights=True,# вернем веса, где модель была лучшей
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,               # уменьшаем lr в 2 раза
        patience=5,               # если 5 эпох без улучшений
        min_lr=1e-6,              # но не меньше 1e-6
        verbose=1
    ),
    ModelCheckpoint(
        "best_model.keras",       # сохраняем веса в файл
        monitor="val_loss",
        save_best_only=True,      # только лучшую модель
        verbose=1
    )
]


# ОБУЧЕНИЕ
epochs = 50
training_results = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=epochs,
    validation_split=0.2,  # выделяем 20% данных под валидацию
    callbacks=callbacks,   # подключаем колбэки
    verbose=1
)


# ТЕСТИРОВАНИЕ
print('\nТЕСТИРОВАНИЕ\n')
scores = model.evaluate(x_test, y_test, return_dict=True, verbose=1)
print(
    f'\nloss = {scores["loss"]:.3f}',
    f'\naccuracy = {scores["acc"]:.1%}\n',
)
# пример результата:
# Restoring model weights from the end of the best epoch: 36.
# loss = 0.760
# accuracy = 85.5%


# ГРАФИКИ
# Строим графики обучения (loss и accuracy на train/val + точка на test)
fig = plt.figure(figsize=(11, 5))
axs = fig.subplots(1, 2)

# Loss
axs[0].plot(range(1, len(training_results.history['loss'])+1),
            training_results.history['loss'], label='loss')
axs[0].plot(range(1, len(training_results.history['val_loss'])+1),
            training_results.history['val_loss'], label='val_loss')
axs[0].scatter(len(training_results.history['loss'])+1,
               scores['loss'], s=30, c='#c80608', label='test_loss')
axs[0].legend()

# Accuracy
axs[1].plot(range(1, len(training_results.history['acc'])+1),
            training_results.history['acc'], label='accuracy')
axs[1].plot(range(1, len(training_results.history['val_acc'])+1),
            training_results.history['val_acc'], label='val_accuracy')
axs[1].scatter(len(training_results.history['acc'])+1,
               scores['acc'], s=30, c='#c80608', label='test_accuracy')
axs[1].legend()

fig.show()
