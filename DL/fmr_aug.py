"""
Модуль аугментации изображений для датасета Ferrari/Mercedes/Renault.

Что делает:
1) Загружает изображения и метки из внешнего модуля fmr_load.
2) Делит на train/test.
3) Увеличивает train с помощью случайных геометрических и цветовых преобразований.
4) Приводит все изображения к единому размеру и масштабу [0, 1].
5) Сохраняет массивы в сжатый .npz (fmr_aug.npz).

"""

from matplotlib import pyplot as plt
from numpy import savez_compressed
from PIL.Image import (
    fromarray,
    Image,
    Resampling,
)
from PIL.ImageEnhance import (
    Brightness,
    Color,
    Contrast,
    Sharpness,
)
from sklearn.model_selection import train_test_split

from math import cos, radians, sin
from random import (
    choice,
    randrange as rr,
    sample,
    uniform,
)

# Функции загрузки/приведения и путь к скрипту берём из соседнего модуля
from fmr_load import coerse_images, load_images, script_dir


def show_two_images(img1: Image, img2: Image, dpi: float = 100) -> None:
    """
    Визуализирует две картинки рядом для сравнения (например, оригинал и аугментацию).

    Args:
        img1: Первая картинка (PIL.Image).
        img2: Вторая картинка (PIL.Image).
        dpi: DPI для matplotlib.
    """
    width = max(img1.width, img2.width)
    height = max(img1.height, img2.height)

    fig = plt.figure(
        figsize=(width / dpi * 2.2 * 4, height / dpi * 1.1 * 4),
        layout='tight'
    )
    axs = fig.subplots(1, 2)

    axs[0].imshow(img1)
    axs[0].set_axis_off()
    axs[1].imshow(img2)
    axs[1].set_axis_off()

    plt.show()
    plt.close()


def rand_crop(img: Image, max_crop_percent: float = 0.07) -> Image:
    """
    Случайное кадрирование: немного обрезаем края с каждой стороны.

    Args:
        img: Входное изображение.
        max_crop_percent: Максимальная доля ширины/высоты, которую можно обрезать.

    Returns:
        Обрезанная копия изображения.
    """
    max_crop_x = int(img.width * max_crop_percent)
    max_crop_y = int(img.height * max_crop_percent)
    return img.crop(
        box=(
            0 + rr(max_crop_x),
            0 + rr(max_crop_y),
            img.width - rr(max_crop_x),
            img.height - rr(max_crop_y),
        )
    )


def rand_angle(
    img: Image,
    min_angle: float = 2,
    max_angle: float = 7,
    auto_crop: bool = True
) -> Image:
    """
    Случайный небольшой поворот изображения.

    Args:
        img: Входное изображение.
        min_angle: Минимальный модуль угла поворота (в градусах).
        max_angle: Максимальный модуль угла поворота (в градусах).
        auto_crop: Если True — приблизительно обрезает чёрные поля после поворота.

    Returns:
        Повернутая (и, возможно, обрезанная) копия изображения.
    """
    angle = uniform(min_angle, max_angle)
    img_rot = img.rotate(
        angle=choice([1, -1]) * angle,
        resample=Resampling.BICUBIC,
        expand=True,  # сохраняем всю картинку после поворота (появятся поля)
    )

    if auto_crop:
        angle = radians(abs(angle))
        img_rot = img_rot.crop(
            box=(
                sin(angle) * img.height,
                sin(angle) * img.width,
                cos(angle) * img.width,
                cos(angle) * img.height,
            )
        )
    return img_rot


def rand_scale(
    img: Image,
    min_scale_percent: float = 0.05,
    max_scale_percent: float = 0.1
) -> Image:
    """
    Случайно масштабирует изображение немного вверх или вниз.

    Args:
        img: Входное изображение.
        min_scale_percent: Минимальная доля изменения (0.05 = 5%).
        max_scale_percent: Максимальная доля изменения (0.1 = 10%).

    Returns:
        Масштабированная копия изображения (без паддинга/кадрирования).
    """
    scale_coef = 1 + choice([1, -1]) * uniform(min_scale_percent, max_scale_percent)
    img_sc = img.resize(
        size=(
            round(img.width * scale_coef),
            round(img.height * scale_coef)
        ),
        resample=Resampling.BICUBIC,
    )
    return img_sc


def rand_bright(
    img: Image,
    min_factor_delta: float = 0.05,
    max_factor_delta: float = 0.20
) -> Image:
    """
    Случайно меняет яркость.

    Args:
        img: Входное изображение.
        min_factor_delta: Минимальная величина отклонения коэффициента от 1.0.
        max_factor_delta: Максимальная величина отклонения коэффициента от 1.0.

    Returns:
        Копия изображения с изменённой яркостью.
    """
    factor = 1 + choice([-1, 1]) * uniform(min_factor_delta, max_factor_delta)
    return Brightness(img).enhance(factor)


def rand_color(
    img: Image,
    min_factor_delta: float = 0.1,
    max_factor_delta: float = 0.3
) -> Image:
    """
    Случайно меняет насыщенность (цветность).

    Args:
        img: Входное изображение.
        min_factor_delta: Минимальная величина отклонения коэффициента от 1.0.
        max_factor_delta: Максимальная величина отклонения коэффициента от 1.0.

    Returns:
        Копия изображения с изменённой насыщенностью.
    """
    factor = 1 + choice([-1, 1]) * uniform(min_factor_delta, max_factor_delta)
    return Color(img).enhance(factor)


def rand_contrast(
    img: Image,
    min_factor_delta: float = 0.05,
    max_factor_delta: float = 0.25
) -> Image:
    """
    Случайно меняет контраст.

    Args:
        img: Входное изображение.
        min_factor_delta: Минимальная величина отклонения коэффициента от 1.0.
        max_factor_delta: Максимальная величина отклонения коэффициента от 1.0.

    Returns:
        Копия изображения с изменённым контрастом.
    """
    factor = 1 + choice([-1, 1]) * uniform(min_factor_delta, max_factor_delta)
    return Contrast(img).enhance(factor)


def rand_sharp(
    img: Image,
    min_factor_delta: float = 0.25,
    max_factor_delta: float = 0.5
) -> Image:
    """
    Случайно меняет резкость.

    Args:
        img: Входное изображение.
        min_factor_delta: Минимальная величина отклонения коэффициента от 1.0.
        max_factor_delta: Максимальная величина отклонения коэффициента от 1.0.

    Returns:
        Копия изображения с изменённой резкостью.
    """
    factor = 1 + choice([-1, 1]) * uniform(min_factor_delta, max_factor_delta)
    return Sharpness(img).enhance(factor)


# Списки доступных преобразований:
# Геометрические — выбираем одну из преднастроенных комбинаций,
geom_transforms = [
    [rand_angle],
    [rand_scale, rand_angle],
    [rand_crop, rand_scale],
]
color_transforms = [
    rand_bright,
    rand_color,
    rand_contrast,
    rand_sharp
]


def rand_tranform(img: Image, min_transforms: int = 2) -> Image:
    """
    Применяет к изображению случайную комбинацию геометрических и цветовых преобразований.

    Логика:
    - Берём случайную геометрическую связку из "geom_transforms".
    - Добавляем случайный набор цветовых преобразований (не менее min_transforms//2).
    - Избегаем одновременного rand_angle и rand_crop.

    Args:
        img: Входное изображение.
        min_transforms: Минимальное число цветовых трансформаций, которые надо добавить.

    Returns:
        Трансформированная копия изображения.
    """
    transforms = (
        choice(geom_transforms)
        +
        sample(color_transforms, rr(min_transforms // 2, len(color_transforms)))
    )

    # Если случайно попалась «тяжёлая» связка поворот + обрезка — без одного из них обойдёмся.
    if rand_angle in transforms and rand_crop in transforms:
        transforms.remove(choice([rand_angle, rand_crop]))

    # Последовательно применяем выбранные функции
    for func in transforms:
        img = func(img)
    return img


def augment(imgs: list[Image], lbls: list[int]) -> tuple[list[Image], list[int]]:
    """
    Увеличивает датасет: к каждому изображению добавляет 1–2 аугментированные версии.

    Args:
        imgs: Список исходных изображений.
        lbls: Соответствующие метки классов.

    Returns:
        (imgs_aug, lbls_aug): расширенные списки изображений и меток,
        где каждый оригинал продублирован с 1–2 вариациями.
    """
    imgs_aug, lbls_aug = [], []
    for i in range(len(imgs)):
        # Оригинал
        imgs_aug.append(imgs[i])
        lbls_aug.append(lbls[i])
        # 1–2 случайные аугментации
        for _ in range(rr(1, 3)):
            imgs_aug.append(rand_tranform(imgs[i]))
            lbls_aug.append(lbls[i])
    return imgs_aug, lbls_aug


if __name__ == '__main__':
    # 1) Загружаем исходные изображения и метки из диска.
    #    load_images() читает папку fmr/ и возвращает списки PIL.Image и целочисленных меток.
    cars_imgs, cars_lbls = load_images()

    # Пример визуальной проверки аугментации
    # i = rr(len(cars_imgs))
    # (img1, img2), _ = coerse_images(
    #     [cars_imgs[i], rand_tranform(cars_imgs[i])],
    #     [0, 1],
    #     128,
    #     72,
    #     scale=False
    # )
    # show_two_images(fromarray(img1, mode='RGB'), fromarray(img2, mode='RGB'))

    # 2) Делим на train/test до аугментации, чтобы аугментация не «перетекла» в тест.
    x_train, x_test, y_train, y_test = train_test_split(
        cars_imgs,
        cars_lbls,
        test_size=0.2,
        random_state=1  # фиксируем разбиение (для воспроизводимости)
    )

    # 3) Аугментируем только train. Тест остаётся «чистым».
    x_train, y_train = augment(x_train, y_train)

    # 4) Приводим все изображения к одному размеру.
    #    Берём минимальную ширину/высоту из train, чтобы уменьшить и выровнять всё под неё.
    width = min([img.width for img in x_train])
    height = min([img.height for img in x_train])

    # 5) Преобразуем train и test одинаково.
    x_train, y_train = coerse_images(
        x_train,
        y_train,
        width,
        height,
    )
    x_test, y_test = coerse_images(
        x_test,
        y_test,
        width,
        height,
    )

    # 6) Сохраняем получившиеся массивы в сжатый архив .npz.
    #    Порядок сохранения: x_train, y_train, x_test, y_test.
    savez_compressed(
        script_dir / 'fmr_aug.npz',
        x_train,
        y_train,
        x_test,
        y_test,
    )
