from keras.utils import to_categorical
from numpy import array, ndarray, savez_compressed
from PIL import Image, ImageFile
from PIL.Image import Resampling

from pathlib import Path
from sys import path


script_dir = Path(path[0])
data_dir = script_dir / 'fmr'


# labels:
#   0 — Ferrari
#   1 — Mercedes
#   2 — Renault

def load_images() -> tuple[list[ImageFile], list[int]]:
    """
    Загружает изображения автомобилей из папки `fmr` и формирует их метки.

    Структура папки:
    └── fmr/
        ├── Ferrari/
        ├── Mercedes/
        └── Renault/

    Returns:
        tuple[list[ImageFile], list[int]]:
            - cars_imgs: список изображений (PIL.ImageFile);
            - cars_lbls: список целочисленных меток (0 = Ferrari, 1 = Mercedes, 2 = Renault).
    """
    cars_imgs, cars_lbls = [], []
    
    for i, subdir in enumerate(data_dir.glob('[!_]*')):  # [!_]* — исключаем папки с "_" в имени
        for img in subdir.iterdir():
            img = Image.open(img)
            cars_imgs.append(img)
            cars_lbls.append(i)  # метка соответствует индексу папки
    return cars_imgs, cars_lbls


def coerse_images(
        cars_imgs: list[ImageFile],
        cars_lbls: list[int],
        new_width: int | None = None,
        new_height: int | None = None,
        scale: bool = True,
) -> tuple[ndarray, ndarray]:
    """
    Преобразует список изображений и меток в массивы numpy для обучения модели.

    Args:
        cars_imgs (list[ImageFile]): список изображений PIL;
        cars_lbls (list[int]): список целочисленных меток классов;
        new_width (int | None): если задано — новая ширина изображений;
        new_height (int | None): если задано — новая высота изображений;
        scale (bool): если True, нормализует пиксели в диапазон [0, 1].

    Returns:
        tuple[ndarray, ndarray]:
            - cars_imgs: массив изображений (float32), размерностью (N, H, W, C);
            - cars_lbls: one-hot массив меток (N, 3), где 3 — число классов.
    """
    resize = new_width is not None and new_height is not None
    for i in range(len(cars_imgs)):
        img = cars_imgs[i]
        if resize:
            # Изменяем размер изображения, если заданы размеры
            img = img.resize((new_width, new_height), Resampling.BICUBIC)
        cars_imgs[i] = array(img)  # переводим в numpy-массив
    
    cars_imgs = array(cars_imgs)
    # Преобразуем метки в one-hot encoding (например, [0] -> [1,0,0])
    cars_lbls = array(to_categorical(cars_lbls, num_classes=3))
    
    if scale:
        # Масштабируем пиксели к диапазону [0,1]
        cars_imgs = cars_imgs / 255.0
    
    return cars_imgs, cars_lbls


if __name__ == '__main__':
    # Загружаем изображения, приводим их к массивам и сохраняем в сжатом .npz файле
    savez_compressed(
        script_dir / 'fmr.npz',
        *coerse_images(*load_images()),
    )
