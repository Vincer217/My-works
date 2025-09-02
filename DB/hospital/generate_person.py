from pathlib import Path
from random import choice, randrange as rr 
from typing import Literal
from datetime import date, timedelta
from pprint import pprint
from sys import path

names = {}
days = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def load_data () -> None:
    """
    Загружает и структурирует данные имен, отчеств и фамилий из текстовых файлов.
    
    Функция читает данные из трех файлов в поддиректории 'names' и заполняет глобальный словарь "names" следующей структурой:
    - 'имя': список женских имен
    - 'имя_м': список мужских имен
    - 'отчество': список женских отчеств
    - 'отчество_м': список мужских отчеств
    - 'фамилия': список женских фамилий
    - 'фамилия_м': список мужских фамилий

    Требуемые файлы:
    - 'женские_имена.txt' - одна строка - одно имя
    - 'мужские_имена_отчества.txt' - строки формата:
        "Иван (Иванович, Ивановна)"
        или "Данил, Данила (Данилович, Даниловна)"
    - 'фамилии.txt' - строки формата:
        "Иванов, Иванова" или "Бойко" (если фамилия общего рода)

    Пример:
    После вызова функции словарь "names" будет содержать:
    {
        'имя': ['Анна', 'Мария', ...],
        'имя_м': ['Иван', 'Петр', ...],
        'отчество': ['Ивановна', 'Петровна', ...],
        'отчество_м': ['Иванович', 'Петрович', ...],
        'фамилия': ['Иванова', 'Петрова', ...],
        'фамилия_м': ['Иванов', 'Петров', ...]
    }
    """
    names['имя'] = []
    names['имя_м'] = []
    names['отчество_м'] = []
    names['отчество'] = []
    names['фамилия_м'] = []
    names['фамилия'] = [] 
    
    names_dir = Path(path[0]) / 'names'
    female_names = Path(names_dir / 'женские_имена.txt')
    male_names_patronymics = Path(names_dir / 'мужские_имена_отчества.txt') 
    surnames = Path(names_dir / 'фамилии.txt')
    
    with open(female_names, encoding='utf-8') as filein:
        female_names = filein.readlines()
    
    for el in female_names:
        names['имя'].append(el.strip('\n'))
        
    with open(male_names_patronymics, encoding='utf-8') as filein:
       male_names_patronymics = filein.readlines()
    
    for el in male_names_patronymics:
        l = el.split()
        if len(l) == 3:        
            names['имя_м'].append(l[0]) 
            names['отчество_м'].append(l[1].strip('(,'))
            names['отчество'].append(l[2].strip(')\n')) 
        else:
            names['имя_м'].append(l[0].strip(','))
            names['имя_м'].append(l[1].strip())
            names['отчество_м'].append(l[2].strip('(,'))
            names['отчество'].append(l[3].strip(')\n'))
            
    with open(surnames, encoding='utf-8') as filein:
       surnames = filein.readlines()
    
    for el in surnames:
        l = el.split(', ')
        if len(l) == 2:
            names['фамилия_м'].append(l[0])
            names['фамилия'].append(l[1].strip('\n'))
        else:
            l = l[0].strip('\n')
            names['фамилия_м'].append(l) 
            names['фамилия'] .append(l)
            
load_data()
            
def generate_person () -> dict[
    'имя': str,
    'отчество': str,
    'фамилия': str,
    'пол': Literal['мужской', 'женский'],
    'дата рождения': date,
    'мобильный': str
    ]:
    """
    Генерирует анкету человека со случайными данными, согласованными по полу.

    Возвращает словарь со следующими ключами:
    - 'имя' (str) – случайное имя, соответствующее полу.
    - 'отчество' (str) – случайное отчество, согласованное с именем.
    - 'фамилия' (str) – случайная фамилия, согласованная с полом.
    - 'пол' (Literal['мужской', 'женский']) – указанный пол.
    - 'дата рождения' (date) – случайная дата в пределах 100 лет от 1900 года.
    - 'мобильный' (str) – номер телефона в формате +79XXXXXXXXX.

    Условия генерации:
    - Имена, отчества и фамилии берутся из предопределённых списков ("names").
    - Дата рождения учитывает високосные годы.
    - Номер телефона соответствует российскому формату (+79...).

    Пример возвращаемого значения:
    {
        'имя': 'Иван',
        'отчество': 'Петрович',
        'фамилия': 'Смирнов',
        'пол': 'мужской',
        'дата рождения': date(1985, 7, 23),
        'мобильный': '+79123456789'
    }
    """    
    gender = choice(['мужской', 'женский'])
    name_key = 'имя_м' if gender == 'мужской' else 'имя'
    patronymic_key = 'отчество_м' if gender == 'мужской' else 'отчество'
    surname_key = 'фамилия_м' if gender == 'мужской' else 'фамилия'

    return {
        'имя': choice(names[name_key]),
        'отчество': choice(names[patronymic_key]),
        'фамилия': choice(names[surname_key]),
        'пол': gender,
        'дата рождения': date(1900, 1, 1) + timedelta(days=rr(36524)),
        'мобильный': f'+7{rr(9000000000, 10000000000)}'
    }   
        
def rand_date(from_year: int = -1, years: int = -1) -> date:
    """
    Генерирует случайную дату в заданном диапазоне лет.

    Параметры:
    ----------
    from_year :
        Год, от которого начинается отсчёт. 
        Если не указан (-1), берётся текущий год.
    years :
        Количество лет, на которые может быть сгенерирована дата (включая "from_year").
        Если не указано (-1), берётся разница между текущим годом и "from_year".

    Примеры:
    --------
    >>> rand_date()  # случайная дата в текущем году
    datetime.date(2024, 5, 15)

    >>> rand_date(2000, 10)  # случайная дата между 2000 и 2009 годами
    datetime.date(2005, 11, 3)

    Примечания:
    -----------
    - Учитываются високосные годы (29 дней в феврале, если год високосный).
    """
    if from_year == -1:
        from_year = date.today().year
    if years == -1:
        years = date.today().year - from_year
    rand_year = rr(from_year, from_year+years)
    rand_month = rr(1,13) 
    leap_febr = rand_month == 2 and is_leap_year(rand_year)
    # "days" - словарь с количеством дней в каждом месяце
    max_day = days[rand_month]+1 if leap_febr else days[rand_month]
    rand_day = rr(1, max_day+1)
    return date(rand_year, rand_month, rand_day)    
    
def is_leap_year(year: int) -> bool:
    """
    Определяет, является ли год високосным.

    Параметры:
    ----------
    year : int
        Год для проверки (должен быть >= 1).

    Возвращает:
    -----------
    bool
        True, если год високосный, иначе False.

    Примеры:
    --------
    >>> is_leap_year(2000)
    True
    >>> is_leap_year(1900)
    False
    >>> is_leap_year(2024)
    True
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)       

# >>> load_data()
# >>> pprint(generate_person(), sort_dicts=False)
# {'имя': 'Савватий',
#  'отчество': 'Парамонович',
#  'фамилия': 'Доронин',
#  'пол': 'мужской',
#  'дата рождения': datetime.date(1911, 12, 31),
#  'мобильный': '+79340517833'}
# >>>
# >>> pprint(generate_person(), sort_dicts=False)
# {'имя': 'Лаура',
#  'отчество': 'Ксенофонтовна',
#  'фамилия': 'Доманская',
#  'пол': 'женский',
#  'дата рождения': datetime.date(1931, 6, 10),
#  'мобильный': '+79080776976'}