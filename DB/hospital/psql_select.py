from psycopg import connect

from json import loads as json_loads
from pprint import pprint
from pathlib import Path
from sys import path

import queries


config_path = Path(path[0]) / 'config.json'
config = json_loads(config_path.read_text())

connection = connect(**config, autocommit=True)

# Запросы выборки 
with connection.cursor() as cursor:
    donations_by_year = cursor.execute(queries.sel_donations_by_year).fetchall()    
    count_week_vacations = cursor.execute(queries.sel_count_week_vacations).fetchall()    
    groupconcat_wards_departments = cursor.execute(queries.sel_groupconcat_wards_departments)\
                                          .fetchall()    
    large_donations = cursor.execute(queries.sel_large_donations).fetchall()    
    all_doctors_specs = cursor.execute(queries.sel_all_doctors_specs).fetchall()    
    doctors_cnt_for_spec = cursor.execute(queries.sel_doctors_cnt_for_spec).fetchall()
    vac_cnt_for_last_two_years = cursor.execute(queries.sel_vac_cnt_for_last_two_years)\
                                       .fetchall()
    avg_vacations_cnt = cursor.execute(queries.sel_avg_vacations_cnt).fetchall()
    max_income_by_spec = cursor.execute(queries.sel_max_income_by_spec).fetchall()                                 
    doctors_salary_lt_avg = cursor.execute(queries.sel_doctors_salary_lt_avg).fetchall()                                 
    
    dep_specs = cursor.execute(queries.sel_dep_specs).fetchall()
    specs = cursor.execute(queries.sel_sepcs).fetchall()


from itertools import chain

specs = set(chain(*specs))

# Список с отделениями и специализациями, которые там ЕСТЬ или НЕТ   
dep_all_specs = []
for dep, d_specs in dep_specs:
    dep_all_specs.append((dep, f'ЕСТЬ: {d_specs}'))
    miss = sorted(specs - set(d_specs.split(', ')))
    dep_all_specs.append((dep, f'НЕТ: {", ".join(miss)}'))

# >>> pprint(dep_all_specs[:3])
# [('Кардиологическое отделение',
#   'ЕСТЬ: Гастроэнтеролог, Дерматолог, Иммунолог, Кардиолог, Стоматолог, '
#   'Хирург, Эндокринолог'),
#  ('Кардиологическое отделение',
#   'НЕТ: Анестезиолог, Диетолог, Нарколог, Невролог, Онколог, Ортопед, '
#   'Оториноларинголог, Офтальмолог, Реаниматолог, Ревматолог, Терапевт, '
#   'Травматолог, Уролог'),
#  ('Неврологическое отделение',
#   'ЕСТЬ: Гастроэнтеролог, Дерматолог, Нарколог, Невролог, Реаниматолог, '
#   'Уролог')]

# >>> pprint(donations_by_year)
# [(Decimal('2014'), Decimal('2488921.11')),
#  (Decimal('2015'), Decimal('1038003.75')),
#  (Decimal('2016'), Decimal('1235262.14')),
#  (Decimal('2017'), Decimal('577262.97')),
#  (Decimal('2018'), Decimal('131079.10')),
#  (Decimal('2019'), Decimal('1820062.00')),
#  (Decimal('2020'), Decimal('3095045.63')),
#  (Decimal('2021'), Decimal('1807173.94')),
#  (Decimal('2022'), Decimal('1486391.72')),
#  (Decimal('2023'), Decimal('3074407.02')),
#  (Decimal('2024'), Decimal('670655.04'))]

# >>> pprint(count_week_vacations)
# [('неделя', 40), ('больше недели', 210)]

# >>> pprint(groupconcat_wards_departments)
# [('Кардиологическое отделение', 'КО-2, КО-1, КО-3'),
#  ('Неврологическое отделение', 'НО-4, НО-3, НО-1, НО-2'),
#  ('Отделение общей терапии', 'ООТ-2, ООТ-1, ООТ-3'),
#  ('Отделение функциональной диагностики', 'ОФД-3, ОФД-1, ОФД-2, ОФД-4'),
#  ('Реанимация и интенсивная терапия', 'РИТ-1, РИТ-2, РИТ-3, РИТ-4'),
#  ('Токсикологическое отделение', 'ТО-1, ТО-2, ТО-3, ТО-4'),
#  ('Физиотерапевтическое отделение', 'ФО-1, ФО-2')]