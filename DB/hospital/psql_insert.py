from psycopg import connect

from json import loads as json_loads
from pprint import pprint
from pathlib import Path
from sys import path

import generate
import queries


config_path = Path(path[1]) / 'config.json'
config = json_loads(config_path.read_text())

# Производим подключение, передаём файл с конфигурацией для авторизации 
connection = connect(**config, autocommit=True)

# Передаём запросы
with connection.cursor() as cursor:
    cursor.executemany(queries.ins_departments, generate.departments)
    cursor.executemany(queries.ins_sponsors, generate.sponsors)
    cursor.executemany(queries.ins_specializations, generate.specializations)
    cursor.executemany(queries.ins_wards, generate.wards)
    cursor.executemany(queries.ins_donations, generate.donations)
    cursor.executemany(queries.ins_doctors, generate.doctors)
    cursor.executemany(queries.ins_doctors_specs, generate.doctors_specs)
    cursor.executemany(queries.ins_vacations, generate.vacations)


