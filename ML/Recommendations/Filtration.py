from pandas import read_csv, DataFrame

from pathlib import Path
from re import compile
from sys import path


script_dir = Path(path[0])


movies = read_csv(script_dir / 'movies.csv', index_col='movie_id') 
ratings = read_csv(script_dir / 'ratings.csv')

# Шаблон для названий
title_pat = compile(
    r'(?P<title>.+?) ?'
    r'(\((?P<loc_title>[^()]+)\) )?'
    r'(\((?P<year>\d{4})\))?'
)

genres = DataFrame(columns=['movie_id', 'genre'])

for i in movies.index:
    # Обработка заголовка
    title = movies.loc[i, 'title'].rstrip()
    mo = title_pat.fullmatch(title)
    movies.loc[i, 'title'] = mo['title']
    # Местное название 
    movies.loc[i, 'loc_title'] = mo['loc_title'].strip(' ()') if type(mo['loc_title']) is str else None 
    movies.loc[i, 'year'] = int(mo['year']) if type(mo['year']) is str else None
    # Обработка жанров
    for g, genre in enumerate(movies.loc[i, 'genres'].rstrip().split('|'), 1):
        genres.loc[f'{i},{g}'] = i, genre

genres.reset_index(drop=True, inplace=True) # удаление двойных индексов

genres = genres.query("genre not in ['IMAX', '(no genres listed)']") # удаление объектов, которые не являются жанрами 

movies.drop(columns='genres', inplace=True)
movies = movies.convert_dtypes()

ratings.drop(columns='timestamp', inplace=True)


movies.to_csv(script_dir / 'movies_ref.csv')
genres.to_csv(script_dir / 'genres_ref.csv')
ratings.to_csv(script_dir / 'ratings_ref.csv')

print(
    '',
    movies,
    genres,
    ratings,
    sep='\n\n'
)

                                       # title loc_title  year
# movie_id
# 1                                  Toy Story      <NA>  1995
# 2                                    Jumanji      <NA>  1995
# 3                           Grumpier Old Men      <NA>  1995
# 4                          Waiting to Exhale      <NA>  1995
# 5                Father of the Bride Part II      <NA>  1995
# ...                                      ...       ...   ...
# 193581    Black Butler: Book of the Atlantic      <NA>  2017
# 193583                 No Game No Life: Zero      <NA>  2017
# 193585                                 Flint      <NA>  2017
# 193587          Bungo Stray Dogs: Dead Apple      <NA>  2018
# 193609          Andrew Dice Clay: Dice Rules      <NA>  1991

# [9742 rows x 3 columns]

       # movie_id      genre
# 0             1  Adventure
# 1             1  Animation
# 2             1   Children
# 3             1     Comedy
# 4             1    Fantasy
# ...         ...        ...
# 22079    193583    Fantasy
# 22080    193585      Drama
# 22081    193587     Action
# 22082    193587  Animation
# 22083    193609     Comedy

# [21892 rows x 2 columns]

        # user_id  movie_id  rating
# 0             1         1     4.0
# 1             1         3     4.0
# 2             1         6     4.0
# 3             1        47     5.0
# 4             1        50     5.0
# ...         ...       ...     ...
# 100831      610    166534     4.0
# 100832      610    168248     5.0
# 100833      610    168250     5.0
# 100834      610    168252     5.0
# 100835      610    170875     3.0

# [100836 rows x 3 columns]
