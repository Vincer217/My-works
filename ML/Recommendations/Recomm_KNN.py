from pandas import read_csv
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

from pathlib import Path
from sys import path


script_dir = Path(path[0])
movies = read_csv(script_dir / 'movies_ref.csv', index_col='movie_id')
genres = read_csv(script_dir / 'genres_ref.csv')
ratings = read_csv(script_dir / 'ratings_ref.csv')

# genres.drop(columns='Unnamed: 0', inplace=True)

movies_ratings = ratings.pivot_table(values='rating', index='user_id', columns='movie_id') # разряженная матрица 


user_votes = ratings.groupby('user_id')['rating'].agg('count') # агрегационная функция, которая будет считать количество оценок пользователя  
movies_votes = ratings.groupby('movie_id')['rating'].agg('count') # количество оценок у каждого фильма 

user_mask = user_votes.loc[user_votes > 50].index # фильтр для того, чтобы отсеять пользователей, у которых оценок меньше 50
movies_mask = movies_votes.loc[movies_votes > 10].index # фильтр для того, чтобы отсеять фильмы, у которых оценок меньше 10


movies_ratings = movies_ratings.loc[user_mask, movies_mask] # матрица предпочтений (фильтрованная)
movies_ratings = movies_ratings.fillna(0)


movies_ratings_csc = csc_matrix(movies_ratings.values) # матрица со значениями и их координатами 

# Алгоритм к-ближайших сососедей. Метрика - косинусное расстояние
model = NearestNeighbors(
    n_neighbors=20,
    metric='cosine',
    algorithm='brute',
    n_jobs=-1,
)
model.fit(movies_ratings_csc.transpose())


recommendations = 7 # количество рекомендаций к фильму
movie_title = 'How to Train Your Dragon'

search_mask = movies['title'].str.contains(movie_title) # маска для того, чтобы найти определённый фильм 
search_results = movies[search_mask]

# Если фильм не найден и search_results пустой 
try:
    movie_id = search_results.iloc[0].name # получение строки (если это франшиза), затем ее имени - это индекс (идентификатор фильма)

except IndexError:
    print(f'отсутствует фильм с названием {movie_title!r}')

else:
    movie_id_pos = movies_ratings.columns.get_loc(movie_id) # берем идентификатор фильма и находим позиционный индекс (нужен для csc)
    movie_vector = movies_ratings_csc[:, movie_id_pos] # вектор оценок нашего фильма 
    
    closest_movies_dist, closest_movies_ind = model.kneighbors(
        movie_vector.transpose()
    ) # возвращает два вектора:  расстояния между точками, индексы ближайших фильмов  
    
    # Собираем в зип, чтобы сортировать по расстоянию индексы, так как индексы могут быть не отсортированными  
    distances_indexes = sorted(zip(
        closest_movies_dist.flatten(),
        closest_movies_ind.flatten(),
    ))[1:] # совмещаем индекс с расстоянием и убираем первый фильм, так как это фильм, к которому мы ищем близкие 

    
    recommendations_ind = []
    for _, movie_id_pos in distances_indexes:
        movie_id_ = movies_ratings.columns[movie_id_pos]
        recommendations_ind.append(movie_id_)
    
    recomm = movies.loc[recommendations_ind] # рекомендации только по оценкам пользователей 
    
# Получаем жанры исходного фильма один раз
source_genres = set(genres[genres['movie_id'] == movie_id]['genre'])

# Создаем словарь {movie_id: set(genres)} для всех рекомендованных фильмов
recomm_movies_genres = genres[genres['movie_id'].isin(recomm.index)]
genre_sets = recomm_movies_genres.groupby('movie_id')['genre'].apply(set).to_dict()

# Фильтруем
recomm_i = [i for i in recomm.index 
           if len(genre_sets.get(i, set()) & source_genres) >= 2] # не меньше двух общих жанров 
        
print(f'Рекомендации по оценкам пользователей и жанрам\n{movies.loc[recomm_i]}')          