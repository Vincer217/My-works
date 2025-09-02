-- Выводит площади регионов и общую площадь всех регионов вместе 

select 
  concat(Continent, ' — ', Region) as cont_reg,
  sum(SurfaceArea) as area
from 
  country
group by 
  cont_reg
union
select 
  "Total",
  sum(SurfaceArea)
from 
  country
order by
  cont_reg,
  area;



select 'минимальное городское население' as q, min(Population) from city
union
select 'максимальное городское население' as q, max(Population) from city;


-- Выводит названия стран, в которых количество городов с населением больше миллиона больше 5

select
  country.name
from
  city
join 
  country on countrycode = code 
          and city.population > 1000000
group by
  country.name
having
  count(*) > 5
order by
  country.name;


-- Выводит названия стран, в которых официальный язык только испанский и продолжительность жизни больше 70 лет 

  select c.name
    from country as c
    join countrylanguage as cl 
      on countrycode = code
     and lifeexpectancy > 70
     and isofficial = 'T'
group by c.name
  having group_concat('', language) like '%Spanish%'
     and count(*) = 1
order by c.name;


