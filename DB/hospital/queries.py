# Шаблоны запросов добавления 

ins_departments = (
    'insert into departments '
    '  (name) ' # имя столбца 
    'values '
    '  (%s) ' # здесь будет происходить подстановка данных вместо "%s"
)
ins_sponsors = (
    'insert into sponsors '
    '  (name) '
    'values '
    '  (%s) '
)
ins_specializations = (
    'insert into specializations '
    '  (name) '
    'values '
    '  (%s) '
)
ins_wards = (
    'insert into wards '
    '  (dep_id, name) '
    'values '
    '  (%s, %s) '
)
ins_donations = (
    'insert into donations '
    '  (sponsor_id, dep_id, date, amount) '
    'values '
    '  (%s, %s, %s, %s) '
)
ins_doctors = (
    'insert into doctors '
    '  (dep_id, last_name, first_name, patr_name, salary, premium) '
    'values '
    '  (%s, %s, %s, %s, %s, %s) '
)
ins_doctors_specs = (
    'insert into doctors_specs '
    '  (doctor_id, spec_id) '
    'values '
    '  (%s, %s) '
)
ins_vacations = (
    'insert into vacations '
    '  (doctor_id, start_date, end_date) '
    'values '
    '  (%s, %s, %s) '
)

# Шаблоны запросов выборки 

# Сумма пожертвований за каждый год 
sel_donations_by_year = '''
    select
      extract(year from "date") as year, 
      sum(amount)
    from
      donations
    group by
      year
    order by
      year
'''

# Сумма пожертвований за каждый год по каждому отделению
sel_donations_by_year_and_dep = '''
    select
      extract(year from "date") as year,
      dep_id,
      sum(amount)
    from
      donations
    group by
      year, 
      dep_id
    order by
      year,
      dep_id
'''
# Считает количество отпускных больше недели
sel_count_week_vacations = '''
    select
      case when end_date - start_date <= 7 then 'неделя'
           else 'больше недели'
      end as week_or_more,
      count(*)
    from 
      vacations
    group by
      week_or_more
'''

# Выводит название палаты и название отделения, к котором относится 
sel_wards_departments = '''
    select 
      wards.name,
      departments.name
    from
      wards
    join
      departments on dep_id = departments.id
'''

# Выводит название отделения и названия палат, которые относятся к нему 
sel_groupconcat_wards_departments = '''
      select d.name as "отделение",
             string_agg(w.name, ', ') as "палаты"
        from wards as w
        join departments as d
          on dep_id = d.id
    group by d.name
'''

# Выводит суммы пожертвований больше 5000, названия отделений, спонсоров и даты
sel_large_donations = '''
      select d.name as "отделение",
             s.name as "спонсор",
             date as "дата",
             amount as "сумма"
        from departments as d
        join donations
          on d.id = dep_id and d.id between 2 and 5
        join sponsors as s
          on sponsor_id = s.id and amount > 500000
    order by "отделение", "сумма"
'''

# Выводит ФИО врача и его специальность(-и), если есть 
sel_all_doctors_specs = '''
       select concat_ws(' ', last_name, first_name, patr_name) as "ФИО",
              string_agg(s.name, ', ') as "Специальность"
         from doctors as d
    left join doctors_specs on d.id = doctor_id
    left join specializations as s on spec_id = s.id
     group by "ФИО"
     order by "ФИО"
'''

# Выводит название специальностей и количество врачей, которые имеет эту специальность
sel_doctors_cnt_for_spec = '''
        select s.name as "Специальность",
               count(doctor_id) as "Кол-во врачей"
          from doctors_specs
    right join specializations as s on spec_id = s.id
      group by s.name
      order by "Кол-во врачей" desc;
'''

# Выводит названия отделений и количество отпусков для каждого за последние два года
sel_vac_cnt_for_last_two_years = '''
    select 
      name,
      count(v.id)
    from 
      vacations as v
    join 
      doctors as d on doctor_id = d.id 
                   and extract(year from start_date) >= extract(year from current_date) - 2
    join 
      departments as dep on dep_id = dep.id
    group by 
      name
    order by
      name;
'''

# Выводит среднее количество отпусков 
sel_avg_vacations_cnt = '''
    select 
      round(avg(subq.cnt), 1) as "average vacations"
    from (
      select
        doctor_id,
        count(*) as cnt
      from 
        vacations
      group by
        doctor_id
    ) as subq
'''

# Выводит самые большие доходы врачей по каждой специальности 
sel_max_income_by_spec = '''
    with max_income_by_spec as (
    select 
        coalesce(s.name, 'Младший персонал') as spec,
        max(d.salary + d.premium) as max_income
    from doctors as d
    left join doctors_specs as ds on d.id = ds.doctor_id
    left join specializations as s on ds.spec_id = s.id
    group by spec
    )
    
    select 
    concat_ws(' ', d.last_name, d.first_name, d.patr_name) as "ФИО",
    (d.salary + d.premium) as "Доход",
    coalesce(s.name, 'Младший персонал') as "Специальность"
    from doctors as d
    left join doctors_specs as ds on d.id = ds.doctor_id
    left join specializations as s on ds.spec_id = s.id
    join max_income_by_spec as m on 
    coalesce(s.name, 'Младший персонал') = m.spec and
    (d.salary + d.premium) = m.max_income
    order by "Доход" desc;
'''

# Выводит оклад врачей ниже среднего 
sel_doctors_salary_lt_avg = '''
    select
      concat_ws(' ', last_name, first_name, patr_name) as full_name,
      salary
    from
      doctors
    where salary < (select avg(salary) from doctors)
'''



sel_sepcs = 'select name from specializations'

# Выводит названия отделений с перчнем специализаций врачей, которые там работаю 
sel_dep_specs = '''
    select
      subq.dep_name,
      string_agg(subq.spec, ', ') as specs
    from (
      select distinct
        dep.name as dep_name,
        s.name as spec
      from
        departments as dep
      join
        doctors as d on dep.id = dep_id
      join
        doctors_specs on d.id = doctor_id
      join
        specializations as s on spec_id = s.id
      order by
        dep.name,
        s.name
    ) as subq
    group by
      subq.dep_name
'''

