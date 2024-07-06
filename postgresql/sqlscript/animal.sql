create table if not exists animal
(
  id bigserial primary key,
  time_start timestamp,
  time_end timestamp,
  animal_class varchar(255),
  mark varchar(255),
  nil varchar(255),
  animal_count integer
);