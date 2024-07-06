create table if not exists file_tb
(
  id bigserial primary key,
  path_to_file varchar(255),
  file_type varchar(255),
  file_name varchar(255),
    
  dat timestamp,
  path_to_result_file varchar(255)
);
