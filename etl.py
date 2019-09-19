import psycopg2 as pg
import yaml
from pathlib import Path
import os
import pandas as pd

def create_tables(config, connection):
    cur = connection.cursor()
    for table in config:
        name = table.get('name')
        schema = table.get('schema')
        ddl = f"""CREATE TABLE IF NOT EXISTS {name} ({schema})"""
        cur.execute(ddl)

    connection.commit()

def transform_tables(config):

    data_path = "path_to_data_here"

    for table in config:
        table_name = table.get('name')
        table_source = data_path.joinpath(f"{table_name}.csv")
        table_cols = []
        for i in table.get('columns'):
            table_cols.append(str.upper(i))
        df = pd.read_csv(table_source)
        df_reorder = df[table_cols]  # rearrange column here
        df_reorder.to_csv(table_source, index=False)

def load_tables(config, connection):

    # iterate and load
    cur = connection.cursor()
    data_path = "path_to_data_here"

    for table in config:
        table_name = table.get('name')
        table_source = data_path.joinpath(f"{table_name}.csv")
        with open(table_source, 'r') as f:
            next(f)
            cur.copy_expert(f"COPY {table_name} FROM STDIN CSV NULL AS ''", f)
        connection.commit()


connection = pg.connect(
    host='localhost',
    port=54320,
    dbname='ht_db',
    user='postgres'
)

with open("path_to_schema.yml_here") as schema_file:
    config = yaml.load(schema_file)
create_tables(config, connection)
transform_tables(config)
load_tables(config, connection)
