

import pandas as pd
from typing import Dict, List, Tuple
from relbench.datasets import get_dataset
from relbench.tasks import get_task


class RelBenchDataSource:


    def __init__(self, dataset_name: str, task_name: str):

        self.dataset = get_dataset(dataset_name)
        self.task = get_task(dataset_name, task_name)


        self.db = self.dataset.get_db() if hasattr(self.dataset, 'get_db') else self.dataset.db

        print(f"Loaded RelBench: {dataset_name} / {task_name}")

    def get_tables(self) -> Dict[str, pd.DataFrame]:


        tables = {}
        for table_name, table_obj in self.db.table_dict.items():

            if hasattr(table_obj, 'df'):
                df = table_obj.df
            elif isinstance(table_obj, pd.DataFrame):
                df = table_obj
            else:
                df = pd.DataFrame(table_obj)

            tables[table_name] = df

        return tables

    def get_foreign_keys(self) -> List[Tuple[str, str, str, str]]:


        foreign_keys = []

        for table_name, table_obj in self.db.table_dict.items():
            if hasattr(table_obj, 'fkey_col_to_pkey_table'):
                fkey_dict = table_obj.fkey_col_to_pkey_table

                for fk_col, target_table in fkey_dict.items():

                    target_obj = self.db.table_dict[target_table]

                    if hasattr(target_obj, 'pkey_col') and target_obj.pkey_col:
                        pk_col = target_obj.pkey_col
                    else:

                        pk_col = target_obj.df.columns[0] if hasattr(target_obj, 'df') else target_obj.columns[0]

                    foreign_keys.append((table_name, fk_col, target_table, pk_col))

        return foreign_keys

    def get_task_info(self) -> Dict:


        return {
            'task_type': self.task.task_type.value,
            'target_table': self.task.entity_table,
            'target_column': self.task.target_col,
            'train_table': self.task.get_table('train'),
            'val_table': self.task.get_table('val'),
            'test_table': self.task.get_table('test'),
        }


if __name__ == "__main__":

    source = RelBenchDataSource('rel-f1', 'driver-top3')


    tables = source.get_tables()
    foreign_keys = source.get_foreign_keys()
    task_info = source.get_task_info()

    print(f"\nTables: {list(tables.keys())}")
    print(f"Foreign keys: {len(foreign_keys)}")
    print(f"Task: {task_info['task_type']} on {task_info['target_table']}")

