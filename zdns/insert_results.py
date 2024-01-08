from uuid import uuid4
from common.clickhouse import ClickhouseInsert

class InsertMappingDNS(ClickhouseInsert):
    verbose = False
    
    def create_table_statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = "client_subnet, client_netmask, hostname, timestamp"
        return f"""
        CREATE TABLE IF NOT EXISTS {self.settings.CLICKHOUSE_DB}.{table_name}
        (
            timestamp              DateTime(),
            client_subnet          IPv4,
            client_netmask         UInt8,
            hostname               String,
            answer                 IPv4
        )
        ENGINE MergeTree
        ORDER BY ({sorting_key})
        """
    
    def insert(
        self,
        input_data: list[str],
        output_table: str,
        drop_table: bool = False,
    ) -> None:
        """insert dns resolution json data into clickhouse db"""
        # create a temp csv file result
        tmp_file_dir = self.common_settings.TMP_PATH / f"{uuid4()}.csv"
        self.common_settings.TMP_PATH.mkdir(parents=True, exist_ok=True)
        
        if drop_table:
            drop_table_statement = self.drop_table_statement(output_table)
            self.execute_iter(drop_table_statement)

        # create table
        statement = self.create_table_statement(output_table)
        self.create_table(statement)

        # write tmp csv file
        self.write_tmp(input_data,tmp_file_dir)
        
        # insert results into db from file
        statement = self.insert_from_csv_statement(
            table_name=output_table,
            input_file_dir=tmp_file_dir,
        )
        self.execute_insert(statement)

        # remove tmp file
        tmp_file_dir.unlink()
    