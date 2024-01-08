"""Insert ping measurements into clickhouse db"""
from uuid import uuid4
from tqdm import tqdm
from time import sleep
from datetime import datetime  # TODO: find correct format for date in clickhouse
from loguru import logger

from common.clickhouse import ClickhouseInsert

class InsertPingMeasurement(ClickhouseInsert):
    def create_table_statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = "src_addr, src_netmask, prb_id, msm_id, dst_addr, proto, rcvd, sent, min, max, avg, rtts"
        return f"""
        CREATE TABLE IF NOT EXISTS {self.settings.CLICKHOUSE_DB}.{table_name}
        (
            timestamp          UInt16,
            src_addr           IPv4,
            src_prefix         IPv4,
            src_netmask        UInt8,
            prb_id             UInt16,
            msm_id             UInt32, 
            dst_addr           IPv4,
            dst_prefix         IPv4,
            proto              String,
            rcvd               UInt8,
            sent               UInt8,
            min                Float32,
            max                Float32,
            avg                Float32,
            rtts               Array(Float32)
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
