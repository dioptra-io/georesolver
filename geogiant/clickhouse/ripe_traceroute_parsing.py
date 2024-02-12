from geogiant.clickhouse import Query


class GetProbeConnectivity(Query):
    def statement(self, table_name: str) -> str:
        """get avg first reply min rtt for each VP found in public traceroute"""
        return f"""
        SELECT
            src_addr,
            arrayAvg(groupUniqArray(first_reply_rtt)) as connectivity
        FROM (
            SELECT
                src_addr,
                dst_addr,
                arraySort(x -> x.2, groupUniqArray((reply_addr, ttl, min)))[1].3 as first_reply_rtt
            FROM
                {self.settings.DATABASE}.{table_name}
            GROUP BY
                (src_addr, dst_addr)    
        )
        GROUP BY
            src_addr
        """


class GetGeolocFromTraceroute(Query):
    def statement(self, table_name: str, threshold: int = 2) -> str:
        """get all IP address at less than 'threshold' latency and associate its coordinates with closest VP"""
        return f"""
        SELECT
            toString(src_addr) as src_addr,
            toString(reply_addr) as reply_addr,
            arrayMin(groupArray(min)) as min_rtt
        FROM
            {self.settings.DATABASE}.{table_name}
        WHERE
            min < {threshold}
        GROUP BY
            (src_addr, reply_addr)
        """
