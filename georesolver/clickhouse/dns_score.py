"""all count classes for extracting metrics"""

from georesolver.clickhouse import Query, NativeQuery

from georesolver.common.settings import ClickhouseSettings, PathSettings

clickhouse_settings = ClickhouseSettings()
path_settings = PathSettings()


class OverallScore(Query):
    def statement(
        self,
        table_name: str,
        **kwargs,
    ) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""
        if target_filter := kwargs.get("target_filter"):
            target_filter = "".join([f",toIPv4('{t}')" for t in target_filter])[1:]
        else:
            target_filter = ""
        if column_name := kwargs.get("column_name"):
            column_name = column_name
        else:
            raise RuntimeError(f"Column name parameter missing for {__class__}")

        # select element from (select element, count(*) as occurrence from (select arrayJoin(['A', 'B', 'B', 'C']) AS element) group by element order by occurrence DESC LIMIT 1);

        # output[target] = [(vp_addr, float)]

        # WITH (WITH groupUniqArray((hostname, answer_bgp_prefix)) as hostname_bgp_prefixes,
        # groupUniqArray(hostname) as hostnames,
        # arrayMap(x->(x, arrayFilter(y->y.1=x, hostname_bgp_prefixes)), hostnames) as hostname_bgp_prefixes_map
        # SELECT groupArray(vp_mapping) FROM (
        # SELECT (subnet, hostname_bgp_prefixes_map) as vp_mapping FROM test_clustering
        # GROUP BY subnet)) as vps_mapping,
        # groupUniqArray((hostname, answer_bgp_prefix)) as hostname_bgp_prefixes,
        # groupUniqArray(hostname) as hostnames,
        # arrayMap(x->(x, arrayFilter(y->y.1=x, hostname_bgp_prefixes)), hostnames) as hostname_bgp_prefixes_map,
        # arrayMap(x->(x.1, arrayIntersect(x.2, arrayFilter(y->, vps_mapping)
        # SELECT subnet, hostname_bgp_prefixes_map, vps_mapping FROM test_clustering
        # GROUP BY subnet
        # LIMIT 10

        return f"""
        WITH groupArray((toString(subnet_2), score)) AS subnet_scores
        SELECT
            toString(subnet_1) as target_subnet,
            arraySort(x -> -(x.2), subnet_scores) as scores
        FROM
        (
            SELECT
                t1.client_subnet AS subnet_1,
                t2.client_subnet AS subnet_2,
                length(arrayIntersect(t1.mapping, t2.mapping)) / least(length(t1.mapping), length(t2.mapping)) AS score
            FROM
            (
                SELECT
                    client_subnet,
                    groupUniqArray({column_name}) AS mapping
                FROM {self.settings.CLICKHOUSE_DATABASE}.{table_name}
                WHERE 
                    client_subnet IN ({target_filter})
                    {hostname_filter}
                    -- AND pop_ip_info_id != -1
                GROUP BY client_subnet
            ) AS t1
            CROSS JOIN
            (
                SELECT
                    client_subnet,
                    groupUniqArray({column_name}) AS mapping
                FROM {self.settings.CLICKHOUSE_DATABASE}.{table_name}
                WHERE 
                    client_subnet NOT IN ({target_filter})
                    {hostname_filter}
                    -- AND pop_ip_info_id != -1
                GROUP BY client_subnet
            ) AS t2
            WHERE t1.client_subnet != t2.client_subnet
        )
        WHERE subnet_1 IN ({target_filter})
        GROUP BY subnet_1
        """


class HostnameScore(Query):
    def statement(
        self,
        table_name: str,
        **kwargs,
    ) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""
        if targets := kwargs.get("targets"):
            targets = "".join([f",toIPv4('{t}')" for t in targets])[1:]
            targets = f"subnet IN ({targets})"
        else:
            targets = ""
        if target_filter := kwargs.get("target_filter"):
            target_filter = "".join([f",toIPv4('{t}')" for t in target_filter])[1:]
            target_filter = f"subnet NOT IN ({target_filter})"
        else:
            target_filter = ""
        if column_name := kwargs.get("column_name"):
            column_name = column_name
        else:
            raise RuntimeError(f"Column name parameter missing for {__class__}")

        return f"""
        WITH groupArray((toString(subnet_2), score)) AS subnet_scores
        SELECT
            toString(subnet_1) as target_subnet,
            arraySort(x -> -(x.2), subnet_scores) as scores
        FROM
        (
            SELECT
                t1.subnet AS subnet_1,
                t2.subnet AS subnet_2,
                length(arrayIntersect(t1.mapping, t2.mapping)) / least(length(t1.mapping), length(t2.mapping)) AS score
            FROM
            (
                SELECT
                    subnet,
                    hostname,
                    groupUniqArray({column_name}) AS mapping
                FROM {self.settings.CLICKHOUSE_DATABASE}.{table_name}
                WHERE 
                    {targets}
                    {hostname_filter}
                GROUP BY (subnet, hostname)
            ) AS t1
            CROSS JOIN
            (
                SELECT
                    subnet,
                    hostname,
                    groupUniqArray({column_name}) AS mapping
                FROM {self.settings.CLICKHOUSE_DATABASE}.{table_name}
                WHERE 
                    {target_filter}
                    {hostname_filter}
                GROUP BY (subnet, hostname)
            ) AS t2
            WHERE t1.subnet != t2.subnet
        )
        GROUP BY subnet_1
        """


class CountRows(NativeQuery):
    def statement(
        self,
        table_name: str,
    ) -> str:
        return f"""
        SELECT
            count(distinct *)
        FROM 
            {self.settings.CLICKHOUSE_DATABASE}.{table_name}
        """
