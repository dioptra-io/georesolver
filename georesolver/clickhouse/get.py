from georesolver.clickhouse.main import Query

from georesolver.common.settings import ClickhouseSettings


class GetTables(Query):
    def statement(self, **kwargs) -> str:
        return f"""
        SELECT
            name
        FROM
            system.tables
        WHERE
            database == '{ClickhouseSettings().CLICKHOUSE_DATABASE}'
        """


class GetVPs(Query):
    def statement(self, table_name: str, is_anchor: bool = False) -> str:
        if is_anchor:
            anchor_statement = "WHERE is_anchor == true"
        else:
            anchor_statement = ""
        return f"""
        SELECT
            toString(address_v4) as addr,
            toString(subnet_v4) as subnet,
            asn_v4 as asn,
            toString(bgp_prefix) as bgp_prefix,
            country_code,
            lat,
            lon,
            id,
            is_anchor
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {anchor_statement}
        """


class GetIPv6VPs(Query):
    def statement(self, table_name: str, is_anchor: bool = False) -> str:
        if is_anchor:
            anchor_statement = "WHERE is_anchor == true"
        else:
            anchor_statement = ""
        return f"""
        SELECT
            toString(address_v6) as addr,
            toString(subnet_v6) as subnet,
            asn_v6 as asn,
            toString(bgp_prefix) as bgp_prefix,
            country_code,
            lat,
            lon,
            id,
            is_anchor
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {anchor_statement}
        """


class GetSubnets(Query):
    def statement(self, table_name: str, **kwargs) -> str:

        subnet_filter = kwargs.get("subnet_filter")
        if subnet_filter:
            subnet_filter = "".join([f",'{s}'" for s in subnet_filter])[1:]
            subnet_filter = f"WHERE subnet IN ({subnet_filter})"
        else:
            subnet_filter = ""

        return f"""
        SELECT 
            DISTINCT toString(subnet) as subnet
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {subnet_filter}
        """


class GetTargets(Query):
    def statement(self, table_name: str, threshold: int = 300, **kwargs) -> str:
        filter_latency_statement = ""
        if threshold:
            filter_latency_statement = f"AND min < {threshold}"
        return f"""
        SELECT 
            DISTINCT toString(dst_addr) as target
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1
            {filter_latency_statement}
        """


class GetVPsSubnets(Query):
    def statement(self, table_name: str, is_anchor: bool = False) -> str:
        if is_anchor:
            anchor_statement = "WHERE is_anchor == true"
        else:
            anchor_statement = ""

        return f"""
        SELECT 
            DISTINCT toString(subnet_v4) as subnet
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {anchor_statement}
        """


class GetTargetScore(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if subnet_filter := kwargs.get("subnet_filter"):
            subnet_filter = "".join([f",toIPv4('{s}')" for s in subnet_filter])[1:]
            subnet_filter = f"WHERE subnet IN ({subnet_filter})"
        else:
            raise RuntimeError(f"Named argument subnet_filter required for {__class__}")

        return f"""
        SELECT 
            subnet, 
            arraySort(x -> -x.2, groupArray((vp_subnet, score))) AS vps_score
            FROM 
                {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name} 
            {subnet_filter}    
            GROUP BY 
                subnet
            
        """


class GetDstPrefix(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        latency_clause = ""
        if "latency_threshold" in kwargs:
            latency_clause = f"AND min < {kwargs['latency_threshold']}"

        return f"""
        SELECT 
            DISTINCT toString(dst_prefix) as dst_prefix
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1
            {latency_clause}
        """


class GetVPsIds(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            DISTINCT prb_id
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        """


class GetPingsPerTargetWithID(Query):
    def statement(
        self, table_name: str, filtered_vps: list = [], nb_packets: int = -1, **kwargs
    ) -> str:
        filter_vps_statement = ""
        filtered_vps_ids = [id for id, _ in filtered_vps]
        filtered_vps_addr = [addr for _, addr in filtered_vps]

        if filtered_vps:
            in_clause_id = f"".join([f",{id}" for id in filtered_vps_ids])[1:]
            in_clause_ip = f"".join(
                [f",toIPv4('{addr}')" for addr in filtered_vps_addr]
            )[1:]

            filter_vps_statement = f"AND prb_id not in ({in_clause_id}) AND dst_addr not in ({in_clause_ip})"

        return f"""
        SELECT 
            DISTINCT toString(dst_addr) as target,
            groupArray((toString(src_addr), prb_id, min)) as pings
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1
            AND dst_addr != src_addr
            AND src_prefix != dst_prefix
            {filter_vps_statement}
        GROUP BY 
            target
        """


class GetPingsPerTarget(Query):
    def statement(
        self, table_name: str, filtered_vps: list = [], nb_packets: int = -1, **kwargs
    ) -> str:
        filter_vps_statement = ""
        if filtered_vps:
            in_clause = f"".join([f",toIPv4('{p}')" for p in filtered_vps])[1:]
            filter_vps_statement = (
                f"AND dst_addr not in ({in_clause}) AND src_addr not in ({in_clause})"
            )

        limit_statement = ""
        if limit := kwargs.get("limit"):
            limit_statement = f"LIMIT {limit}"

        if nb_packets == -1:
            query = f"""
            SELECT 
                DISTINCT toString(dst_addr) as target,
                groupArray((toString(src_addr), min)) as pings
            FROM 
                {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
            WHERE
                min > -1
                AND dst_addr != src_addr
                AND src_prefix != dst_prefix
                {filter_vps_statement}
            GROUP BY 
                target
            {limit_statement}
            """
        else:
            query = f"""
            SELECT 
                DISTINCT toString(dst_addr) as target,
                groupArray((toString(src_addr), rtts[1])) as pings
            FROM 
                {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
            WHERE
                rtts[1] > -1
                AND dst_addr != src_addr
                AND src_prefix != dst_prefix
                {filter_vps_statement}
            GROUP BY 
                target
            {limit_statement}
            """
        return query


class GetPingsPerTargetExtended(Query):
    def statement(
        self,
        table_name: str,
        filtered_vps: list = [],
        latency_threshold: int = 300,
        **kwargs,
    ) -> str:
        filter_vps_statement = ""
        filtered_vps_ids = [id for id, _ in filtered_vps]
        filtered_vps_addr = [addr for _, addr in filtered_vps]

        if filtered_vps:
            in_clause_id = f"".join([f",{id}" for id in filtered_vps_ids])[1:]
            in_clause_ip = f"".join(
                [f",toIPv4('{addr}')" for addr in filtered_vps_addr]
            )[1:]

            filter_vps_statement = f"AND prb_id not in ({in_clause_id}) AND dst_addr not in ({in_clause_ip})"

        filter_latency_statement = ""
        if latency_threshold:
            filter_latency_statement = f"AND min < {latency_threshold}"

        limit_statement = ""
        if limit := kwargs.get("limit"):
            limit_statement = f"LIMIT {limit}"

        return f"""
        SELECT 
            toString(dst_addr) as target,
            groupArray((toString(src_addr), prb_id, min)) as pings
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1 
            AND dst_addr != src_addr
            AND src_prefix != dst_prefix
            {filter_vps_statement}
            {filter_latency_statement}
        GROUP BY 
            target
        {limit_statement}
        """


class GetPingsPerVP(Query):
    def statement(
        self, table_name: str, filtered_vps: list = [], threshold: int = 300, **kwargs
    ) -> str:
        filter_vps_statement = ""
        filtered_vps_ids = [id for id, _ in filtered_vps]
        filtered_vps_addr = [addr for _, addr in filtered_vps]

        if filtered_vps:
            in_clause_id = f"".join([f",{id}" for id in filtered_vps_ids])[1:]
            in_clause_ip = f"".join(
                [f",toIPv4('{addr}')" for addr in filtered_vps_addr]
            )[1:]

            filter_vps_statement = f"AND prb_id not in ({in_clause_id}) AND dst_addr not in ({in_clause_ip})"

        filter_latency_statement = ""
        if threshold:
            filter_latency_statement = f"AND min < {threshold}"

        limit_statement = ""
        if limit := kwargs.get("limit"):
            limit_statement = f"LIMIT {limit}"

        return f"""
        SELECT 
            toString(src_addr) as vp_addr,
            prb_id as vp_id,
            groupArray((toString(dst_addr), min)) as pings
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1 
            AND dst_addr != src_addr
            AND src_prefix != dst_prefix
            {filter_vps_statement}
            {filter_latency_statement}
        GROUP BY 
            (src_addr, vp_id)
        {limit_statement}
        """


class GetCachedTargets(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        target_filter = ""
        if filtered_targets := kwargs.get("filtered_targets"):
            target_filter = "".join([f",toIPv4('{s}')" for s in filtered_targets])[1:]
            target_filter = f"WHERE dst_addr IN ({target_filter})"

        return f"""
        SELECT 
            DISTINCT toString(dst_addr) as target
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {target_filter}
        """


class GetShortestPingResults(Query):
    def statement(self, table_name: str, removed_vps: list[str]) -> str:
        filter_vps_statement = ""
        filtered_vps_ids = [id for id, _ in removed_vps]
        filtered_vps_addr = [addr for _, addr in removed_vps]

        if removed_vps:
            in_clause_id = f"".join([f",{id}" for id in filtered_vps_ids])[1:]
            in_clause_ip = f"".join(
                [f",toIPv4('{addr}')" for addr in filtered_vps_addr]
            )[1:]

            filter_vps_statement = f"AND prb_id not in ({in_clause_id}) AND dst_addr not in ({in_clause_ip})"

        return f"""
        SELECT
            dst_addr,
            arrayReduce(
                'argMin', 
                arrayMap(
                    x -> (x.1, x.3), 
                    groupUniqArray((src_addr, min, prb_id, msm_id))
                ), 
                arrayMap(
                    x -> (x.2), 
                    groupUniqArray((src_addr, min, prb_id,  msm_id))
                )
            ) AS shortest_ping,
            arrayMin(
                x -> (x.2), 
                groupUniqArray((src_addr, min, prb_id, msm_id))
            ) AS min_rtt
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE 
            min > -1
            {filter_vps_statement}
        GROUP BY 
            dst_addr
        """


class GetLastMileDelayPerId(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            prb_id,
            arrayMin(groupArray(min)) as min_rtt
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1 
            AND dst_addr != src_addr
        GROUP BY 
            prb_id
        """


class GetLastMileDelay(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            toString(src_addr) as vp_addr,
            arrayMin(groupArray(min)) as min_rtt
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1 
            AND dst_addr != src_addr
        GROUP BY 
            src_addr
        """


class GetPingsPerSrcDst(Query):
    def statement(
        self,
        table_name: str,
        filtered_vps: list[str] = [],
        threshold: int = 300,
    ) -> str:
        filter_vps_statement = ""
        if filtered_vps:
            in_clause = f"".join([f",toIPv4('{p}')" for p in filtered_vps])[1:]
            filter_vps_statement = (
                f"AND dst_addr not in ({in_clause}) AND src_addr not in ({in_clause})"
            )
        return f"""
        WITH  
            arrayMin(groupArray(min)) AS min_rtt 
        SELECT 
            prb_id,
            toString(dst_addr) as dst,
            min_rtt
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE
            min > -1 
            AND dst_addr != src_addr
            AND dst_prefix != src_prefix
            AND min < {threshold}
            {filter_vps_statement}
        GROUP BY 
            (prb_id, dst)
        """


class GetHostnames(Query):
    def statement(self, table_name, **kwargs):
        return f"""
        SELECT
            DISTINCT hostname
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        """


class GetAllNameServers(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            hostname,
            groupUniqArray(name_server) as name_servers
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        GROUP BY
            hostname
        """


class GetHostnamesAnswerSubnet(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"WHERE hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""
        return f"""
        SELECT 
            hostname,
            groupUniqArray(toString(answer)) as answers
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {hostname_filter}
        GROUP BY hostname
        """


class GetDNSMapping(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            toString(subnet) as subnet, 
            hostname as hostname, 
            toString(answer) as answer,
            toString(answer_bgp_prefix) as answer_bgp_prefix
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name} 
        """


class GetDNSMappingHostnames(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""

        if subnet_filter := kwargs.get("subnet_filter"):
            subnet_filter = "".join([f",toIPv4('{s}')" for s in subnet_filter])[1:]
            subnet_filter = f"subnet IN ({subnet_filter})"
        else:
            subnet_filter = ""

        where_clause = ""
        if subnet_filter or hostname_filter:
            where_clause = f"WHERE {subnet_filter} {hostname_filter}"

        return f"""
        SELECT
            toString(subnet) as client_subnet,
            hostname,
            source_scope,
            groupUniqArray((answer)) as answers,
            groupUniqArray((answer_subnet)) as answer_subnets,
            groupUniqArray((answer_bgp_prefix)) as answer_bgp_prefixes,
            groupUniqArray((answer_asn)) as answer_asns
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {where_clause}
        GROUP BY
            (subnet, hostname, source_scope)
        """


class GetDNSMappingAnswers(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        return f"""
        SELECT
            DISTINCT answer
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        """


class GetAnswersPerHostname(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        return f"""
        SELECT
            hostname,
            groupUniqArray(answer) as answers
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        GROUP BY
            hostname
        """


class GetDNSMappingPerHostnames(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""

        if subnet_filter := kwargs.get("subnet_filter"):
            subnet_filter = "".join([f",toIPv4('{s}')" for s in subnet_filter])[1:]
            subnet_filter = f"subnet IN ({subnet_filter})"
        else:
            raise RuntimeError(f"Named argument subnet_filter required for {__class__}")

        return f"""
        SELECT
            toString(subnet) as client_subnet,
            hostname,
            groupUniqArray((answer_bgp_prefix)) as answer_bgp_prefixes
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        WHERE 
            {subnet_filter}
            {hostname_filter}
        GROUP BY
            (client_subnet, hostname)
        """


class GetECSResults(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if subnet_filter := kwargs.get("subnet_filter"):
            subnet_filter = "".join([f",toIPv4('{s}')" for s in subnet_filter])[1:]
            subnet_filter = f"subnet IN ({subnet_filter})"
        else:
            subnet_filter = ""

        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""

        where_clause = ""
        if subnet_filter or hostname_filter:
            where_clause = f"WHERE {subnet_filter} {hostname_filter}"

        return f"""
        SELECT
            toString(subnet) as client_subnet,
            hostname,
            groupUniqArray((answer_subnet)) as answer_subnets
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        {where_clause}
        GROUP BY
            (subnet, hostname)
        """


class GetMeasurementIds(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT
            distinct(
                msm_id
            ) as msm_id
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name} 
        WHERE
            min > -1
        """
