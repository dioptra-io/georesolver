from geogiant.clickhouse.query import Query


class GetCompleVPs(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT
            *
        FROM
            {self.settings.DATABASE}.{table_name}
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
            id,
            country_code,
            asn_v4,
            lat,
            lon
        FROM
            {self.settings.DATABASE}.{table_name}
        {anchor_statement}
        """


class GetVPsAndAnchors(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT
            toString(address_v4) as address_v4,
            toString(subnet_v4) as vp_subnet,
            bgp_prefix as vp_bgp_prefix,
            country_code,
            lat,
            lon
        FROM
            {self.settings.DATABASE}.{table_name}
        """


class GetTargets(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT
            toString(address_v4) as address_v4,
            toString(subnet_v4) as target_subnet,
            lat,
            lon
        FROM
            {self.settings.DATABASE}.{table_name}
        WHERE
            is_anchor == true
        """


class GetSubnets(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            DISTINCT toString(subnet) as subnet
        FROM 
            {self.settings.DATABASE}.{table_name}
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
            {self.settings.DATABASE}.{table_name}
        {anchor_statement}
        """


class GetSubnets(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            DISTINCT toString(subnet) as subnet
        FROM 
            {self.settings.DATABASE}.{table_name}
        """


class GetDstPrefix(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        latency_clause = ""
        if "latency_threshold" in kwargs:
            latency_clause = f"WHERE min < {kwargs['latency_threshold']}"

        return f"""
        SELECT 
            DISTINCT toString(dst_prefix) as dst_prefix
        FROM 
            {self.settings.DATABASE}.{table_name}
        {latency_clause}
        """


class GetVPSInfo(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            toString(address_v4),
            groupUniqArray(
                (
                    lat, 
                    lon,
                    country_code
                )
            )
        FROM 
            {self.settings.DATABASE}.{table_name}
        GROUP BY
            toString(address_v4)
        """


class GetVPSInfoPerSubnet(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            toString(subnet_v4) as subnet,
            groupUniqArray(
                (
                    toString(address_v4),
                    lat, 
                    lon,
                    country_code
                )
            ) as vps
        FROM 
            {self.settings.DATABASE}.{table_name}
        GROUP BY
            subnet
        """


class GetSubnetPerHostname(Query):
    def statement(self, table_name: str, anycast_filter: list = None) -> str:
        if anycast_filter:
            anycast_filter = "".join([f",'{a}'" for a in anycast_filter])[1:]
            anycast_filter = f"WHERE answer_bgp_prefix NOT IN ({anycast_filter})"
        else:
            anycast_filter = ""

        return f"""
        SELECT 
            hostname,
            groupArray((answer_bgp_prefix, subnets)) AS vps_per_bgp_prefix
        FROM(
            SELECT
                hostname,
                answer_bgp_prefix,
                groupUniqArray(subnet) as subnets
            FROM 
            {self.settings.DATABASE}.{table_name}
            {anycast_filter}
            GROUP BY
                (hostname, answer_bgp_prefix)
        )
        GROUP BY
            hostname
        
        """


class GetPingsPerTarget(Query):
    def statement(self, table_name: str, filtered_vps: list = []) -> str:
        filter_vps_statement = ""
        if filtered_vps:
            in_clause = f"".join([f",toIPv4('{p}')" for p in filtered_vps])[1:]
            filter_vps_statement = (
                f"AND dst_addr not in ({in_clause}) AND src_addr not in ({in_clause})"
            )
        stat = f"""
        SELECT 
            toString(dst_addr) as target,
            groupArray((toString(src_addr), min)) as pings
        FROM 
            {self.settings.DATABASE}.{table_name}
        WHERE
            min > -1 
            AND dst_addr != src_addr
            AND src_prefix != dst_prefix
            {filter_vps_statement}
        GROUP BY 
            target
        """
        print(stat)
        return stat


class GetLastMileDelay(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            toString(src_addr) as src_addr,
            arrayMin(groupArray(min)) as min_rtt
        FROM 
            {self.settings.DATABASE}.{table_name}
        WHERE
            min > -1 
            AND toString(dst_addr) != src_addr
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
        stat = f"""
        WITH  arrayMin(groupArray(min)) as min_rtt
        SELECT 
            toString(src_addr) as src,
            toString(dst_addr) as dst,
            min_rtt
        FROM 
            {self.settings.DATABASE}.{table_name}
        WHERE
            min > -1 
            AND dst_addr != src_addr
            AND dst_prefix != src_prefix
            AND min < {threshold}
            {filter_vps_statement}
        GROUP BY 
            (src, dst)
        """
        print(stat)
        return stat


class GetPingsPerSubnet(Query):
    def statement(
        self,
        table_name: str,
    ) -> str:
        return f"""
        SELECT
            toString(dst_prefix) as subnet,
            toString(dst_addr) as target,
            groupArray((toString(src_addr), min)) as ping_to_target
        FROM 
            {self.settings.DATABASE}.{table_name}
        WHERE
            min > -1 
            AND toString(dst_addr) != toString(src_addr)
        GROUP BY 
            (subnet, target)
        """


class GetAvgRTTPerSubnet(Query):
    """base function for counting classes"""

    def statement(self, table_name: str) -> str:
        return f"""
        SELECT
            subnet,
            arraySort(
                x -> x.2,
                groupArray((vp, avg_rtt))
            ) as vps_avg_rtt
            FROM (
                SELECT 
                    toString(dst_prefix) as subnet,
                    toString(src_addr) as vp,
                    arrayAvg(groupArray(min)) as avg_rtt
                FROM 
                    {self.settings.DATABASE}.{table_name}
                WHERE 
                    min > -1 
                    AND toString(dst_addr) != vp
                GROUP BY 
                    (subnet, vp)
                ORDER BY 
                    subnet
            )
        GROUP BY
            subnet
        """


class GetAllNameServers(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            hostname,
            groupUniqArray(name_server) as name_servers
        FROM 
            {self.settings.DATABASE}.{table_name}
        GROUP BY
            hostname
        """


class GetHostnames(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            DISTINCT hostname as hostname
        FROM 
            {self.settings.DATABASE}.{table_name}
        """


class GetAllDNSMapping(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"WHERE hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""
        return f"""
        SELECT 
            *
        FROM 
            {self.settings.DATABASE}.{table_name}
        {hostname_filter}
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
            {self.settings.DATABASE}.{table_name}
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
            {self.settings.DATABASE}.{table_name} 
        """


class GetPoPInfo(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"WHERE hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""

        return f"""
        SELECT 
            hostname,
            toString(answer) as answer,
            toString(answer_subnet) as answer_subnet,
            toString(answer_bgp_prefix) as answer_bgp_prefix,
            pop_lat,
            pop_lon,
            pop_city,
            pop_country,
            pop_continent
        FROM 
            {self.settings.DATABASE}.{table_name} 
        {hostname_filter}
        """


class GetPoPPerHostname(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"WHERE hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""

        return f"""
        SELECT 
            hostname,
            groupUniqArray((answer_subnet, pop_lat, pop_lon)) as pop
        FROM 
            {self.settings.DATABASE}.{table_name} 
        {hostname_filter}
        GROUP BY
            hostname
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
            raise RuntimeError(f"Named argument subnet_filter required for {__class__}")

        return f"""
        SELECT
            toString(subnet) as client_subnet,
            hostname,
            source_scope,
            groupUniqArray((answer)) as answers,
            groupUniqArray((answer_subnet)) as answer_subnets,
            groupUniqArray((answer_bgp_prefix)) as answer_bgp_prefixes
        FROM
            {self.settings.DATABASE}.{table_name}
        WHERE 
            {subnet_filter}
            {hostname_filter}
        GROUP BY
            (client_subnet, hostname, source_scope)
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
            {self.settings.DATABASE}.{table_name}
        WHERE 
            {subnet_filter}
            {hostname_filter}
        GROUP BY
            (client_subnet, hostname)
        """


class GetDNSMappingHostnamesNew(Query):
    def statement(self, table_name: str, **kwargs) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""
        if subnet_filter := kwargs.get("subnet_filter"):
            subnet_filter = "".join([f",toIPv4('{s}')" for s in subnet_filter])[1:]
            subnet_filter = f"client_subnet IN ({subnet_filter})"
        else:
            raise RuntimeError(f"Named argument subnet_filter required for {__class__}")

        try:
            answer_granularity = kwargs["answer_granularity"]
            client_granularity = kwargs["client_granularity"]
        except KeyError:
            raise RuntimeError(f"Column name parameter missing for {__class__}")

        return f"""
        SELECT
            toString({client_granularity}) as client_granularity,
            hostname,
            groupUniqArray({answer_granularity}) as mapping
        FROM
            {self.settings.DATABASE}.{table_name}
        WHERE 
            {subnet_filter}
            {hostname_filter}
        GROUP BY
            (client_granularity, hostname)
        """


class GetMeasurementIds(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT
            distinct(
                msm_id
            ) as msm_id
        FROM 
            {self.settings.DATABASE}.{table_name} 
        """


class GetDNSMappingOld(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT
            distinct(
                toString(client_subnet) as client_subnet, 
                hostname as hostname, 
                toString(answers) as answer
            ) as data
        FROM 
            {self.settings.DATABASE}.{table_name} 
        """


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
