from geogiant.clickhouse.query import Query, NativeQuery


class GetVPs(Query):
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
        WHERE
            is_anchor == false
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
            DISTINCT toString(subnet_v4) as subnet
        FROM 
            {self.settings.DATABASE}.{table_name}
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
    def statement(
        self,
        table_name: str,
    ) -> str:
        return f"""
        SELECT 
            toString(dst_addr) as target,
            groupArray((toString(src_addr), min)) as pings
        FROM 
            {self.settings.DATABASE}.{table_name}
        WHERE
            min > -1 
            AND target != toString(src_addr)
        GROUP BY 
            target
        """


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


class GetHostnames(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            DISTINCT hostname as hostname
        FROM 
            {self.settings.DATABASE}.{table_name}
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
            groupUniqArray(toString(answer)) as answer
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
        try:
            answer_granularity = kwargs["answer_granularity"]
            client_granularity = kwargs["client_granularity"]
        except KeyError:
            raise RuntimeError(f"Column name parameter missing for {__class__}")

        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""

        if subnet_filter := kwargs.get("subnet_filter"):
            subnet_filter = "".join([f",toIPv4('{s}')" for s in subnet_filter])[1:]
            subnet_filter = f"{client_granularity} IN ({subnet_filter})"
        else:
            raise RuntimeError(f"Named argument subnet_filter required for {__class__}")

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
