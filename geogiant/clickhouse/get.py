from geogiant.clickhouse.query import Query


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
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            hostname,
            answer_bgp_prefix,
            groupUniqArray(subnet) as subnets
        FROM 
            {self.settings.DATABASE}.{table_name}
        GROUP BY
            (hostname, answer_bgp_prefix)
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


class GetDNSMapping(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            distinct(
                toString(client_subnet), 
                hostname, 
                toString(answers)
            ) as data
        FROM 
            {self.settings.DATABASE}.{table_name} 
        """
