from collections import defaultdict

from geogiant.clickhouse.query import Query


class GetSubnets(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            DISTINCT toString(subnet_v4) as subnet
        FROM 
            {self.settings.DB}.{table_name}
        """

    # def subnet(self, table_name: str) -> dict:
    #     """get all vps per pop with RTT info"""
    #     statement = self.subnet_statement(table_name)
    #     resp = self.execute_iter(statement)

    #     vps_subnet = [row[0] for row in resp]
    #     return vps_subnet


class Get(Query):
    def vps_country_per_subnet_statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            subnet_v4,
            groupUniqArray(
                (
                    toString(address_v4),
                    lat, 
                    lon,
                    country_code
                )
            )
        FROM 
            {self.settings.CLICKHOUSE_DB}.{table_name}
        GROUP BY
            subnet_v4
        """

    def vps_country_per_subnet(self, table_name: str) -> dict:
        """get all vps per pop with RTT info"""
        vps_per_subnet = defaultdict(list)

        statement = self.vps_country_per_subnet_statement(table_name)
        resp = self.execute_iter(statement)

        for row in resp:
            subnet = row[0]
            subnet_data = row[1][0]

            vps_per_subnet[subnet].append(
                {
                    "address_v4": subnet_data[0],
                    "lat": subnet_data[1],
                    "lon": subnet_data[2],
                    "country_code": subnet_data[3],
                }
            )

        return vps_per_subnet

    def subnet_per_hostname_statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            hostname,
            answer_bgp_prefix,
            groupUniqArray(subnet)
        FROM 
            {self.settings.CLICKHOUSE_DB}.{table_name}
        GROUP BY
            (hostname, answer_bgp_prefix)
        """

    def subnet_per_hostname(self, table_name: str) -> dict[dict]:
        """get all vps for each given frontend server"""
        subnet_per_hostname = {}

        statement = self.subnet_per_hostname_statement(table_name)
        resp = self.execute_iter(statement)

        for row in resp:
            subnet_per_hostname[(row[0], row[1])] = row[2]

        return subnet_per_hostname
