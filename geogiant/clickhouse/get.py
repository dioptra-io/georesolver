from geogiant.clickhouse import Clickhouse


class Get(Clickhouse):
    def subnet_statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            DISTINCT toString(subnet_v4)
        FROM 
            {self.settings.CLICKHOUSE_DB}.{table_name}
        """

    def subnet(self, table_name: str) -> dict:
        """get all vps per pop with RTT info"""
        statement = self.subnet_statement(table_name)
        resp = self.execute_iter(statement)

        vps_subnet = [row[0] for row in resp]
        return vps_subnet

    def vps_country_per_subnet_statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            subnet_v4,
            groupUniqueArray((toString(address_v4), lat, lon, continent))
        FROM 
            {self.settings.CLICKHOUSE_DB}.{table_name}
        GROUP BY
            subnet_v4
        """

    def vps_country_per_subnet(self, table_name: str) -> dict:
        """get all vps per pop with RTT info"""
        vps_per_subnet = {}

        statement = self.vps_country_per_subnet_statement(table_name)
        resp = self.execute_iter(statement)

        for row in resp:
            vps_per_subnet[row[0]] = {
                "address_v4": row[1],
                "lat": row[2],
                "lon": row[3],
                "continent": row[4],
            }

        return vps_per_subnet

    def subnet_per_hostname_statement(self, table_name: str) -> str:
        return f"""
        SELECT 
            hostname,
            answer_bgp_prefix,
            groupUniqueArray(subnet)
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
            subnet_per_hostname[row[0]][row[1]] = row[2]

        return subnet_per_hostname
