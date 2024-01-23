"""all count classes for extracting metrics"""
from geogiant.clickhouse import Query

from geogiant.common.settings import ClickhouseSettings, PathSettings

clickhouse_settings = ClickhouseSettings()
path_settings = PathSettings()


class OverallScore(Query):
    def statement(
        self,
        table_name: str,
        **kwargs,
    ) -> str:
        if hostname_filter := kwargs["hostname_filter"]:
            hostnames_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
        else:
            hostname_filter = ""
        if target_filter := kwargs["target_filter"]:
            target_filter = "".join([f",toIPv4('{t}')" for t in target_filter])[1:]
        else:
            target_filter = ""
        if column_name := kwargs["column_name"]:
            column_name = column_name
        else:
            raise RuntimeError(f"Column name is necessary for {__class__}")

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
                FROM {self.settings.DATABASE}.{table_name}
                GROUP BY client_subnet
            ) AS t1
            CROSS JOIN
            (
                SELECT
                    client_subnet,
                    groupUniqArray({column_name}) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE client_subnet NOT IN ({target_filter})
                GROUP BY client_subnet
            ) AS t2
            WHERE t1.client_subnet != t2.client_subnet
        )
        WHERE subnet_1 IN ({target_filter})
        GROUP BY subnet_1
        """


class OverallPoPSubnetScore(Query):
    def statement(
        self,
        table_name: str,
        target_subnets: list,
        hostname_filter: list[str] = None,
    ) -> str:
        hostnames_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
        target_filter = "".join([f",toIPv4('{t}')" for t in target_subnets])[1:]

        return f"""
        WITH groupArray((toString(subnet_2), score)) AS subnet_scores
        SELECT
            toString(subnet_1),
            arraySort(x -> (-(x.2)), subnet_scores)
        FROM
        (
            SELECT
                t1.client_subnet AS subnet_1,
                t2.client_subnet AS subnet_2,
                length(arrayIntersect(t1.mapping, t2.mapping)) / least(length(t1.mapping), length(t2.mapping)) AS score
            FROM
            (
                WITH
                    IPv4StringToNum('255.255.255.0') AS netmask,
                    IPv4NumToString(bitAnd(answers, netmask)) AS subnet_answers
                SELECT
                    client_subnet,
                    groupArray(subnet_answers) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE hostname NOT IN ({hostnames_filter})
                GROUP BY client_subnet
            ) AS t1
            CROSS JOIN
            (
                WITH
                    IPv4StringToNum('255.255.255.0') AS netmask,
                    IPv4NumToString(bitAnd(answers, netmask)) AS subnet_answers
                SELECT
                    client_subnet,
                    groupArray(subnet_answers) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE hostname NOT IN ({hostnames_filter}) AND client_subnet NOT IN ({target_filter})
                GROUP BY client_subnet
            ) AS t2
            WHERE t1.client_subnet != t2.client_subnet
        )
        WHERE subnet_1 IN ({target_filter})
        GROUP BY subnet_1
        """


class HostnamePoPFrontendScore(Query):
    def statement(
        self,
        table_name: str,
        target_subnets: list,
        hostname_filter: list[str] = None,
        subset: list = None,
    ) -> str:
        hostnames_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
        target_filter = "".join([f",toIPv4('{t}')" for t in target_subnets])[1:]
        filter_on_subset = "".join([f",toIPv4('{t}')" for t in subset])[1:]

        return f"""
        WITH groupArray((toString(subnet_2), score)) AS subnet_scores
        SELECT
            toString(subnet_1),
            arraySort(x -> -(x.2), subnet_scores)
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
                    hostname,
                    groupUniqArray(answers) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE hostname NOT IN ({hostnames_filter})  AND client_subnet IN ({filter_on_subset})
                GROUP BY (client_subnet, hostname)
            ) AS t1
            CROSS JOIN
            (
                SELECT
                    client_subnet,
                    hostname,
                    groupUniqArray(answers) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE hostname NOT IN ({hostnames_filter}) AND client_subnet NOT IN ({target_filter})
                GROUP BY (client_subnet, hostname)
            ) AS t2
            WHERE t1.client_subnet != t2.client_subnet
        )
        GROUP BY subnet_1
        """


class HostnamePoPSubnetScore(Query):
    def statement(
        self,
        table_name: str,
        target_subnets: list,
        hostname_filter: list[str] = None,
        subset: list = None,
    ) -> str:
        hostnames_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
        target_filter = "".join([f",toIPv4('{t}')" for t in target_subnets])[1:]
        filter_on_subset = "".join([f",toIPv4('{t}')" for t in subset])[1:]

        return f"""
        WITH groupArray((toString(subnet_2), score)) AS subnet_scores
        SELECT
            toString(subnet_1),
            arraySort(x -> -(x.2), subnet_scores)
        FROM
        (
            SELECT
                t1.client_subnet AS subnet_1,
                t2.client_subnet AS subnet_2,
                length(arrayIntersect(t1.mapping, t2.mapping)) / least(length(t1.mapping), length(t2.mapping)) AS score
            FROM
            (
                WITH
                    IPv4StringToNum('255.255.255.0') AS netmask,
                    IPv4NumToString(bitAnd(answers, netmask)) AS subnet_answers
                SELECT
                    client_subnet,
                    groupArray(subnet_answers) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE hostname NOT IN ({hostnames_filter})  AND client_subnet IN ({filter_on_subset})
                GROUP BY (client_subnet, hostname)
            ) AS t1
            CROSS JOIN
            (
                WITH
                    IPv4StringToNum('255.255.255.0') AS netmask,
                    IPv4NumToString(bitAnd(answers, netmask)) AS subnet_answers
                SELECT
                    client_subnet,
                    groupArray(subnet_answers) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE hostname NOT IN ({hostnames_filter}) AND client_subnet NOT IN ({target_filter})
                GROUP BY (client_subnet, hostname)
            ) AS t2
            WHERE t1.client_subnet != t2.client_subnet
        )
        GROUP BY subnet_1
        """
