from geogiant.clickhouse import Insert


class VPsTable(Insert):
    def create_table_statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = (
            "address_v4, asn_v4, bgp_prefix, country_code, lat, lon, is_anchor"
        )
        return f"""
            CREATE TABLE IF NOT EXISTS {self.settings.CLICKHOUSE_DB}.{table_name}
            (
                address_v4         IPv4,
                subnet_v4          IPv4,
                asn_v4             Int32,
                bgp_prefix         String,
                country_code       String,
                lat                Float32,
                lon                Float32,
                id                 Int32,
                is_anchor          Bool,                
            )
            ENGINE MergeTree
            ORDER BY ({sorting_key})
            """


class PingTable(Insert):
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


class DNSMappingTable(Insert):
    def create_table_statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = "subnet, netmask, hostname, timestamp"
        return f"""
        CREATE TABLE IF NOT EXISTS {self.settings.CLICKHOUSE_DB}.{table_name}
        (
            timestamp              DateTime(),
            subnet                 IPv4,
            netmask                UInt8,
            hostname               String,
            answer                 IPv4,
            answer_asn             UInt32,
            answer_bgp_prefix      String,
        )
        ENGINE MergeTree
        ORDER BY ({sorting_key})
        """
