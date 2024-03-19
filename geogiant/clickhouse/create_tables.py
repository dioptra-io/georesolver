from geogiant.clickhouse.query import Query


class CreateVPsTable(Query):
    def statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = (
            "address_v4, asn_v4, bgp_prefix, country_code, lat, lon, is_anchor"
        )
        return f"""
            CREATE TABLE IF NOT EXISTS {self.settings.DATABASE}.{table_name}
            (
                address_v4         IPv4,
                subnet_v4          IPv4,
                asn_v4             Int32,
                bgp_prefix         String,
                country_code       String,
                lat                Float32,
                lon                Float32,
                id                 Int32,
                is_anchor          Bool                
            )
            ENGINE MergeTree
            ORDER BY ({sorting_key})
            """


class CreatePingTable(Query):
    def statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = "src_addr, src_netmask, prb_id, msm_id, dst_addr, proto, rcvd, sent, min, max, avg, rtts"
        return f"""
            CREATE TABLE IF NOT EXISTS {self.settings.DATABASE}.{table_name}
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


class CreateTracerouteTable(Query):
    def statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = "src_addr, src_netmask, prb_id, msm_id, dst_addr, ttl, reply_addr, proto, rcvd, sent, min, max, avg, rtts"
        return f"""
            CREATE TABLE IF NOT EXISTS {self.settings.DATABASE}.{table_name}
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
                reply_addr         IPv4,
                reply_prefix       IPv4,
                ttl                UInt32,
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


class CreateGeolocTable(Query):
    def statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = "addr, subnet, bgp_prefix, asn, vp_addr"
        return f"""
            CREATE TABLE IF NOT EXISTS {self.settings.DATABASE}.{table_name}
            (
                addr               IPv4,
                subnet             IPv4,
                bgp_prefix         String,
                asn                Int16,
                lat                Float32, 
                lon                Float32,
                country_code       String,
                vp_addr            IPv4,
                vp_subnet          IPv4,
                vp_bgp_prefix      String,
                min_rtt            Float32,
                measured_dst       Float32
            )
            ENGINE MergeTree
            ORDER BY ({sorting_key})
            """


class CreateDNSMappingTable(Query):
    def statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        sorting_key = "subnet, netmask, hostname, timestamp"
        return f"""
        CREATE TABLE IF NOT EXISTS {self.settings.DATABASE}.{table_name}
        (
            timestamp              DateTime(),
            subnet                 IPv4,
            netmask                UInt8,
            hostname               String,
            answer                 IPv4,
            answer_subnet          IPv4,
            answer_bgp_prefix      String,
            answer_asn             UInt32,
            source_scope           UInt8
        )
        ENGINE MergeTree
        ORDER BY ({sorting_key})
        """


class CreateDNSMappingWithMetadataTable(Query):
    def statement(self, table_name: str) -> str:
        sorting_key = "client_subnet,client_bgp_prefix, client_asn, hostname, answer, answer_subnet, answer_bgp_prefix, answer_asn, pop_ip_info_id"
        return f"""
            CREATE TABLE IF NOT EXISTS {self.settings.DATABASE}.{table_name}
            (
                client_subnet          IPv4,
                client_bgp_prefix      String,
                client_asn             UInt32,
                hostname               String,
                answer                 IPv4,
                answer_subnet          IPv4,
                answer_bgp_prefix      String,
                answer_asn             UInt32,
                pop_ip_info_id         Int32,
                pop_lat                Float32,
                pop_lon                Float32,
                pop_city               String,
                pop_country            String,
                pop_continent          String
            )
            ENGINE MergeTree
            ORDER BY ({sorting_key})
            """
