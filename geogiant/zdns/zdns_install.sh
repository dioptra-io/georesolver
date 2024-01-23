git clone https://github.com/zmap/zdns.git
cd zdns
go build

cp zdns/zdns geogiant/zdns/zdns_binary
rm -rf zdns