# list all indexes
curl -X GET 127.0.0.1:9200/_cat/indices?v 

# create and index
curl -XPUT "http://localhost:9200/bandittest"

# get data from index
curl -XGET 'http://localhost:9200/bandittest/_search?pretty=true&q=*:*'

# send data to index
curl -H "Content-Type: application/json" -XPOST "http://localhost:9200/bandit-test" -d "test value"

curl -H "Content-Type: application/json" -XPOST "http://localhost:9200/bandittest/elo" -d "{ \"field\" : \"value\"}"

http://localhost:9200/bandit-test/?_create
POST
{
"name": "Andrew",
"age" : 45,
"experienceInYears" : 10
}
