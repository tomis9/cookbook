import json
import requests
from elasticsearch import Elasticsearch


# check if connection works
host = "192.168.33.10"
res = requests.get("http://" + host + ":9200")
print(res.content)

# create an ES instance
es = Elasticsearch([{'host': host, 'port': 9200}])


r = requests.get("http://" + host + ":9200")
i = 1
while r.status_code == 200:
    r = requests.get('http://swapi.co/api/people/' + str(i))
    content = r.content.decode("utf-8")
    es.index(index='sw', doc_type='people', id=i, body=json.loads(content))
    i = i + 1
    print(i)


es.get(index='sw', doc_type='people', id=5)
es.search(index="sw", body={"query": {"match": {'name': 'Darth Vader'}}})
es.search(index="sw", body={"query": {"prefix": {"name": "lu"}}})


e1 = {
    "first_name": "nitin",
    "last_name": "panwar",
    "age": 27,
    "about": "Love to play cricket",
    "interests": ['sports', 'music'],
}


res = es.index(index='megacorp', doc_type='employee', id=1, body=e1)

# Let's insert some more documents
e2 = {
    "first_name":  "Jane",
    "last_name":   "Smith",
    "age":         32,
    "about":       "I like to collect rock albums",
    "interests":   ["music"]
}
e3 = {
    "first_name":  "Douglas",
    "last_name":   "Fir",
    "age":         35,
    "about":        "I like to build cabinets",
    "interests":   ["forestry"]
}


res = es.index(index='megacorp', doc_type='employee', id=2, body=e2)
print(res['created'])
res = es.index(index='megacorp', doc_type='employee', id=3, body=e3)
print(res['created'])
