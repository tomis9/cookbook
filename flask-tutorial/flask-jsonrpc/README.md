### a minimal example of flask-JSONRPC

app.py:


[That](https://github.com/cenobites/flask-jsonrpc) is a very short intro to flas-JSONRPC.

Check if it works:

curl -i -X POST \
   -H "Content-Type: application/json; indent=4" \
   -d '{
    "jsonrpc": "2.0",
    "method": "App.index",
    "params": {},
    "id": "1"
}' http://localhost:5000/api


minimal.py:

That is a slightly more complicated case ([link](https://github.com/cenobites/flask-jsonrpc/tree/master/examples/minimal)).

In the link above you will find further instructions.
