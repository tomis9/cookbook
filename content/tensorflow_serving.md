---
title: "tensorflow_serving"
date: 2019-05-31T15:29:17+02:00
draft: false
categories: ["scratchpad"]
tags: []
---

```
#!/usr/bin/python3
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.utils import build_tensor_info

placeholder_name = 'a'
operation_name = 'add'

a = tf.placeholder(tf.int32, name=placeholder_name)
b = tf.constant(10)

# This is our model
add = tf.add(a, b, name=operation_name)

with tf.Session() as sess:

    # super complicated model
    ten_plus_two = sess.run(add, feed_dict={a: 2})
    print('10 + 2 = {}'.format(ten_plus_two))

    # od tego momentu robimy wszystko, zeby zapisac model
    # inputy i outputy chcemy przetworzyc do zapisywalnego formatu
    # najpierw robimy z nich tensory
    a_tensor = sess.graph.get_tensor_by_name(placeholder_name + ':0')
    sum_tensor = sess.graph.get_tensor_by_name(operation_name + ':0')

    print("a_tensor:", a_tensor)
    print("sum_tensor:", sum_tensor)

    # potem budujemy te tensory
    model_input = build_tensor_info(a_tensor)
    model_output = build_tensor_info(sum_tensor)

    print("model_input:", model_input)
    print("model_output:", model_output)

    # inputy i outputy zapisujemy razem do signature_definition
    # mowimy tez, co ma sie wykonac przy zapytanie modelu
    # (tensorflowserving/predict)
    signature_definition = signature_def_utils.build_signature_def(
        inputs={placeholder_name: model_input},
        outputs={operation_name: model_output},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    print("signature_definition:", signature_definition)

    builder = saved_model_builder.SavedModelBuilder('./models/simple_model/1')

    print("tag_constants.SERVING:", tag_constants.SERVING)
    print("signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:",
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition
        })

    # Save the model so we can serve it with a model server :)
    builder.save()
```

```
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=simple_model --model_base_path=$HOME/tf_test/models/simple_model
```

https://www.tensorflow.org/tfx/serving/api_rest

```
curl -X POST localhost:8501/v1/models/simple_model:predict -d '{"inputs":{"a": 10}}'
```

```
https://towardsdatascience.com/using-tensorflow-serving-grpc-38a722451064
```

```
import grpc
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc

hostport = "localhost:8500"
channel = grpc.insecure_channel(hostport)
# grpc.ChannelConnectivity(channel)

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'simple_model'
# request.model_spec.signature_name = 'serving_default'

tensor = tf.contrib.util.make_tensor_proto(10, dtype="int32")
request.inputs['a'].CopyFrom(tensor)

pred = stub.Predict(request)
print(pred)
print(pred.outputs)
```
