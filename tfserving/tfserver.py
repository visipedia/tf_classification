"""
TensorFlow Serving caller code.

Requirements:
pip install numpy tensorflow tensorflow-serving-api

Author: Grant Van Horn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

def predict(image_data,
            model_name='inception',
            host='localhost',
            port=9000,
            timeout=10):
  """
  Arguments:
    image_data (list): A list of image data. The image data should either be the image bytes or
      float arrays.
    model_name (str): The name of the model to query (specified when you started the Server)
    model_signature_name (str): The name of the signature to query (specified when you created the exported model)
    host (str): The machine host identifier that the classifier is running on.
    port (int): The port that the classifier is listening on.
    timeout (int): Time in seconds before timing out.

  Returns:
    PredictResponse protocol buffer. See here: https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto
  """

  if len(image_data) <= 0:
    return None

  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name

  if type(image_data[0]) == str:
    request.model_spec.signature_name = 'predict_image_bytes'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data)]))
  else:
    request.model_spec.signature_name = 'predict_image_array'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, shape=[len(image_data), len(image_data[1])]))

  result = stub.Predict(request, timeout)
  return result

def process_classification_prediction(predictions, max_classes=10):
  """
  Arguments:
    prediction (PredictResponse protocol buffer): TensorFlow Serving prediction response.
    num_classes (int): Maximum number of results to return. Set to 0 for all results.
  Returns:
    list of lists: A list of (name, score) tuples, one for each prediction.
  """

  # Determine how many outputs there are
  dims = predictions.outputs['classes'].tensor_shape.dim
  num_inputs = dims[0].size
  num_classes = dims[1].size

  all_class_names = np.array(predictions.outputs['classes'].string_val).reshape(num_inputs, num_classes)
  all_scores = np.array(predictions.outputs['scores'].float_val).reshape(num_inputs, num_classes)

  results = []
  for i in range(num_inputs):

    scores = all_scores[i]
    class_names = all_class_names[i]

    idxs = np.argsort(scores)[::-1]
    scores = scores[idxs]
    class_names = class_names[idxs]

    num_to_return = min(num_classes, max_classes)
    if num_to_return <= 0:
      num_to_return = scores.shape[-1]

    names_scores = [(class_names[i], scores[i]) for i in range(num_to_return)]
    results.append(names_scores)

  return results