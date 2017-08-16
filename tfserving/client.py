"""
A simple client to query a TensorFlow Serving instance.

Example:
$ python client.py \
--images IMG_0932_sm.jpg \
--num_results 10 \
--model_name inception \
--host localhost \
--port 9000 \
--timeout 10

Author: Grant Van Horn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import tfserver

def parse_args():

  parser = argparse.ArgumentParser(description='Command line classification client. Sorts and prints the classification results.')

  parser.add_argument('--images', dest='image_paths',
                        help='Path to one or more images to classify (jpeg or png).',
                        type=str, nargs='+', required=True)

  parser.add_argument('--num_results', dest='num_results',
                      help='The number of results to print. Set to 0 to print all classes.',
                      required=False, type=int, default=0)

  parser.add_argument('--model_name', dest='model_name',
                        help='The name of the model to query.',
                        required=False, type=str, default='inception')

  parser.add_argument('--host', dest='host',
                        help='Machine host where the TensorFlow Serving model is.',
                        required=False, type=str, default='localhost')

  parser.add_argument('--port', dest='port',
                      help='Port that the TensorFlow Server is listening on.',
                      required=False, type=int, default=9000)

  parser.add_argument('--timeout', dest='timeout',
                      help='Amount of time to wait before failing.',
                      required=False, type=int, default=10)

  args = parser.parse_args()

  return args

def main():

  args = parse_args()

  # Read in the image bytes
  image_data = []
  for fp in args.image_paths:
    with open(fp) as f:
      data = f.read()
    image_data.append(data)

  # Get the predictions
  t = time.time()
  predictions = tfserver.predict(image_data, model_name=args.model_name,
    host=args.host, port=args.port, timeout=args.timeout
  )
  dt = time.time() - t
  print("Prediction call took %0.4f seconds" % (dt,))

  # Process the results
  results = tfserver.process_classification_prediction(predictions, max_classes=args.num_results)

  # Print the results
  for i, fp in enumerate(args.image_paths):
    print("Results for image: %s" % (fp,))
    for name, score in results[i]:
      print("%s: %0.3f" % (name, score))
    print()

if __name__ == '__main__':
  main()