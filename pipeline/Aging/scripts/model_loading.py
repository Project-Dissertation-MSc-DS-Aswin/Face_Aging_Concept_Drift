from context import Constants, Args
from experiment.model_loader import FaceNetKerasModelLoader, FaceRecognitionBaselineKerasModelLoader
import whylogs
import pickle
import logging
import os

"""
Initializing constants from YAML file
"""
constants = Constants()
args = Args({})

args.model = os.environ.get('model', 'FaceNetKeras')
args.model_path = os.environ.get('model_path', 'facenet_keras.h5')
args.batch_size = os.environ.get('batch_size', 128)
args.input_shape = os.environ.get('input_shape', (-1,160,160,3))

parameters = list(
    map(lambda s: re.sub('$', '"', s),
        map(
            lambda s: s.replace('=', '="'),
            filter(
                lambda s: s.find('=') > -1 and bool(re.match(r'[A-Za-z0-9_]*=[.\/A-Za-z0-9]*', s)),
                sys.argv
            )
    )))

for parameter in parameters:
    logging.warning('Parameter: ' + parameter)
    exec("args." + parameter)

args.batch_size = int(args.batch_size)
if type(args.input_shape) == str:
    input_shape = args.input_shape.replace('(','').replace(')','').split(",")
    args.input_shape = tuple([int(s) for s in input_shape if s.strip() != '' or s.strip() != ','])
    print(args.input_shape)

if args.model == 'FaceNetKeras':
  model_loader = FaceNetKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
elif args.model == 'FaceRecognitionBaselineKeras':
  model_loader = FaceRecognitionBaselineKerasModelLoader(whylogs, args.model_path, input_shape=args.input_shape)
model_loader.load_model()

pickle.dump(model_loader, open("model_loader.pkl", "wb"))
