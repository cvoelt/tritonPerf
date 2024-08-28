import os
import typing
import shutil
import torch
import onnxruntime
import logging
import pathlib

dynamic_input_dims: typing.Dict[int, str] = {0: 'batch_size', 2: 'width', 3: 'height'}
dynamic_ouput_dims: typing.Dict[int, str] = {0: 'batch_size'}
model_dir = '/home/constantin/Daten/timmmodels/'


def get_model_path(model_name: str, create: bool = False) -> pathlib.Path:
  model_root_folder = pathlib.Path(model_dir + model_name)
  if not create and not model_root_folder.exists():
    return None
  model_root_folder.mkdir(exist_ok=True)
  return model_root_folder

def create_file(path: pathlib.Path, file_name: str = None) :
  if file_name is None:
    return open(path.as_posix(), 'w')
  else:
    return open(path.as_posix() + '/' + file_name, 'w')


def create_model_config(model: str, platform_name: str, input_shape: typing.List[int],
                        output_shape: typing.List[int], cpu_on: bool,
                        precision: str = 'FP32') -> None:
  instances = 1
  model_name = model
  name = 'name: "' + model_name + '"\n'
  platform = 'platform: "' + platform_name + '"\n'
  batch_size = 'max_batch_size: 32' '\n'
  input = 'input [\n\t{\n\t\tname: "input"\n'
  data_type = '\t\tdata_type: TYPE_' + precision + '\n'
  input_dims = '\t\tdims: ' + str(input_shape) + '\n'
  output = 'output [\n\t{\n\t\tname: "output"\n'
  output_dims = '\t\tdims: ' + str(output_shape) + '\n'
  close_brackets = '\t}\n]\n'

  instance_kind = 'KIND_GPU\n\t\tgpus: [ 0 ]'
  if cpu_on:
    instance_kind = 'KIND_CPU'

  instance_gr = 'instance_group [\n\t{\n'
  instance_gr += '\t\tcount: ' + str(instances) + '\n'
  instance_gr += '\t\tkind: ' + instance_kind + '\n'
  instance_gr += '\t}\n'
  instance_gr += ']'

  header = name + platform + batch_size
  input = input + data_type + input_dims + close_brackets
  output = output + data_type + output_dims + close_brackets
  result = header + input + output + instance_gr

  model_path = get_model_path(model, create=False)
  with create_file(model_path, 'config.pbtxt') as config_file:
    config_file.write(result)

def get_input_output_shapes_from_onnx(onnx_file: str) -> typing.Tuple[typing.List[int], typing.List[int]]:
  onnx_file = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  input_shape = onnx_file.get_inputs()[0].shape  # e.g.[1, 3, 224, 224]
  output_shape = onnx_file.get_outputs()[0].shape  # e.g. [1, 1000]
  for dim in dynamic_input_dims:  # e.g.[-1, 3, -1, -1]
    input_shape[dim] = -1
  if len(output_shape) <= 2:
    for dim in dynamic_ouput_dims:
      output_shape[dim] = -1
  else:
    for dim in range(0, len(output_shape)):  # dynamic_ouput_dims:# e.g. [-1, 1000]
      output_shape[dim] = -1
  input_shape = input_shape[1:]  # e.g.[3, -1, -1]
  output_shape = output_shape[1:]  # e.g. [1000]
  return input_shape, output_shape

def create_model_folder(model_name: str,
                        file_type: str):
  root = get_model_path(model_name, create=True)
  subdirs = os.listdir(root)
  if not subdirs:  # no subdirectories available
    new_folder = root.as_posix() + '/1/'
    os.makedirs(new_folder, exist_ok=True)
    return pathlib.Path(new_folder)

def create_onnx_model(model_name: str) -> bool:

  logging.info('Downloading model data: ' + model_name + ' please wait')
  model_instance = timm.create_model(model_name)
  if model_instance is None:
    logging.error('Could not find the model ' + model_name + ' in the model zoo. Cant create ONNX-File')
    return False
  logging.info('Finished download of model data : ' + model_name + ' please wait')
  onnx_folder = create_model_folder(model_name, 'onnx')
  input_tensor = torch.randn(1, 3, 224, 224)

  # create ONNX-Model with fixed shapes to read the input- and output-dimensions
  logging.debug('Creating onnx-model with fixed axes: ' + model_name + ' please wait')

  try:
    torch.onnx.export(model_instance,
                      input_tensor,
                      onnx_folder.as_posix() + '/fixed_axes_delete_me.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=13)
  except:
    logging.error('failed to create onnx-file for model:' + model_name)
    return False

  input_shape, output_shape = get_input_output_shapes_from_onnx(onnx_folder.as_posix() + '/fixed_axes_delete_me.onnx')
  os.remove(onnx_folder.as_posix() + '/fixed_axes_delete_me.onnx')

  logging.info('Creating onnx-model: ' + model_name + ' please wait')

  try:
    torch.onnx.export(model_instance,
                      input_tensor,
                      onnx_folder.as_posix() + '/model.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': dynamic_input_dims,
                                    'output': dynamic_ouput_dims},
                      opset_version=13)
  except:
    logging.error('failed to create onnx-file for model:' + model_name)
    return False

  create_model_config(model_name, 'onnxruntime_onnx',input_shape, output_shape, cpu_on=False)
  return True



if __name__ == '__main__':
  import timm
  from pprint import pprint
  available_models = timm.list_models()
  print('available models')
  pprint(available_models)
  not_working = []
  working = []

  for model in available_models:

    if not model == 'resnetv2_152x4_bitm_in21k' and not model == 'vit_gigantic_patch14_224':
      if os.path.exists(model_dir + model):
        working.append(model)
        continue
      elif create_onnx_model(model):
        working.append(model)
        print('success with model: ' + model)
      else:
        root = get_model_path(model, create=True)
        shutil.rmtree(root.as_posix())
        print('no success with model:' + model)
        not_working.append(model)

  print('working:')
  pprint(working)
  print('not working:')
  pprint(not_working)
  'machen Probleme: resnetv2_152x4_bitm_in21k , vit_gigantic_patch14_224 '''


