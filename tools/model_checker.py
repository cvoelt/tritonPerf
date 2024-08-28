import os
import pprint
import shutil

if __name__ == '__main__':

  create_config_model_list = False
  show_models_with_errors = False
  check_if_modelfolder_empty = True
  copy_trt_models = False
  print_model_list_from_config = False
  model_dir = '/tritonPerformance/models/'
  zoo_dir = '/tritonPerformance/final_fp16/'

  if create_config_model_list:
    iterator = 1
    for name in sorted(os.listdir(model_dir)):
      print(str(iterator) + ' = ' + name)
      iterator += 1

  if show_models_with_errors:
    with open('logfile.txt') as logfile:
      print_next = False
      for line in logfile:
        if 'error' in line:
          for word in line.split():
            if print_next:
              print(word.replace('\'', ''))
              print_next = False
            if '-m\'' in word:
              print_next = True

  if check_if_modelfolder_empty:
    models_exist = []
    models_dont_exist = []
    onnx_models = []
    for model in sorted(os.listdir(model_dir)):
      if os.path.exists(model_dir + model + '/1/model.plan'):
        models_exist.append(model.replace('_i1',''))
      elif os.path.exists(model_dir + model + '/1/model.onnx'):
        onnx_models.append(model.replace('_i1',''))
      else:
        models_dont_exist.append(model.replace('_i1',''))

    print('working models')
    pprint.pprint(models_exist)
    print('onnx models')
    pprint.pprint(onnx_models)
    print('non-working models')
    pprint.pprint(models_dont_exist)

  if copy_trt_models:
    models_copied = []
    models_dont_exist = []
    onnx_models = []

    for model in sorted(os.listdir(model_dir)):
      if os.path.exists(model_dir + model + '/1/model.plan'):
        if 'fp16' in model_dir + model:
          shutil.copytree(model_dir + model + '/', zoo_dir + model.replace('_i1','') + '/')
          models_copied.append(model.replace('_i1',''))
      elif os.path.exists(model_dir + model + '/1/model.onnx'):
        onnx_models.append(model.replace('_i1',''))
      else:
        models_dont_exist.append(model.replace('_i1',''))

    print('copied models')
    pprint.pprint(models_copied)
    print('onnx models')
    pprint.pprint(onnx_models)
    print('non-working models')
    pprint.pprint(models_dont_exist)

  if print_model_list_from_config:
    with open ('models.txt') as models:
      for line in models:
        if not line == '\n':
          line = line.replace('\n','')
          before, after = line.split('=')
          print(after.replace(' ','') + ',')

  
