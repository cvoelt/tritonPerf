from . import utilities
from . import config
import onnxruntime
import pathlib
import logging
import os
import torch
import typing
import torchvision



dynamic_input_dims: typing.Dict[int, str] = {0:'batch_size', 2:'width', 3:'height'}
dynamic_ouput_dims: typing.Dict[int, str] = {0:'batch_size'}


def create_model_config(model: str, platform_name: str, instances: int, input_shape: typing.List[int], cpu_on: bool, main_config: config.MainConfig, precision: str= 'FP32') -> None:
    
    model_name = model + '_i' + str(instances) 
    name = 'name: "' + model_name + '"\n'
    platform = 'platform: "' + platform_name + '"\n'
    batch_size = 'max_batch_size: ' + '16' + '\n' #str(main_config.get_max_batchsize())
    input = 'input [\n\t{\n\t\tname: "input"\n'
    data_type = '\t\tdata_type: TYPE_' + precision +'\n'
    input_dims = '\t\tdims: ' + str(input_shape) +'\n'
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
    result = header + input + instance_gr

    model_path = utilities.get_model_path(model, instances, main_config, create= False)
    with utilities.create_file(model_path, 'config.pbtxt') as config_file:
        config_file.write(result)


def get_input_output_shapes_from_onnx(onnx_file: str) -> typing.Tuple[typing.List[int], typing.List[int]]:

    onnx_file = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    input_shape = onnx_file.get_inputs()[0].shape # e.g.[1, 3, 224, 224]
    output_shape = onnx_file.get_outputs()[0].shape# e.g. [1, 1000]
    for dim in dynamic_input_dims:# e.g.[-1, 3, -1, -1]
        input_shape[dim] = -1
    if len(output_shape) <= 2:
        for dim in dynamic_ouput_dims:
            output_shape[dim] = -1
    else:
        for dim in range(0,len(output_shape)): #dynamic_ouput_dims:# e.g. [-1, 1000]
            output_shape[dim] = -1
    input_shape = input_shape[1:]# e.g.[3, -1, -1]
    output_shape = output_shape[1:]# e.g. [1000]
    return input_shape, output_shape

def get_input_output_shapes_from_config_file(model_config_file: pathlib.Path) -> typing.Tuple[typing.List[int], typing.List[int]]:

    shapes = {'input': None, 'output': None}
    for shape in shapes:
        detected_category = False

        with open(model_config_file, "r") as file:
            for line in file:
                if shape + ' [' in line:
                    detected_category = True
                    pass
                if 'dims:' in line and detected_category:
                    dims, array = line.split(':')
                    array = array.replace('[', '').replace(']','').replace('\n','').split(',')
                    shapes[shape] = list(map(int, array))
                    detected_category = False
    
    return shapes.values()



def create_onnx_model(model_name: str, instances: int, main_config: config.MainConfig) -> bool:

    #check if model is already available in the zoo, copy it
    if pathlib.Path(main_config.get_zoo_dir().as_posix() + '/' + model_name + '/').exists():
        zoo_model_folder = utilities.copy_from_zoo(model_name, instances, main_config, 'onnx')
        input_shape, output_shape = get_input_output_shapes_from_config_file(zoo_model_folder.as_posix() + '/config.pbtxt')

        '''
        lower_instances_folder = utilities.model_with_lower_instances(model_name, instances, main_config)
        #if model already exists, copy from lower instances-model
        elif lower_instances_folder is not None:
            current_instances_folder = utilities.get_model_path(model_name, instances, main_config, create=True)
            utilities.copy_folder(lower_instances_folder, current_instances_folder)
            input_shape, output_shape = get_input_output_shapes_from_config_file(current_instances_folder.as_posix() + '/config.pbtxt')
        '''

    #otherwise, create new onnx-model
    else:
        logging.info('Downloading model data: ' + model_name + ' please wait')
        model_instance = model_zoo(model_name)
        if model_instance is None:
            logging.error('Could not find the model ' + model_name + ' in the model zoo. Cant create ONNX-File')
            return
        logging.info('Finished download of model data : ' + model_name + ' please wait')
        onnx_folder = utilities.create_model_folder(model_name, instances, main_config, 'onnx')
        input_tensor = torch.randn(1,3,256,256)

        # create ONNX-Model with fixed shapes to read the input- and output-dimensions
        logging.debug('Creating onnx-model with fixed axes: ' + model_name + ' please wait')
        
        try:
            torch.onnx.export(model_instance,
                                input_tensor,
                                onnx_folder.as_posix() + '/fixed_axes_delete_me.onnx',
                                input_names = ['input'],
                                output_names = ['output'],
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
                            input_names = ['input'],
                            output_names = ['output'],
                            dynamic_axes = {'input' : dynamic_input_dims,
                            'output' : dynamic_ouput_dims},
                            opset_version=13)
        except:
            logging.error('failed to create onnx-file for model:' + model_name)
            return False
    
    create_model_config(model_name, 'onnxruntime_onnx', instances, input_shape, cpu_on=False, main_config= main_config)
    return True

def create_cpu_model(model_name: str, instances: int, main_config: config.MainConfig) -> bool:
    
    #check if model is already available in the zoo, copy it
    if pathlib.Path(main_config.get_zoo_dir().as_posix() + '/' + model_name + '/').exists():
        zoo_model_folder = utilities.copy_from_zoo(model_name, instances, main_config, 'onnx', '_cpu')
        input_shape, output_shape = get_input_output_shapes_from_config_file(zoo_model_folder.as_posix() + '/config.pbtxt')

    #TODO:create symlinks instead of copies of the GPU-ONNX-file
    else:
        gpu_onnx_folder = utilities.get_model_path(model_name, instances, main_config, create=False)
        if gpu_onnx_folder is None:
            create_onnx_model(model_name, instances, main_config)
        cpu_onnx_folder = utilities.get_model_path(model_name + '_cpu', instances, main_config, create=True)
        utilities.copy_folder(gpu_onnx_folder,cpu_onnx_folder)
        input_shape, output_shape = get_input_output_shapes_from_config_file(gpu_onnx_folder.as_posix() + '/config.pbtxt')
    
    create_model_config(model_name + '_cpu', 'onnxruntime_onnx', instances, input_shape, cpu_on=True, main_config= main_config)



def create_trt_model(model_name: str, instances: int, precision: str, main_config: config.MainConfig) -> bool:

    #check if model is already available in the zoo, then copy it
    if pathlib.Path(main_config.get_zoo_dir().as_posix() + '/' + model_name + '_trt_' + precision).exists():
        try:
            #try to copy the trt-file and create a model folder, this does not copy the config-file
            model_folder = utilities.copy_from_zoo(model_name + '_trt_' + precision, instances, main_config, filetype='plan')
        except:
            #create an empty model folder, in case the model folder in the zoo is empty
            trt_folder = utilities.create_model_folder(model_name + '_trt_' + precision, instances, main_config)
            return
        
        input_shape, output_shape = get_input_output_shapes_from_config_file(model_folder.as_posix() + '/config.pbtxt')
        

    #otherwise, create new trt-model
    else:
        onnx_file = utilities.get_onnx_file(model_name, instances, main_config)

        if onnx_file is None:
            logging.error('No corresponding ONNX-file, TRT-Optimization not possible for the model: ' + model_name )
            return False

        trt_folder = utilities.create_model_folder(model_name + '_trt_' + precision, instances, main_config, 'plan')
        min_shape = '1x3x256x256'
        opt_shape = '16x3x512x512'
        max_shape = '32x3x1024x1024'

        trt_exec = '/usr/src/tensorrt/bin/trtexec --verbose'
        onnx_file =  ' --onnx=' + onnx_file.as_posix()
        min_shapes = ' --minShapes=input:' + min_shape
        opt_shapes = ' --optShapes=input:' + opt_shape
        max_shapes = ' --maxShapes=input:' + max_shape
        if precision == 'fp32':
            precision_command = ''
        else:
            precision_command = ' --' + precision
        workspace = ' --workspace=' + str(main_config.get_trt_workspace_size())
        build_only = ' --buildOnly'
        save_engine = ' --saveEngine=' + trt_folder.as_posix() + '/model.plan'

        command = trt_exec + build_only + workspace + onnx_file + min_shapes + opt_shapes + max_shapes + precision_command + save_engine
        logging.info('Executing TRT-Optimization: ' + command)

        try:
            os.system(command)
        except:
            logging.error('TRT-Optimization for model ' + model_name + precision + ' failed!')

    model_folder = utilities.get_model_path(model_name, instances, main_config, create = False)
    input_shape, output_shape = get_input_output_shapes_from_config_file(model_folder.as_posix() + '/config.pbtxt')
    
    create_model_config(model_name + '_trt_' + precision, 'tensorrt_plan', instances, input_shape, cpu_on=False, main_config=main_config)


def check_download_models(main_config: config.MainConfig)-> None:

    #download models, create onnx and CPU-Models if necessary
    for model in main_config.get_models():
        for instances in main_config.get_instance_count():

            #create onnx and trt models
            if utilities.get_model_path(model, instances, main_config, create=False) is None:
                create_onnx_model(model, instances, main_config)
            
            if main_config.trt_is_on(): #create trt models
                for precision in main_config.get_chosen_trt_precisions():
                    if utilities.get_model_path(model + '_trt_' + precision, instances, main_config, create=False) is None:
                        create_trt_model(model, instances, precision, main_config)
            
            if main_config.cpu_is_on(): #create cpu models
                if utilities.get_model_path(model + '_cpu', instances, main_config, create= False) is None:
                    create_cpu_model(model, instances, main_config)
    return None


def model_zoo(model_name: str) -> typing.Type[torch.nn.Module]:

    model_dict = {
        'alexnet' : torchvision.models.alexnet(),
        'wide_resnet101_2' : torchvision.models.wide_resnet101_2(),
        'wide_resnet50_2' : torchvision.models.wide_resnet50_2(),
        'resnext101_32x8d' : torchvision.models.resnext101_32x8d(),
        'resnext50_32x4d' : torchvision.models.resnext50_32x4d(),
        'resnet152' : torchvision.models.resnet152(),
        'resnet101' : torchvision.models.resnet101(),
        'resnet50' : torchvision.models.resnet50(),
        'resnet34' : torchvision.models.resnet34(),
        'resnet18' : torchvision.models.resnet18(),
        'vgg19_bn' : torchvision.models.vgg19_bn(),
        'vgg19' : torchvision.models.vgg19(),
        'vgg16_bn' : torchvision.models.vgg16_bn(),
        'vgg16' : torchvision.models.vgg16(),
        'vgg13_bn' : torchvision.models.vgg13_bn(),
        'vgg13' : torchvision.models.vgg13(),
        'vgg11_bn' : torchvision.models.vgg11_bn(),
        'vgg11' : torchvision.models.vgg11(),
        'squeezenet1_1' : torchvision.models.squeezenet1_1(),
        'squeezenet1_0' : torchvision.models.squeezenet1_0(),
        'inception_v3' : torchvision.models.inception_v3(),
        'densenet201' : torchvision.models.densenet201(),
        'densenet169' : torchvision.models.densenet169(),
        'densenet161' : torchvision.models.densenet161(),
        'densenet121' : torchvision.models.densenet121(),
        'googlenet' : torchvision.models.googlenet(),
        'mobilenet_v2' : torchvision.models.mobilenet_v2(),
        'mobilenet_v3_small' : torchvision.models.mobilenet_v3_small(),
        'mobilenet_v3_large' : torchvision.models.mobilenet_v3_large(),
        'mnasnet1_3' : torchvision.models.mnasnet1_3(),
        'mnasnet1_0' : torchvision.models.mnasnet1_0(),
        'mnasnet0_75' : torchvision.models.mnasnet0_75(),
        'mnasnet0_5' : torchvision.models.mnasnet0_5(),
        'shufflenet_v2_x2_0' : torchvision.models.shufflenet_v2_x2_0(),
        'shufflenet_v2_x1_5' : torchvision.models.shufflenet_v2_x1_5(),
        'shufflenet_v2_x1_0' : torchvision.models.shufflenet_v2_x1_0(),
        'shufflenet_v2_x0_5' : torchvision.models.shufflenet_v2_x0_5(),
        'fcn_resnet50' : torchvision.models.segmentation.fcn_resnet50(),
        'lraspp_mobilenet_v3_large' : torchvision.models.segmentation.lraspp_mobilenet_v3_large(),
        'deeplabv3_mobilenet_v3_large' :torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(),
        'deeplabv3_resnet101' : torchvision.models.segmentation.deeplabv3_resnet101(),
        'deeplabv3_resnet50' : torchvision.models.segmentation.deeplabv3_resnet50(),
        'fcn_resnet101' : torchvision.models.segmentation.fcn_resnet101(),
        'fcn_resnet50' : torchvision.models.segmentation.fcn_resnet50(),
        'fasterrcnn_mobilenet_v3_large_fpn' : torchvision.models.detection.fasterrcnn_resnet50_fpn(),
        'fasterrcnn_mobilenet_v3_large_320_fpn' : torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(),
        'fasterrcnn_resnet50_fpn' : torchvision.models.detection.fasterrcnn_resnet50_fpn(),
        'maskrcnn_resnet50_fpn' : torchvision.models.detection.maskrcnn_resnet50_fpn(),
        'keypointrcnn_resnet50_fpn' : torchvision.models.detection.keypointrcnn_resnet50_fpn(),
        'retinanet_resnet50_fpn' : torchvision.models.detection.retinanet_resnet50_fpn(),
        'ssdlite320_mobilenet_v3_large' : torchvision.models.detection.ssdlite320_mobilenet_v3_large()
    }

    return model_dict[model_name]
