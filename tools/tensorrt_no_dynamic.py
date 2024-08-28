import pathlib
import os



def create_model_folder(model_name: str, model_path: str, file_type: str) -> pathlib.Path:
    root = pathlib.Path(model_path + '/' + model_name + '/')
    root.mkdir(exist_ok=True)
    subdirs = os.listdir(root)
    if not subdirs: #no subdirectories available
        new_folder = root.as_posix() + '/1/'
        os.makedirs(new_folder, exist_ok=True)
        return pathlib.Path(new_folder)

    else:
        subfolders_numbers = list(map(lambda ele : int(ele) if ele.isdigit() else 1 , subdirs)) #convert all strings to int if possible
        current_max_folder = max(subfolders_numbers)
        if not os.path.exists(root.as_posix() + '/' + str(current_max_folder) + '/model.' + file_type): #if current max folder has no model, deploy here
            return pathlib.Path(root.as_posix() + '/' + str(current_max_folder))
        else: #create new folder with consecutive number
            new_folder_path = pathlib.Path(root.as_posix() + '/' + str(current_max_folder + 1)+ '/')
            new_folder_path.mkdir()
            return new_folder_path



if __name__ == '__main__':

    model_name = 'resnet18'
    model_path = os.getcwd() + '//models/'
    #onnx_path =  model_path + model_name + '/1/model.onnx'
    onnx_path = os.getcwd() + '//zoo/' + model_name + '/1/model.onnx'

    batch_size = 1
    img_sizes = [224,320,448]

    for img_size in img_sizes:

        trt_folder = create_model_folder(model_name + 'trt_s' + str(img_size), model_path, 'plan')            

        min_shape = str(batch_size) + 'x3x' + str(img_size) + 'x' + str(img_size)
        opt_shape = str(batch_size) + 'x3x' + str(img_size) + 'x' + str(img_size)
        max_shape = str(batch_size) + 'x3x' + str(img_size) + 'x' + str(img_size)


        trt_exec = '/usr/src/tensorrt/bin/trtexec --verbose'
        onnx_file =  ' --onnx=' + onnx_path
        min_shapes = ' --minShapes=input:' + min_shape
        opt_shapes = ' --optShapes=input:' + opt_shape
        max_shapes = ' --maxShapes=input:' + max_shape
        workspace = ' --workspace=8192'
        save_engine = ' --saveEngine=' + trt_folder.as_posix() + '/model.plan'

        command = trt_exec + workspace + onnx_file + min_shapes + opt_shapes + max_shapes + save_engine
        print('Executing TRT-Optimization: ' + command)

        try:
            os.system(command)
        except:
            print('TRT-Optimization for model ' + model_name + ' failed!')