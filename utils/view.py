import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import title
from . import config
from . import utilities
import math
import pandas
import pathlib


def create_plot(model_name:str, df: pandas.DataFrame, main_config: config.MainConfig) -> None:
    df = df.set_index('execution_count')
    df = df.sort_values('execution_count')
    plotFrame = df[['ClientSend','NetworkandServerSend_Recv','ServerQueue','ServerComputeInput','ServerComputeInfer','ServerComputeOutput','ClientRecv']]
    plotFrame = plotFrame.div(1000)#divide to get milliseconds

    barPlot = plotFrame.plot.bar(stacked=True,title=model_name)

    x_labels = []
    for execution_count in plotFrame.index:
        width = main_config.get_measurement_config(execution_count).get_shape().rsplit(',', 1)[1]
        x_labels.append('ex_c: ' + str(execution_count) + ' shape: ' + width)
        #batchsize = main_config.get_measurement_config(execution_count).get_batchsize()
        #x_labels.append(' batchs: ' + str(batchsize))
        #x_labels.append(' shape: ' + width)



    barPlot.set_xticklabels(x_labels, rotation='horizontal')
    barPlot.set_ylabel('milliseconds')
    barPlot.set_xlabel('')
    barPlot.get_figure().savefig(main_config.get_results_dir().as_posix() + '/' + model_name +'plot')
    
    return barPlot.get_figure()

def create_overview_plot(main_config: config.MainConfig) -> None:

    image_paths = utilities.get_jpg_files_from_results_folder(main_config)
    concat_images(image_paths, main_config)
    num_models = main_config.get_num_of_models()

    grid_size = math.ceil(math.sqrt(num_models))
    overview_fig = matplotlib.pyplot.figure()
    
    return grid_size, overview_fig

def concat_images(images: pathlib.Path, main_config: config.MainConfig):
    import sys
    from PIL import Image

    images = [Image.open(x) for x in images]
    try:
        widths, heights = zip(*(i.size for i in images))
    except:
        logging.error('there are no images, cant concatenate them!')
        return

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

        new_im.save(main_config.get_results_dir().as_posix() + '/overview.jpg')