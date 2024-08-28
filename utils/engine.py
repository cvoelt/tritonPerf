import subprocess
import typing

from numpy import append
from  . import config
import os
import logging
import ipaddress
import requests
import time
import threading

#handles the communication withe the triton-server and perf_analyzer


def measure(main_config: config.MainConfig, measure_config: config.MeasurementConfig, warmup= False, warmup_requests = 10) -> bool:

    command = ['/triton_server/clients/bin/perf_analyzer']
    if warmup:
        logging.info('Warming up Triton server for model ' + measure_config.get_folder_name() )
        #do at least 20 inferences for at least 1 second
        command.append('--measurement-mode')
        command.append('count_windows')
        #allow 99 percent difference between measurements since the results don't matter
        command.append('--stability-percentage')
        command.append('99')
        command.append('--measurement-request-count')
        command.append(str(warmup_requests))#repeats the requests at least three times, therefore *3 warmup requests are executed
    else:
        logging.info('executing measurement ' + str(measure_config.get_execution_count()) + ' of ' + str(main_config.get_num_measurements()))
        command.append('--measurement-interval')
        command.append(str(measure_config.get_measurement_interval() * 1000))
        command.append('-f')
        command.append(main_config.get_csv_dir().as_posix())    
    if measure_config.grpc_is_on():
        command.append('-u')
        command.append(main_config.get_triton_ip().compressed + ':8001')
        command.append('-i')
        command.append('gRPC')
    else:
        command.append('-u')
        command.append(main_config.get_triton_ip().compressed + ':8000')


    command.append('-m')
    command.append(measure_config.get_folder_name())
    command.append('-b')
    command.append(str(measure_config.get_batchsize()))
    command.append('--concurrency-range')
    command.append(str(measure_config.get_concurrency()) + ':' + str(measure_config.get_concurrency()))
    command.append('--shape')
    command.append('input:' + measure_config.get_shape())



    logging.info('executing the command' + str(command))

    measure_process = None
    try:
        measure_process = subprocess.Popen(command, stdout=subprocess.PIPE)
    except:
        logging.error('the command ' + str(command) + ' failed')
        return False, None, str(command)

    return True, measure_process, str(command)



def kill_triton_server(main_config: config.MainConfig, expect_off: bool = False) -> None:
    try:
        pid = int(subprocess.check_output(['pidof', main_config.get_triton_server_dir()]))
        subprocess.check_output(['kill', '-2', str(pid)])
        logging.debug('Sucessfully shut down triton server with p-id ' + str(pid))
        print('Shuting down triton server')
    except:
        if not expect_off:
            logging.error('Cant shutdown tritonserver: either no triton server or more than one is running!')
        else:
            logging.debug('Cant kill triton because no triton-server is running. This is an expected behaviour')


#execute ping request to trtion http-port, if timeout = 0, do it forever
def wait_for_triton_to_be_running(ip: ipaddress.IPv4Address, port: int = 8000, timeout = 100) -> bool:
    logging.debug('checking if triton has started with timeout ' + str(timeout) + ' seconds')
    time_elapsed = 0
    info_displayed = False

    while True:
        try:
            command = 'http://' + ip.compressed + ':' + str(port)
            logging.debug('executing ping request to ' + command)
            requests.post(command)
            logging.debug('server started sucessfully')
            return True
        except:
            time.sleep(1)
            if timeout == -1:
                if not info_displayed:
                    logging.error('Triton should be running! Please restart Trition manually!')
                    info_displayed = True
                continue
            time_elapsed +=1
            if time_elapsed > timeout:
                logging.error('Timeout for for triton start reached! Shutting down!')
                return False

def check_restart_triton(main_config: config.MainConfig, force_restart = False):

    #check if triton is running
    if not wait_for_triton_to_be_running(main_config.get_triton_ip(),main_config.get_triton_http_port(),0) or force_restart:
        logging.error('triton is not running/responding. Restarting triton server')
        if main_config.is_client_only():
            logging.error('Please restart triton manually!')
            wait_for_triton_to_be_running(main_config.get_triton_ip(),main_config.get_triton_http_port(),-1)
        else:
            kill_triton_server(main_config)
            time.sleep(35)
            start_triton_server(main_config)
            wait_for_triton_to_be_running(main_config.get_triton_ip(),main_config.get_triton_http_port())
            logging.info('load_unload model: Restart of Triton sucessful')
    return


def start_triton_server(main_config: config.MainConfig) -> None:
    kill_triton_server(main_config, expect_off=True)
    command = [main_config.get_triton_server_dir(), '--model-control-mode=explicit', '--model-repository=' + main_config.get_model_dir().as_posix(), '--strict-model-config=false', '--backend-directory=/triton_server/backends/']
    logging.info('Starting triton-server')
    logging.debug(' with the following settings:' + str(command))
    process = subprocess.Popen(command)


def load_unload_model_thread(main_config: config.MainConfig, model_name: str, operation_successful: typing.Dict[str, bool], unload = False) -> None:
    triton_ip = main_config.get_triton_ip().compressed
    triton_port = main_config.get_triton_http_port()
    command = ''
    r = None

    try:
        if unload:
            command = 'http://' + triton_ip + ':' + str(triton_port) + '/v2/repository/models/' + model_name + '/unload'
        else:
            command = 'http://' + triton_ip + ':' + str(triton_port) + '/v2/repository/models/' + model_name + '/load'
        r = requests.post(command)
    except:
        logging.error('there was en error executing th command' + command)
        operation_successful['success'] = False
        return

    if r.status_code != 200:
        if unload:
            logging.error('Could not unload: ' + model_name + ' on triton server')
        else:
            logging.error('Did not find the model: ' + model_name + ' on triton server')            
        operation_successful['success'] = False
        return
    if unload:
        logging.info('sucessfully unloaded the model: ' + model_name + ' on triton server')
    else:
        logging.info('sucessfully loaded the model: ' + model_name + ' on triton server')

    time.sleep(1)#allow an extra second for the server to get ready
    operation_successful['success'] = True
    return

def load_unload_model(main_config: config.MainConfig, model_name: str, unload=False, retries= 3) -> bool:
    
    #first, check if trtion is running. Restart if necessary
    check_restart_triton(main_config)

    for retry in range(0,retries):
        operation_successful = {'success':False}
        unloader_loader_thread: threading.Thread = threading.Thread(target=load_unload_model_thread(main_config, model_name, operation_successful, unload))
        unloader_loader_thread.start()
        unloader_loader_thread.join(timeout=120)

        #restart triton if load/unload timed out
        if unloader_loader_thread.is_alive():
            logging.error('load_unload model ' + model_name + ' timed out. Restarting Triton and trying again!')
            check_restart_triton(main_config, force_restart=True)
            if retry < retries-1:
                continue
            return False

        if not operation_successful['success']:
            if retry < retries-1:
                continue
            return False
        return True
    return False


def model_available(main_config: config.MainConfig, model_name: str) -> bool:
    triton_ip = main_config.get_triton_ip().compressed
    triton_port = main_config.get_triton_http_port()

    r = requests.get('http://' + triton_ip + ':' + str(triton_port) + '/v2/models/' + model_name + '/ready')
    if r.status_code != 200:
        logging.error('Model not (yet) available on triton-server: ' + model_name)
        return False
    logging.info('Model ' + model_name + ' is available on triton-server')
    return True
