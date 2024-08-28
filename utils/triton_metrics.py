from . import config
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json
import logging
import subprocess
import requests
import typing
import statistics
import time
try:
    import jtop
    jtop_import_successful = True
except ImportError:
    jtop_import_successful = False
    logging.error('Could not import module jtop')



class Handler(BaseHTTPRequestHandler):

    def setup(self):
        BaseHTTPRequestHandler.setup(self)
        self.request.settimeout(5)

    def do_GET(self) -> None:
        try:
            jetson = jtop.jtop()
            jetson.start()
            message = None  
            message = str.encode(json.dumps(jetson._stats))
            jetson.close()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(message)
            self.wfile.write(str.encode('\n'))
        except:
            logging.error('failed to send GPU-Metrics! Check if the Docker image is \n started with -v /run/jtop.sock:/run/jtop.sock Turn the setting \'gpu_metrics\' off if not on jetson')
        return
    
    def log_message(self, format, *args):
        return


class JtopServer():

    def start(self, main_config: config.MainConfig) -> None:
        if not jtop_import_successful:
            logging.error('failed to start GPU-Metrics Server! Check if the package jtop is installed. Turn the setting \'gpu_metrics\' off if not on jetson')
        else:
            logging.info('Starting GPU-Metrics-Server')
            server_ip = main_config.get_metrics_server_ip().compressed
            server_port = main_config.get_metrics_server_port()
            self.server = HTTPServer(('0.0.0.0', server_port), Handler)
            self.thread = threading.Thread(target = self.server.serve_forever)
            self.thread.start()

    def stop(self) -> None:
        logging.info('Shutting down GPU-Metrics-Server')
        self.server.shutdown()
        print(self.thread.is_alive())
        print('finished')

def measure_ram_metrics(main_config: config.MainConfig) -> typing.Dict[str,int]:
    server_ip = main_config.get_metrics_server_ip()
    server_port = main_config.get_metrics_server_port()
    server_address = 'http://' + server_ip.compressed + ':' + str(server_port)
    metrics = None
    try:
        metrics = requests.get(server_address, timeout=7).json()
    except:
        logging.error('could not connect to metrics server with ip: ' + server_address)
        return_metrics = {
            'gpu_ram' : 0,
            'total_ram' : 0,
            'all_stats' : json.dumps(metrics)
        }
        return return_metrics

    return_metrics = {
        'gpu_ram' : metrics['ram']['shared']/1000,
        'total_ram' : metrics['ram']['use']/1000,
        'all_stats' : json.dumps(metrics)
    }

    return return_metrics

def measure_ram_until_process_finished(measure_process: subprocess.Popen, main_config: config.MainConfig, measurement_config: config.MeasurementConfig) -> None:
    logging.info('collecting metrics during performance measurement')
    gpu_ram_during_measure = []
    total_ram_during_measure = []
    measurement_finished = False
    while not measurement_finished:
        ram_during_measure = measure_ram_metrics(main_config)
        if ram_during_measure['gpu_ram'] != 0:
            gpu_ram_during_measure.append(ram_during_measure['gpu_ram'])
            total_ram_during_measure.append(ram_during_measure['total_ram'])
        time.sleep(1)
        measurement_finished = measure_process.poll() is not None

    if not gpu_ram_during_measure:
        gpu_ram_during_measure = [0]
        total_ram_during_measure = [0]
        logging.error('RAM measurement during subprocess was not successful')
    measurement_config.add_value('gpu_ram_during', statistics.mean(gpu_ram_during_measure))
    measurement_config.add_value('total_ram_during', statistics.mean(total_ram_during_measure))

def measure_ram_once(main_config: config.MainConfig, measurement_config: config.MeasurementConfig, prefix: str) -> None:
    ram_before_measure = measure_ram_metrics(main_config)
    for value in ram_before_measure:
        measurement_config.add_value(value + prefix, ram_before_measure[value])