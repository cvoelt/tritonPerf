import pandas
import utils.config as config
import utils.utilities as utilities
import utils.engine as engine
import utils.database as database
import utils.view as view
import utils.model_repo as model_repo
import utils.triton_metrics as triton_metrics
import logging
import threading


def main():

    #variables
    result_frame: pandas.DataFrame = None
    model_number: int = 0
    execution_count: int = -1
    measurement_config: config.MeasurementConfig = None

    
    #read config, setup tempdir, download models
    utilities.enable_console_and_root_folder_logging()
    main_config: config.MainConfig = config.MainConfig()
    main_config.set_results_dir(utilities.create_results_folder(main_config.get_results_dir()))
    utilities.enable_result_folder_logging(main_config)
    utilities.store_config_file(main_config) #saves the current config in the temp folder



    if not main_config.is_client_only():
        model_repo.check_download_models(main_config)        
        #start server
        server_thread: threading.Thread = threading.Thread(target = engine.start_triton_server(main_config))
        server_thread.start()
        #wait for triton, if timeout reached close program
        if not engine.wait_for_triton_to_be_running(main_config.get_triton_ip(), main_config.get_triton_http_port(), timeout=100):
            quit()

        if main_config.gpu_metrics_on():
            metrics_server = triton_metrics.JtopServer()
            metrics_server.start(main_config)
        


    #don't measure if we only need the server

    if not main_config.is_server_only():
        #iterate over all provided models and settings
        for model in main_config.get_models():

            for grpc_on in main_config.get_grpc_iterator():

                #iterate over trt if it is on
                #trt_suffixes is e.g. [''], ['','_trt_fp32'], ['','_trt_fp16','_cpu'], 
                for trt_cpu_precision_suffix in main_config.get_trt_cpu_suffixes():

                    #iterate over instances if it is on
                    #instance_suffixes is e.g. [''], ['','_i1'] or ['','_i1','_i2]
                    for instances, i_suffix in main_config.get_instance_suffixes().items():

                        #create measurement_config to get the proper model-folder-name
                        measurement_config = config.MeasurementConfig(
                            main_config = main_config,
                            execution_count = execution_count,
                            measurement_interval = main_config.get_measurement_interval(),
                            model_name = model,
                            grpc_on = grpc_on,
                            trt_cpu_suffix = trt_cpu_precision_suffix,
                            instances = instances,
                            model_number = model_number
                        )

                        models_with_longer_time_window = ('wide_resnet101_2', 'unetplusplus_efficientnet-b8', 'unetplusplus_resnet101', 'unetplusplus_resnet50', )
                        if model in models_with_longer_time_window:
                            measurement_config.add_value('measurement_interval', 20)

                        if main_config.gpu_metrics_on():
                            triton_metrics.measure_ram_once(main_config, measurement_config, '_before_model_load')

                        #unload model, try again, if request failed
                        if not engine.load_unload_model(main_config, measurement_config.get_folder_name(), unload=False, retries=3):
                            continue
    

                        for concurrency in main_config.get_concurrencies():
                            #abort if concurrency is higher than the instance count
                            if concurrency > instances:
                                continue

                            for shape in main_config.get_str_shapes():

                                #if model has a fixed shape, only allow this shape
                                fixed_shapes = ('256', '384', '512', '768', '1024')
                                if model.endswith(fixed_shapes):
                                    if shape[-3:] != model[-3:]:
                                        continue

                                measured_batchsizes: int = 0
                                measure_with_batchsize_minus_one: bool = False

                                for batchsize in main_config.get_batchsizes():

                                    if measure_with_batchsize_minus_one:
                                        batchsize -= 3

                                    execution_count += 1
                                    measured_batchsizes += 1

                                    one_measurement_frame = None
                                    measurement_config.add_value('execution_count', execution_count)
                                    measurement_config.add_value('batchsize', batchsize)       
                                    measurement_config.add_value('shape', shape)
                                    measurement_config.add_value('concurrency', concurrency)   
                               

                                    if main_config.gpu_metrics_on():
                                        triton_metrics.measure_ram_once(main_config, measurement_config, '_before_measure')

                                    if main_config.warmup_is_on():
                                        measure_start_successful, measure_process, command = engine.measure(main_config, measurement_config, warmup=True)
                                        if not measure_start_successful:
                                            database.add_empty_row_to_dataframe(result_frame, execution_count, measurement_config, main_config)
                                            database.save_results(result_frame, main_config)
                                            continue
                                        measure_process.communicate()
                                        if measure_process.returncode != 0:
                                            logging.error('there was an error executing the command' + command)
                                            database.add_empty_row_to_dataframe(result_frame, execution_count, measurement_config, main_config)
                                            database.save_results(result_frame, main_config)
                                            continue                       

                                    #measure and if it fails, start loop again with next model
                                    measure_start_successful, measure_process, command = engine.measure(main_config, measurement_config)

                                    if not measure_start_successful:
                                        logging.error('there was an error executing the command' + command)
                                        database.add_empty_row_to_dataframe(result_frame, execution_count, measurement_config, main_config)
                                        database.save_results(result_frame, main_config)
                                        continue

                                    #collect metrics while measurement is running
                                    if main_config.gpu_metrics_on():
                                        triton_metrics.measure_ram_until_process_finished(measure_process, main_config, measurement_config)

                                    #wait for measurement to finish
                                    out = measure_process.communicate()[0]
                                    if measure_process.returncode != 0:
                                        logging.error('there was an error executing the command' + command)
                                        database.add_empty_row_to_dataframe(result_frame, execution_count, measurement_config, main_config)
                                        database.save_results(result_frame, main_config)
                                        continue

                                    std_dev = utilities.find_standard_deviation(out)
                                    measurement_config.add_value('standard_dev', std_dev)
                                    one_measurement_frame = database.csv_to_dataframe(main_config.get_csv_dir(), main_config, measurement_config)                                    
                                    result_frame = database.concat_dataframes(result_frame, one_measurement_frame)
                                    database.save_results(result_frame, main_config)

                                    if main_config.find_max_batchsize_mode_on():
                                        print('exec count')
                                        print(measured_batchsizes)
                                        if measured_batchsizes > 1:
                                            print('exec count higher thatn 1')
                                            if measure_with_batchsize_minus_one:
                                                break
                                            if database.less_inferences(result_frame,one_measurement_frame):
                                                if measured_batchsizes == 2:
                                                    break
                                                measure_with_batchsize_minus_one = True

                        

                        #unload model, try again, if request failed
                        if not engine.load_unload_model(main_config, measurement_config.get_folder_name(), unload=True, retries=3):
                            continue

            #if at least one measurement worked, create plot and save data
            #if one_model_frame is not None:
            model_number += 1
                #view.create_plot(model,one_model_frame, main_config)              
                #result_frame = database.concat_dataframes(result_frame, one_model_frame)
                #save database in results-folder
            

        #view.create_overview_plot(main_config)

        #display and store results
        if result_frame is not None:
            print('Final results:')
            print(result_frame)
            

            #add results to existing main database
            connection = database.create_connection(utilities.global_db_filename())
            database.save_dataframe_to_db(result_frame, utilities.get_timestamp(), connection)

        else:
            logging.error('no measurements have been performed! Check log and model names!')

    #server_only-mode
    if main_config.is_server_only():
        logging.info('Server only mode, waiting for input')
        exit = False
        while not exit:
            user_input = input('Server only mode, type \'exit\' to stop server...')
            if user_input == 'exit':
                exit = True
        logging.info('server only mode, waiting for input')

    if not main_config.is_client_only():
        if main_config.gpu_metrics_on():
            metrics_server.stop()
        engine.kill_triton_server(main_config)
        logging.info('tritonPerformance finished!')
    
if __name__ == '__main__':
    main()