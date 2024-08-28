
def check():
    continue_now = False    
    while not continue_now:
        time.sleep(0.1)
        try:
            command = 'http://' + '0.0.0.0' + ':' + str(8000)
            requests.post(command)
            print('on')
            continue_now = True
        except:
            print('off')

def start_server():
    command = ['/triton_server/bin/tritonserver', '--model-control-mode=explicit', '--model-repository=/tmp/models/', '--strict-model-config=false', '--backend-directory=/triton_server/backends/']
    process = subprocess.Popen(command)
    print('Starting triton-server')    

if __name__ == '__main__':
    import requests
    import subprocess
    import time
    import threading
    server_thread: threading.Thread = threading.Thread(target = start_server())
    server_thread.start()


    check()

    try:
        pid = int(subprocess.check_output(['pidof', '/triton_server/bin/tritonserver']))
        subprocess.check_output(['kill', '-2', str(pid)])
        print('should be off')
        print('Shuting down triton server')
    except:
        if True:
            print('Cant shutdown tritonserver: either no triton server or more than one is running!')
        else:
            print('Cant kill triton because no triton-server is running. This is an expected behaviour')
    check()