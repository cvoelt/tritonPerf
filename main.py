
if __name__ == '__main__':
    import subprocess
    with open("tape.txt", "w+") as output:
        subprocess.call(["python3", "./tritonPerformance.py"], stdout=output)