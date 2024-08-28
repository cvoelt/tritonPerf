import jtop
import json

if __name__ == '__main__':
    with jtop.jtop() as jetson:

        #print(jetson.cpu)
        #print(json.dumps(jtop.jtop().gpu)
        import pprint
        pprint.pprint(jetson._stats)