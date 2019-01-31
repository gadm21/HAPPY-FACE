from subprocess import Popen
import sys

filename = sys.argv[1]

try:
    while True:
        # print("\nStarting " + filename)
        # p = Popen("python3 " + filename, shell=True)    
        # timeout
        p = Popen([sys.executable, filename], 
            stdout=open('log/out.log', 'a+'), 
            stderr=open('log/err.log', 'a+'),
            )
        p.wait()
except:
    print('Terminate long running script')
    p.terminate()