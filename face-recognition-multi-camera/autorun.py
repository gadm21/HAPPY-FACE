from subprocess import Popen
import sys

filename = sys.argv[1]

while True:
    # print("\nStarting " + filename)
    # p = Popen("python3 " + filename, shell=True)    
    p = Popen([sys.executable, filename], 
        stdout=open('log/out', 'a+'), 
        stderr=open('log/err', 'a+'))
    p.wait()