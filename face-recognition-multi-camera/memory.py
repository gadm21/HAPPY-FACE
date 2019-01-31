# import resource
import psutil
import GPUtil
import threading
import time
import logging

formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                              "%Y-%m-%d %H:%M:%S")

def setupLogger(name,logFile,level=logging.INFO):
    handler = logging.FileHandler(logFile)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

class Memory:
    def __init__(self):
        self.timeInterval = 30
        self.gpuMemoryFull = False
        self.cpuMemoryFull = False
        self.cpuLogger = setupLogger('cpuLogger','log/cpu.log')
        self.gpuLogger = setupLogger('gpuLogger','log/gpu.log')
        self.cpuFreeMemoryThreshold = 100000000

    def cpuInfo(self):
        # memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # msg = 'CPU Memory: {0}{1}'.format(memory,'kB')
        memory = psutil.virtual_memory().free
        msg = 'CPU Free Memory: {0}{1}'.format(memory,'B')
        self.cpuLogger.info(msg)
        if memory < self.cpuFreeMemoryThreshold:
            self.cpuLogger.info('Memory above threshold')
            self.cpuMemoryFull = True
            return
        time.sleep(self.timeInterval)
        self.cpuInfo()

    def gpuInfo(self):
        gpus = GPUtil.getGPUs()
        if len(gpus) > 0:
            msg = 'GPU Free Memory: {0}{1}'.format(gpus[0].memoryFree,'MB')
            self.gpuLogger.info(msg)
        else:
            msg = 'No GPU available'
            self.gpuLogger.info(msg)
        time.sleep(self.timeInterval)
        self.gpuInfo()

    def checkMemory(self):
        result = not self.gpuMemoryFull and not self.cpuMemoryFull
        return result

    def run(self):
        cpuThread = threading.Thread(target=self.cpuInfo)
        cpuThread.daemon = True
        cpuThread.start()
        gpuThread = threading.Thread(target=self.gpuInfo)
        gpuThread.daemon = True
        gpuThread.start()