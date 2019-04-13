import time
from statistics import mean


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


class SimpleInstrument:
    def __init__(self):
        self.start_t = None
        self.end_t = None

    def start(self):
        self.start_t = time.time()

    def end(self):
        self.end_t = time.time()


class SimpleReport:
    def __init__(self):
        self.instr = []
        self.elapsed = []

    def append(self, instr):
        self.instr.append(instr)
        t = instr.end_t - instr.start_t
        self.elapsed.append(t)

    def dump(self):
        hits = len(self.instr)
        ave = mean(self.elapsed)
        print(f"hits : {hits}, average : {ave}")


class WorkerInstrument:
    def __init__(self):
        self.main_start_t = None
        self.worker_start_t = None
        self.worker_return_t = None
        self.main_stop_t = None

    def main_start(self):
        self.main_start_t = time.time()

    def worker_start(self):
        self.worker_start_t = time.time()

    def worker_return(self):
        self.worker_return_t = time.time()

    def main_stop(self):
        self.main_stop_t = time.time()


class TimingReport:
    def __init__(self):
        self.timings = []
        self.init_t = []
        self.work_t = []
        self.return_t = []

    def append(self, instr):
        self.timings.append(instr)
        init_t = instr.worker_start_t - instr.main_start_t
        work_t = instr.worker_return_t - instr.worker_start_t
        return_t = instr.main_stop_t - instr.worker_return_t
        self.init_t.append(init_t)
        self.work_t.append(work_t)
        self.return_t.append(return_t)

    def dump(self):
        hits = len(self.timings)
        ave_init = mean(self.init_t)
        ave_work = mean(self.work_t)
        ave_return = mean(self.return_t)

        print(f'hits : {hits} ave_init : {ave_init} ave_work : {ave_work} ave_return : {ave_return}')
