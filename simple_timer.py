import time

# A simple class for timing contexts
class IndividualTimer:
    def __init__(self, name, n_iter=None):
        self.name = name
        self.t_start = None
        self.t_end = None
        self.n_iter = n_iter
        self.t_elapsed = None
        self.sec_per_iter = None
        self.iter_per_sec = None
        
        
    def __enter__(self):
        self.t_start = time.time()
        
        
    def __exit__(self, type, value, traceback):
        self.t_end = time.time()
        self.t_elapsed = self.t_end - self.t_start
        
        self.n_iter = 1 if self.n_iter is None else self.n_iter
        self.sec_per_iter = self.t_elapsed / self.n_iter
        self.iter_per_sec = self.n_iter / self.t_elapsed 
        
        self.print_timing()
        
        
    def print_timing(self):
        
        if self.n_iter == 1:
            print(f'Total time elapsed for timer {self.name}: {self.t_elapsed:.3f} seconds')
        else:
            print(f'Timings ({self.name}):')
            print(f'  {self.n_iter} iterations in {self.t_elapsed:.3f} seconds')
            print(f'  {self.iter_per_sec:.4f} iter/sec')
            print(f'  {self.sec_per_iter:.5f} sec/iter')
            
            
# This is a Singleton timer wrapper class, which will track all timers in a simple way
class SimpleTimers:
    all_timers = {}
    
    @classmethod
    def new(cls, name, n_iter=None):
        cls.all_timers[name] = IndividualTimer(name, n_iter)
        return cls.all_timers[name]
        
    @classmethod
    def get_timer(cls, name):
        return cls.all_timers[name]

    @classmethod
    def get_iter_per_sec(cls, name):
        return cls.all_timers[name].iter_per_sec
    
    @classmethod
    def get_sec_per_iter(cls, name):
        return cls.all_timers[name].sec_per_iter
