from abc import ABCMeta, abstractclassmethod

"""
"""
class Source(metaclass=ABCMeta):
    @abstractclassmethod
    def open(self):
        pass

    @abstractclassmethod
    def close(self):
        pass

    '''
    The size of data for "read" must be defined in each child class.
    The shape of data may be [Len, Ch] with the numpy darray format 
    in many cases. 
    '''
    @abstractclassmethod
    def read(self):
        pass

'''
'''
class Sink(metaclass=ABCMeta):
    @abstractclassmethod
    def open(self):
        pass

    @abstractclassmethod
    def close(self):
        pass

    @abstractclassmethod
    def write(self, data):
        pass


'''
'''
class Processor(metaclass=ABCMeta):
    def open(self):
        pass

    def close(self):
        pass

    def reset(self):
        pass

    @abstractclassmethod
    def update(self, data, isEOS):
        pass

'''
'''
class Pipeline:
    class _True:
        def do(self):
            return True
    
    def __init__(self, source, sinks=[], indices=[]):
        self.source = source
        self.sinks = sinks
        self.indices = indices
        self.procs = []
        pass

    def open(self):
        self.source.open()
        if self.sinks is not None:
            for sink in self.sinks:
                sink.open()
        for x in self.procs:
            x.open()
            
    def close(self):
        for x in self.procs:
            x.close()
            
        self.source.close()
        if self.sinks is not None:
            for sink in self.sinks:
                sink.close()

    def add(self, proc):
        self.procs.append(proc)

    def update(self):
        # input
        inputs = self.source.read()

        # end of input
        isEOS = True if inputs is None else False

        # 
        data = []
        data.append(inputs)
        
        # proc
        for m in self.procs:
            inputs = m.update(inputs, isEOS)
            data.extend(list(inputs)) if type(inputs) == tuple else data.append(inputs)
        
        # output
        if self.sinks is not None:
            for sink, idx in zip(self.sinks, self.indices):
                if data[idx] is not None:
                    sink.write(data[idx])
        
        if isEOS is True:
            return None
                
        return data

    # 
    def run(self, judge=None):
        if judge is None:
            judge = self._True()

        try:
            while judge.do() and (ret := self.update()) is not None:            
                pass
        except KeyboardInterrupt:
            print('')
            pass

