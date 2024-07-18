import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

class RealtimeBufferedPlotWindow:
    """
      pipline: processor
      nbuffer: buffering time for plot (refresh if new data are buffered for nbuffer samples)
      selects: select data for plot
      nskip: decimate data for plot      
    """
    def __init__(self, pipeline, nbuffer,
                 selects=[0],
                 nsample=16000*2,
                 nch=1, nskip=1, scale=1.0):
        # main processor
        self.pipeline = pipeline
        self.nsample = nsample
        self.scale = scale
        self.nch = nch
        self.nbuffer = nbuffer
        self.selects = selects
        self.nskip = nskip
        
        # initialization of GUI
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle('Real-time Plot')

        #
        self.plots = []
        self.curves = []
        for c in range(nch):
            plot = self.win.addPlot(row=c, col=0)
            plot.setYRange(-1.1, 1.1)
            
            self.plots.append(plot)
            self.curves.append(plot.plot())
        
        # audio buffer
        self.data = np.zeros((nsample, nch), dtype=float)
        
    def update(self):
        n_total = 0
        while n_total < self.nbuffer:
            wavs = self.pipeline.update()
        
            if wavs is None:
                self.win.close()
                return
            
            if isinstance(wavs, list):
                wavs_ = []
                for i in self.selects:
                    wavs_.append(wavs[i])
                wavs = np.concatenate(wavs_, axis=1)

            ###
            n_len = len(wavs)
            self.data = np.roll(self.data, shift=-n_len, axis=0)
            self.data[-n_len:] = wavs

            n_total += n_len

        for c in range(self.nch):
            self.curves[c].setData(self.data[::self.nskip,c] * self.scale)

    def run(self):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)        
        self.timer.start(0)

        try:
            pg.exec()
        except KeyboardInterrupt:
            pass

        print('\n[LOG]: exit loop')

if __name__=="__main__":
    pass
        
