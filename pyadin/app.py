import pyadin.io
import pyadin.pipeline
import pyadin.processor
import re
import numpy as np
import logging

def check_device(args):
    devicelist = pyadin.io.SoundDeviceSource.query_device()
    """
    args: argparse.Namespece
    """
    print('--- avairable device list ---')
    print(devicelist)


def setup_source(config):
    """
    config: dict
    """
    logger = logging.getLogger(__name__)

    # microphone input
    if config['in'] == 'mic':
        devlist = pyadin.io.SoundDeviceSource.query_device()
        devname2id = {x['name']: i for i, x in enumerate(devlist)}
        devid2name = [x['name'] for i, x in enumerate(devlist)]

        device = config['device']
        if re.fullmatch(r'[0-9]+', device):
            deviceid = int(device)
            logger.info(f'[LOG]: SOURCE: use device {deviceid}:{devid2name[deviceid]}')
        else:
            deviceid = devname2id.get(config['device'])
            if deviceid is None:
                logger = logging.getLogger(__name__)
                logger.info(f'[LOG]: no such device name "{config["device"]}". use default setting.')

                deviceid = pyadin.io.SoundDeviceSource.get_default_device()
                if deviceid is None:
                    logger.info(f'[ERROR]: no such deviceid {deviceid}:{devid2name[deviceid]}.')
                    quit()

        logger.info(f'[LOG]: SOURCE: use device {deviceid}:{devid2name[deviceid]}')
        source = pyadin.io.SoundDeviceSource(deviceid, config['freq'], config['nch'], tgt_chs=config['tgt_chs'])

    # audio file input
    if config['in'] == 'file':
        filename = config.get('infile')
        if filename is None:
            print('type filename: ', end='')
            filename = input().strip()

        source = pyadin.io.AudioSourceFile(filename,
                                        config['freq'],
                                        config['nch'],
                                        160, tgt_chs=config['tgt_chs'],
                                        block=False)
        logger.info(f'[LOG]: SOURCE: use audio file {filename}')

    return source


def setup_sink(config):
    """
    config: dict
    """
    logger = logging.getLogger(__name__)
    sinks = []
    indices = []

    if 'adinnet' in config['out']:
        sink = pyadin.io.AdinnetSinkSocket(config['server'], config['port'])
        sinks.append(sink)
        indices.append(-1)
        logger.info(f'[LOG]: SINK: set out as "adinnet" {config["server"]} {config["port"]}')

    if 'file' in config['out']:
        filename = config.get('filename')
        if filename is None:
            logger.info(f'[ERROR]: there is no parameter "filename"')
            quit()
        
        sink = pyadin.io.SegmentedAudioSinkFile(config['filename'],
                                             config['startid'],
                                             config['freq'], config['nch'])
        sinks.append(sink)
        indices.append(-1)
        logger.info(f'[LOG]: SINK: set out as "file"')

    if 'queue' in config['out']:
        filename = config.get('filename')
        if filename is None:
            logger.info(f'[ERROR]: there is no parameter "filename"')
            quit()
        
        sink = pyadin.io.AudioSinkQueue()
        sinks.append(sink)
        indices.append(-1)
        logger.info(f'[LOG]: SINK: set out as "queue"')
        
    if config['enable_timestamp'] is True:
        sink = pyadin.io.TimestampTextSinkFile(config['timestampfile'])
        sinks.append(sink)
        indices.append(-1)
        logger.info(f'[LOG]: SINK: save timestamp')

    if config['enable_rawsave'] is True:
        sink = pyadin.io.AudioSinkFileRotate(config['rawfilefmt'],
                                          config['rotate_min'],
                                          config['freq'], 1)
        sinks.append(sink)
        indices.append(0)
        logger.info(f'[LOG]: SINK: save raw audio')

    return sinks, indices


def setup_lms(config):
    """
    config: dict
    """
    logger = logging.getLogger(__name__)
        
    lms = pyadin.processor.LMSblockFFT(config['L'], config['mu'])
    lms.load(config['filterfile'], config['floorfile'])
    
    logger.info(f'[LOG]: PROCESSOR: load echo-canceller: LMSblockFFT')

    return lms


def setup_tagger(config):
    """
    config: dict
    """
    logger = logging.getLogger(__name__)
    
    import importlib
    package = importlib.import_module(config['package'])
    classname = getattr(package, config['class'])
    tagger = classname(**config['params'])

    logger.info(f'[LOG]: PROCESSOR: load tagger: {config["package"]}.{config["class"]}')

    return tagger


def setup_postproc(config):
    """
    config: dict
    """
    logger = logging.getLogger(__name__)
    
    import importlib
    package = importlib.import_module(config['package'])
    classname = getattr(package, config['class'])
    postproc = classname(**config['params'])

    logger.info(f'[LOG]: PROCESSOR: load postproc: {config["package"]}.{config["class"]}')
    
    return postproc


def setup_plotwin(config, pipeline):
    """
    config: dict
    pipeline: class
    """
    logger = logging.getLogger(__name__)

    import pyadin.plot as plot
    plotwin = plot.RealtimeBufferedPlotWindow(pipeline, **config['plotwin'])

    return plotwin


def setup_logger(enable_logsave, logfilefmt):
    pyadin.io.setup_logger(enable_logsave, logfilefmt)


def run_realtime(config):
    """
    config: dict
    """
    logger = logging.getLogger(__name__)
    logger.info(f'[LOG]: setup pipelines')
    
    # io (source and sinks)
    source = setup_source(config)
    sinks, indices = setup_sink(config)
    pipeline = pyadin.pipeline.Pipeline(source, sinks=sinks, indices=indices)
    
    # processors
    if config['enable_ec'] is True:
        lms = setup_lms(config['lms'])
        pipeline.add(lms)        
    
    tagger = setup_tagger(config['tagger'])
    pipeline.add(tagger)

    postproc = setup_postproc(config['postproc'])
    if postproc is not None:
        pipeline.add(postproc)
    
    #
    if config['enable_plot'] is True:
        plotwin = setup_plotwin(config, pipeline)
        pipeline.open()
        logger.info(f'[LOG]: start processing')
        plotwin.run()
    else:
        pipeline.open()
        logger.info(f'[LOG]: start processing')
        pipeline.run()

    ##
    logger.info(f'[LOG]: end processing')
    pipeline.close()


def run_proclist(config):
    """
    config: dict
    """
    logger = logging.getLogger(__name__)
    logger.info(f'[LOG]: setup pipelines')

    #
    infilelist = config.get('inlist')
    tsfilelist = config.get('tslist')

    #
    if infilelist is None or tsfilelist is None:
        logger.info(f'[ERROR]: --inlist and --tslist are required for list processing')
        quit()
    
    # processors
    tagger = setup_tagger(config['tagger'])
    postproc = setup_postproc(config['postproc'])

    # change configurations
    config['in'] = 'file'
    config['out'] = '---'
    config['enable_timestamp'] = True

    #
    with open(infilelist) as inlist, open(tsfilelist) as tslist:
        for infile, tsfile in zip(inlist, tslist):
            infile = infile.strip()
            tsfile = tsfile.strip()
            
            config['infile'] = infile
            config['timestampfile'] = tsfile
            
            # io (source and sinks)
            source = setup_source(config)
            sinks, indices = setup_sink(config)
            pipeline = pyadin.pipeline.Pipeline(source, sinks=sinks, indices=indices)
            pipeline.add(tagger)
            if postproc is not None:
                pipeline.add(postproc)

            pipeline.open()
            logger.info(f'[LOG]: start processing for {infile} {tsfile}')
            pipeline.run()

            ##
            logger.info(f'[LOG]: end processing')
            pipeline.close()

            # reset state of processors
            tagger.reset()
            if postproc is not None:
                postproc.reset()

    logger.info(f'[LOG]: end of processing')

def usage_pyadintool():
    """
    return
        args: argparse.Namespece
    """
    import argparse

    # argument analysis
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('config', type=str)

    ###
    #  optional
    ##
    #  io
    parser.add_argument('--in', type=str,
                        help='mic | file | adinnet')
    parser.add_argument('--out', type=str,
                        help='file | adinnet')
    parser.add_argument('--filename', type=str,
                        help='output filename')
    parser.add_argument('--startid', type=int,
                        help='start number for filename')
    parser.add_argument('--server', type=str,
                        help='hostname of adin-server')
    parser.add_argument('--port', type=int,
                        help='port number of adin-server')

    parser.add_argument('--tgt_chs', type=int, nargs="*",
                        help='selected channel list for audio source')

    #
    parser.add_argument('--freq', type=int,
                        help='sampling frequency of input device in Hz')
    parser.add_argument('--nch', type=int,
                        help='the number of channels of input device')

    # parser.add_argument('--nosegment', action='store_const', const=True)
    # parser.add_argument('--oneshot', action='store_const', const=True)

    #
    parser.add_argument('--device', help='audio device id or name')

    #
    parser.add_argument('--infile', type=str,
                        help='input audio filename')

    parser.add_argument('--enable_logsave', action='store_const',
                        const=True, help='save log file')
    parser.add_argument('--logfilefmt', type=str,
                        help='log file format')

    #
    parser.add_argument('--enable_rawsave',
                        action='store_const', const=True,
                        help='save raw input signal to files')
    parser.add_argument('--rawfilefmt',
                        help='raw audio file format')
    parser.add_argument('--rotate_min', type=int,
                        help="duration in minutes for saving raw audio files")

    parser.add_argument('--enable_timestamp',
                        action='store_const', const=True,
                        help='output voice active sections in seconds')
    parser.add_argument('--timestampfile', type=str,
                        help="filename of timestamp file")
    #
    parser.add_argument('--enable_plot', action='store_const', const=True,
                        help='plot waveform and speech activations')

    #
    parser.add_argument('--enable_list',
                        action='store_const', const=True,
                        help='run batch processing')
    parser.add_argument('--inlist', type=str,
                        help='input audiofile list for batch processing')
    parser.add_argument('--tslist', type=str,
                        help='output labelfile list for batch processing')

    #
    parser.add_argument('--enable_ec',
                        action='store_const', const=True,
                        help='use LMS-based echo canceller. 2-channel audio for input is assumed.')
    
    #
    args = parser.parse_args()

    return args


def setup_config(args):
    """
    args: argparse.Namespece
    """
    # search package defautl config-file
    try:
        import os, sys
        d = os.path.dirname(sys.modules['pyadin'].__file__)
        filename = os.path.join(d, args.config)
        if os.path.isfile(filename) is not True:
            filename = args.config

        # load default config
        import yaml
        with open(filename, 'r') as yml:
            config = yaml.safe_load(yml)
    except Exception as e:
        print(f'{e}')
        quit()

    # update config
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    return config

def app_pyadintool():
    """
    """
    args = usage_pyadintool()

    #####################
    # preparation
    #####################
    # show list of devices
    if args.config == 'devinfo':
        check_device(args)
        quit()

    # load default config
    config = setup_config(args)

    # setup logger
    setup_logger(config['enable_logsave'], config['logfilefmt'])
    logger = logging.getLogger(__name__)
    logger.info(f'[LOG]: {config}')
    
    #####################
    #
    #####################
    if config.get('enable_list') is not None:
        run_proclist(config)
    else:
        run_realtime(config)
    pass

def setup_pipeline(queue):
    args = usage_pyadintool()
    if args.config == 'devinfo':
        check_device(args)
        quit()
    config = setup_config(args)
    setup_logger(config['enable_logsave'], config['logfilefmt'])
    logger = logging.getLogger(__name__)
    logger.info(f'[LOG]: {config}')

    logger.info(f'[LOG]: setup pipelines')
    
    # io (source and sinks)
    source = setup_source(config)
    sinks, indices = setup_sink(config)
    for x in sinks:
        if type(x) == pyadin.io.AudioSinkQueue:
            x.set(queue)
    pipeline = pyadin.pipeline.Pipeline(source, sinks=sinks, indices=indices)
    
    # processors
    if config['enable_ec'] is True:
        lms = setup_lms(config['lms'])
        pipeline.add(lms)        
    
    tagger = setup_tagger(config['tagger'])
    pipeline.add(tagger)

    postproc = setup_postproc(config['postproc'])
    if postproc is not None:
        pipeline.add(postproc)
    
    return pipeline


#######################
def estimate_filter(L, mu, filterfile, floorfile, n_sec, 
                    freq, nch, tgt_chs, deviceid=None):
    import sounddevice as sd
    import numpy as np
    from pyadin.processor import LMSblockFFT

    nlimit = freq * n_sec
    outdata = np.random.randn(nlimit,1).astype('float32')

    print('[LOG]: now recording ...')
    if deviceid is not None:
        sd.default.device = deviceid
    wavdata = sd.playrec(outdata, freq, channels=nch, blocking=True)

    print('[LOG]: now estimating filter ...')    
    # lazy implementation :P
    lms = LMSblockFFT(L, mu)
    errs = []
    for idx in range(0, len(wavdata), 1):
        errs.append(lms.update(wavdata[idx:idx+1,:], False))
    errs = np.concatenate(errs)
    
    lms.save(filterfile)
    np.savetxt(floorfile, np.array([np.var(errs[-freq*int(n_sec/2):])]))

    print(f'[LOG]: save to {filterfile} and {floorfile}')

def estimate_framepower(n_sec, freq,
                        default_pow_dur=20, n_fwd=0,
                        n_bwd=50, deviceid=None, nch=1):
    import sounddevice as sd
    import numpy as np
    import torch
    from usr.fdvad import BufferedWav2AmpSpec

    nlimit = freq * n_sec
    pow_dur = default_pow_dur if n_bwd > default_pow_dur else n_bwd

    print('[LOG]: now recording ...')
    if deviceid is not None:
        sd.default.device = deviceid
    wavdata = sd.rec(nlimit, freq, channels=nch, blocking=True)
    wav2aspec = BufferedWav2AmpSpec(n_fwd, n_bwd)
    
    feats = wav2aspec.get(torch.tensor(wavdata).reshape(1,-1), n_sec)
    sum_feats = torch.mean(torch.sum(feats[:,-pow_dur:,:]**2, dim=2),dim=1)

    n_len = len(sum_feats)
    beg_frame = int(n_len * 0.2)
    end_frame = int(n_len * 0.8)

    print(f'[LOG]: estimated frame-power:',
          f'mean: {torch.mean(sum_feats[beg_frame:end_frame]).item():.4f},',
          f'std: {torch.std(sum_feats[beg_frame:end_frame]).item():.4f}')


def calib_filter(args):
    print('[LOG]:', args)
    print('[LOG]: calibrate filter')
    pyadin.estimate_filter(args.L, args.mu,
                           args.filter_filename, args.floor_filename,
                           args.nsec, args.freq, args.nch,
                           args.tgt_chs, args.deviceid)

def calib_framepower(args):
    print('[LOG]:', args)
    print('[LOG]: calibrate power')
    pyadin.estimate_framepower(args.nsec, args.freq,
                               args.default_pow_dur,
                               args.n_fwd, args.n_bwd,
                               deviceid=args.deviceid, nch=args.nch)

def usage_auxtool():
    """
    return
        args: argparse.Namespece
    """
    import argparse

    # argument analysis
    parser = argparse.ArgumentParser()

    # required
    #parser.add_argument('mode', type=str)

    ###
    subparsers = parser.add_subparsers()

    ###
    parser_dev = subparsers.add_parser('devinfo')
    parser_dev.set_defaults(func=pyadin.check_device)
                           
    ### calibration of filter
    parser_ec = subparsers.add_parser('calib_filter')
    parser_ec.add_argument('--filter_filename', type=str, default='conf/ecfilter.txt',
                        help='filename for the estimated filter of echo canceller')
    parser_ec.add_argument('--floor_filename', type=str, default='conf/ecfloor.txt',
                        help='filename for the estimated variance of error signal')
    parser_ec.add_argument('--L', type=int, default=512,
                        help='filter length of echo canceller')
    parser_ec.add_argument('--mu', type=float, default=0.005,
                        help='learning rate of echo canceller')
    parser_ec.add_argument('--deviceid', type=int, default=None,
                        help='device id')
    parser_ec.add_argument('--freq', type=int, default=16000,
                        help='sampling frequency')
    parser_ec.add_argument('--nch', type=int, default=2,
                        help='number of channels')
    parser_ec.add_argument('--nsec', type=int, default=10,
                        help='recording time')
    parser_ec.add_argument('--tgt_chs', type=int, nargs="*", default=[0,1], 
                        help='channel list of audio source for processing')
    parser_ec.set_defaults(func=calib_filter)

    ### calibration of frame power
    parser_fp = subparsers.add_parser('calib_framepower')
    parser_fp.add_argument('--deviceid', type=int, default=None,
                        help='device id')
    parser_fp.add_argument('--default_pow_dur', type=int, default=20,
                        help='the number of frames for frame-power average calculation')
    parser_fp.add_argument('--nsec', type=int, default=10,
                        help='recording time')
    parser_fp.add_argument('--freq', type=int, default=16000,
                           help='sampling frequency')
    parser_fp.add_argument('--nch', type=int, default=1,
                        help='number of channels')
    parser_fp.add_argument('--n_fwd', type=int, default=0,
                        help='forward frames in FD-VAD based on DNN-HMM')
    parser_fp.add_argument('--n_bwd', type=int, default=50,
                        help='backward frames in FD-VAD based on DNN-HMM')
    parser_fp.set_defaults(func=calib_framepower)

    args = parser.parse_args()
    
    return args

def app_auxtool():
    args = usage_auxtool()
    try:
        args.func(args)
    except:
        print('use "--help" option for usage')
