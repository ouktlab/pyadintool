import lib.io
import lib.pipeline
import re
import numpy as np
import logging
    
def usage():
    import argparse
    
    # argument analysis
    parser = argparse.ArgumentParser(add_help=False)

    # required
    parser.add_argument('config', type=str)

    ###
    # optional
    ##
    # io
    parser.add_argument('--in', type=str, #default='mic',
                        help='mic | file | adinnet')
    parser.add_argument('--out', type=str, #default='adinnet',
                        help='file | adinnet')
    parser.add_argument('--filename', type=str, #default='result',
                        help='output filename')
    parser.add_argument('--startid', type=int, #default=0,
                        help='start number for filename')
    parser.add_argument('--server', type=str, #default='localhost',
                        help='hostname of adin-server')
    parser.add_argument('--port', type=int, #default=5530,
                        help='port number of adin-server')

    # 
    parser.add_argument('--freq', type=int)
    parser.add_argument('--nch', type=int)

    #
    #parser.add_argument('--nosegment', action='store_const', const=True)
    #parser.add_argument('--oneshot', action='store_const', const=True)

    #
    parser.add_argument('--device')
    
    #
    parser.add_argument('--infile', type=str, help='')
    
    parser.add_argument('--enable_logsave', action='store_const', const=True)
    parser.add_argument('--logfilefmt', type=str, help='')
    
    #
    parser.add_argument('--enable_rawsave', action='store_const', const=True)
    parser.add_argument('--rawfilefmt', help='')
    parser.add_argument('--rotate_min', type=int,
                        help="rotation time in minutes for saving raw audio files")

    parser.add_argument('--enable_timestamp', action='store_const', const=True)
    parser.add_argument('--timestampfile', type=str,
                        help="filename of timestamp file")

    #
    parser.add_argument('--enable_plot', action='store_const', const=True)

    ### 
    parser.add_argument('--enable_list', action='store_const', const=True)
    parser.add_argument('--inlist', type=str)
    parser.add_argument('--tslist', type=str)
    
    
    ###
    args = parser.parse_args()

    return args

'''
'''
def setup_config(args):
    # load default config
    import yaml
    with open(args.config, 'r') as yml:
        config = yaml.safe_load(yml)

    # update config
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    return config

'''
'''
def check_device(args):
    devicelist = lib.io.SoundDeviceSource.query_device()
    print('avairable device list---')
    print(devicelist)


'''
'''
def setup_source(config):
    logger = logging.getLogger(__name__)
    
    # microphone input
    if config['in'] == 'mic':
        devlist = lib.io.SoundDeviceSource.query_device()
        devname2id = {x['name']:i for i, x in enumerate(devlist)}
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

                deviceid = lib.io.SoundDeviceSource.get_default_device()
                if deviceid is None:
                    logger.info(f'[ERROR]: no such deviceid {deviceid}:{devid2name[deviceid]}.')
                    quit()

        logger.info(f'[LOG]: SOURCE: use device {deviceid}:{devid2name[deviceid]}')   
        source = lib.io.SoundDeviceSource(deviceid, config['freq'], config['nch'])

    # audio file input
    if config['in'] == 'file':
        filename = config.get('infile')
        if filename is None: 
            print('type filename: ', end='')
            filename = input().strip()
        
        source = lib.io.AudioSourceFile(filename,
                                    config['freq'],
                                    config['nch'],
                                    1600, block=False)
        logger.info(f'[LOG]: SOURCE: use audio file {filename}')            

    return source

'''
'''
def setup_sink(config):
    logger = logging.getLogger(__name__)
    sinks = []
    indices = []

    if 'adinnet' in config['out']:
        sink = lib.io.AdinnetSinkSocket(config['server'], config['port'])
        sinks.append(sink)
        indices.append(-1)
        logger.info(f'[LOG]: SINK: set out as "adinnet" {config["server"]} {config["port"]}')
    
    if 'file' in config['out']:
        filename = config.get('filename')
        if filename is None:
            logger.info(f'[ERROR]: there is no parameter "filename"')
            quit()
        
        sink = lib.io.SegmentedAudioSinkFile(config['filename'],
                                             config['startid'],
                                             config['freq'], config['nch'])
        sinks.append(sink)
        indices.append(-1)
        logger.info(f'[LOG]: SINK: set out as "file"')

    if config['enable_timestamp'] is True:
        sink = lib.io.TimestampTextSinkFile(config['timestampfile'])
        sinks.append(sink)
        indices.append(-1)
        logger.info(f'[LOG]: SINK: save timestamp')

    if config['enable_rawsave'] is True:
        sink = lib.io.AudioSinkFileRotate(config['rawfilefmt'],
                                          config['rotate_min'],
                                          config['freq'], 1)
        sinks.append(sink)
        indices.append(0)
        logger.info(f'[LOG]: SINK: save raw audio')

    return sinks, indices

'''
'''
def setup_tagger(config):
    logger = logging.getLogger(__name__)
    
    import importlib
    package = importlib.import_module(config['package'])
    classname = getattr(package, config['class'])
    tagger = classname(**config['params'])

    logger.info(f'[LOG]: PROCESSOR: load tagger: {config["package"]}.{config["class"]}')

    return tagger

'''
'''
def setup_postproc(config):
    logger = logging.getLogger(__name__)
    
    import importlib
    package = importlib.import_module(config['package'])
    classname = getattr(package, config['class'])
    postproc = classname(**config['params'])

    logger.info(f'[LOG]: PROCESSOR: load postproc: {config["package"]}.{config["class"]}')
    
    return postproc

'''
'''
def setup_plotwin(config, pipeline):
    logger = logging.getLogger(__name__)

    import lib.plot as plot
    plotwin = plot.RealtimeBufferedPlotWindow(pipeline, **config['plotwin'])

    return plotwin

"""
"""
def run_realtime(config):
    logger = logging.getLogger(__name__)
    logger.info(f'[LOG]: setup pipelines')
    
    # io (source and sinks)
    source = setup_source(config)
    sinks, indices = setup_sink(config)
    pipeline = lib.pipeline.Pipeline(source, sinks=sinks, indices=indices)
    
    # processors
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

"""
"""
def run_proclist(config):
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
            pipeline = lib.pipeline.Pipeline(source, sinks=sinks, indices=indices)
            pipeline.add(tagger)
            if postproc is not None:
                pipeline.add(postproc)

            pipeline.open()
            logger.info(f'[LOG]: start processing for {infile} {tsfile}')
            pipeline.run()

            ##
            logger.info(f'[LOG]: end processing')
            pipeline.close()

            ## reset state of processors
            tagger.reset()
            if postproc is not None:
                postproc.reset()
                         

    logger.info(f'[LOG]: end processing for all files')




"""
"""
def main():
    args = usage()

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
    lib.io.setup_logger(config['enable_logsave'], config['logfilefmt'])
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

"""

"""    
if __name__=="__main__":
    main()
