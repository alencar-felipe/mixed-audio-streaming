#!/usr/bin/env python

import argparse
import logging
import msgpack
import numpy as np
import os
import socket
import sounddevice as sd

from pathlib import Path
from queue import SimpleQueue, Empty
from scipy.io import wavfile
from threading import Thread, Event
from time import sleep, time

# if False, all threads should stop
running = True

# Server ======================================================================#

def server_main(addr: tuple[str, int], max_clients: int):
    """
    Server main function

    Parameters
    ----------
    addr : (str, int)
        a (host, port) pair, where  host is a string representing either a
        hostname or an IPv4 address and port is an integer
    
    max_clients : int
        the maximum number of simultaneous clients
    """

    global running

    logging.info(f'Starting server on: {addr}')

    # creates socket object
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # allow port reuse
    tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 

    # listen of specified port
    tcp.bind(addr)
    tcp.listen(max_clients)

    threads = []

    # accept clients while running
    try:
        while running:
            client, addr = tcp.accept()
            new_thread = Thread(target=client_thread,args=(client, addr))
            new_thread.start()
            threads.append(new_thread)

    except KeyboardInterrupt:
        logging.info("Exiting")
        running = False

def client_thread(client, addr):
    """
    Target function for the threads that handle clients

    Parameters
    ----------
    client : socket
        The client socket
    
    addr : (str, int)
        The client address
    """

    global running

    logging.info(f'{addr}: Starting client thread')

    # creates udp socket object
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # creates iterator for incoming msgpack data
        unpacker = tcp_unpaker(client)

        # receive client header
        h = next(unpacker)
        audio_file = h['audio_file']
        udp_port = h['udp_port']
        part_size = h['part_size']
        parts_in_window = h['parts_in_window']
        mid_start = h['mid_start']
        mid_end = h['mid_end']

        # check if file is in project subfolder and is an wav file
        if not is_legal_audio_file(audio_file):
            raise RuntimeError('Ilegal audio file')

        # Load audio samples into memory
        sample_rate, samples = wavfile.read(audio_file)

        # normalize audio
        samples = normalize(samples)

        # send server header
        client.send(msgpack.packb({
            'sample_rate': sample_rate
        }))

        # calculates discrete cossine transform window size 
        win2 = part_size * parts_in_window
        win = win2 * 2

        # loop through audio sending data
        index = 0
        start = time()
        for k in range(0, len(samples) - win, win2):
            
            # stops if necessary
            if not running:
                break
            
            # performs discrete consine transform
            spec = mdct4(samples[k:k+win])
            
            # cut spectrum in parts
            parts = spec.reshape((parts_in_window, part_size)) 

            # enumerate parts
            enum_parts = list(enumerate(parts)) 

            # lossy (UDP tramsmission)
            low = enum_parts[:mid_start]
            high = enum_parts[mid_end:]
            for (num, data) in low or high:
                udp.sendto(msgpack.packb({
                    'index': index,
                    'part': num,
                    'data': np_encode(data)
            }), (addr[0], udp_port))

            # waits in order not to overfeed the client
            while time() + 1 < start + k*(1/sample_rate):
                sleep(0.01)

            # lossless (TCP transmission)
            mid = parts[mid_start:mid_end]
            client.send(msgpack.packb({
                'index': index,
                'mid': np_encode(mid)
            }))

            index += 1

    except Exception as e:
        logging.error(f'{addr}: {str(e)}')

    logging.info(f'{addr}: Closing client thread')
    client.close()

# Client ======================================================================#

def client_main(addr: tuple[str, int], audio_file: str, part_size: int,
    parts_in_window: int, mid_start: int, mid_end: int):
    """
    Client main function

    Parameters
    ----------
    addr : (str, int)
        a (host, port) pair, where  host is a string representing either a
        hostname or an IPv4 address and port is an integer
    
    audio_file : str
        relative location of file in remote server

    part_size : int
        size of the spectrum slice that each udp packet will carry

    parts_in_window : int
        number of spectrum slices in a window

    mid_start : int
        index to beggining of medium frequencies slices

    mid_end : int
        index to end of medium frequencies slices
    """

    global running

    logging.info(f'Starting client')
    
    # creates both socket objects
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # bind to any available udp port
        udp.bind(('', 0))
        udp_port = udp.getsockname()[1]

        # connect to server via tcp
        tcp.connect(addr)
        unpacker = tcp_unpaker(tcp)
        
        # send client header
        tcp.send(msgpack.packb({ 
            'audio_file': audio_file,
            'udp_port': udp_port,
            'part_size': part_size,
            'parts_in_window': parts_in_window,
            'mid_start': mid_start,
            'mid_end': mid_end
        }))

        # receive server header
        h = next(unpacker)
        sample_rate = h['sample_rate']

        # calculates discrete cossine transform window size 
        win2 = part_size * parts_in_window
        win = win2 * 2

        # creates thread that receives udp packets
        lossy_parts = []
        thread1 = Thread(target=udp_thread,args=(udp, 1024, lossy_parts))
        thread1.start()

        # creates thread that actually plays the samples
        queue = SimpleQueue()
        thread2 = Thread(target=play_thread,args=(sample_rate, win2, queue))
        thread2.start()

        # helper variable
        mid_size = mid_end - mid_start
        
        # mdct madness
        last_z = np.zeros(win)

        # loop through incoming msgpack data (TCP)
        for unpacked in unpacker:
            # stops if necessary
            if not running:
                break
            
            # decode data
            index = unpacked['index']
            mid = np_decode(unpacked['mid'])
            mid = mid.reshape((mid_size, part_size))

            # create empty spectrum
            spec = np.zeros((parts_in_window, part_size))
            
            # glue mid in spectrum
            spec[mid_start:mid_end] = mid
            
            # glue low and high in spectrum
            parts = [part for part in lossy_parts if part['index'] == index]
            for part in parts:
                j = part['part']
                spec[j] = np_decode(part['data'])
                lossy_parts.remove(part)

            # remove parts that arrived late and wont be used
            parts = [part for part in lossy_parts if part['index'] < index]
            for part in parts:
                lossy_parts.remove(part)

            # performs inverse cossine transform
            z = imdct4(spec.reshape((-1)))

            # inverse conssine transform windows madness
            overlap = last_z[win2:win] + z[:win2]
            last_z = z

            # push samples to play_thread
            queue.put(overlap/2)

    except KeyboardInterrupt as e:
        logging.info('KeyboardInterrupt')
    except Exception as e:
        raise e
        logging.error(str(e))

    logging.info('Closing client')
    running = False
    tcp.close()
    udp.close()

def udp_thread(udp, buflen, lossy_parts):
    """
    Target function for thread that receives udp packets

    Parameters
    ----------
    udp : socket
        socket for incoming UDP packets
    
    buflen : int
        max size of UDP packet

    lossy_parts: list
        list that stores udp packets
    """

    # enable non-blocking mode
    udp.setblocking(0)

    # receives packets while running
    while running:
        try:
            data, addr = udp.recvfrom(buflen)
            part = msgpack.unpackb(data)
            lossy_parts.append(part) # thread safe operation
        except BlockingIOError as e:
            sleep(0.01)

    logging.info('Closing udp_thread')

def play_thread(sample_rate, blocksize, samples):
    """
    Target function for thread that plays audio samples

    Parameters
    ----------
    sample_rate : int
        sample rate

    blocksize : int
        size of each element in queue

    samples: SimpleQueue
        queue for blocks of audio samples
    """

    global running

    event = Event()

    def callback(outdata, frames, time, status):
        while samples.empty() and running:
            sleep(0.01)

        if not running:
            raise sd.CallbackAbort

        data = samples.get_nowait()

        #while not samples.empty():
        #    samples.get_nowait()

        outdata[:] = data.reshape((-1, 1))

    stream = sd.OutputStream(samplerate=sample_rate, channels=1,
        callback=callback, blocksize=blocksize, finished_callback=event.set)

    with stream:
        while not event.is_set() and running:
            sleep(0.01)

    logging.info('Closing play_thread')

# MDCT ========================================================================#
# https://github.com/smagt/mdct

def mdct4(x):
    N = x.shape[0]
    if N%4 != 0:
        raise ValueError(
            "MDCT4 only defined for vectors of length multiple of four.")
    
    M = N // 2
    N4 = N // 4
    
    rot = np.roll(x, N4)
    rot[:N4] = -rot[:N4]
    t = np.arange(0, N4)
    w = np.exp(-1j*2*np.pi*(t + 1./8.) / N)
    c = np.take(rot,2*t) - np.take(rot, N-2*t-1) \
        - 1j * (np.take(rot, M+2*t) - np.take(rot,M-2*t-1))
    c = (2./np.sqrt(N)) * w * np.fft.fft(0.5 * c * w, N4)
    y = np.zeros(M)
    y[2*t] = np.real(c[t])
    y[M-2*t-1] = -np.imag(c[t])
    return y

def imdct4(x):
    N = x.shape[0]
    if N%2 != 0:
        raise ValueError("iMDCT4 only defined for even-length vectors.")
    M = N // 2
    N2 = N*2
    
    t = np.arange(0,M)
    w = np.exp(-1j*2*np.pi*(t + 1./8.) / N2)
    c = np.take(x,2*t) + 1j * np.take(x,N-2*t-1)
    c = 0.5 * w * c
    c = np.fft.fft(c,M)
    c = ((8 / np.sqrt(N2))*w)*c
    
    rot = np.zeros(N2)
    
    rot[2*t] = np.real(c[t])
    rot[N+2*t] = np.imag(c[t])
    
    t = np.arange(1,N2,2)
    rot[t] = -rot[N2-t-1]
    
    t = np.arange(0,3*M)
    y = np.zeros(N2)
    y[t] = rot[t+M]
    t = np.arange(3*M,N2)
    y[t] = -rot[t-3*M]
    return y

# Common ======================================================================#

def normalize(array):
    """
    normalizes numpy array

    Parameters
    ----------
    array : numpy.ndarray
        array to be normalized
    """

    return array / np.max(np.abs(array))

def np_encode(array):
    """
    encodes numpy array to bytes

    Parameters
    ----------
    array : numpy.ndarray
        array to be encoded
    """

    return array.astype(np.float16).tobytes()

def np_decode(buf):
    """
    decodes numpy array from bytes

    Parameters
    ----------
    buf : bytearray
        bytes to be decoded
    """

    return np.frombuffer(buf, dtype=np.float16).astype(np.float64)

def tcp_unpaker(sock):
    """
    iterates incoming msgpack data from TCP stream 

    Parameters
    ----------
    sock : socket
        TCP socket
    """

    global running

    # creates msgpack unpacker
    unpacker = msgpack.Unpacker()

    # feeds unpacker with incoming TCP data
    while running:
        try:
            data = sock.recv(1024)
        except StopIteration:
            break

        unpacker.feed(data)

        for msg in unpacker:
            yield msg

        sleep(0.001)

def is_legal_audio_file(audio_file):
    """
    check if file is in project subfolder and is an wav file

    Parameters
    ----------
    audio_file : str
        audio_file path
    """

    # check if file is in project subfolder
    rel = os.path.relpath(audio_file)
    if rel.startswith('..'):
        return False

    # check if file is an wav file
    if Path(audio_file).suffix != '.wav':
        return False

    return True

class CustomFormatter(logging.Formatter):
    """
        logging formatter that colors output

        https://stackoverflow.com/questions/384076/
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(levelname)-8s %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Main ========================================================================#

if __name__ == "__main__":

    # Setup logger ============================================================#

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    # Argument parser configuration ===========================================#

    parser = argparse.ArgumentParser(description='Mixed audio streaming')

    subparsers = parser.add_subparsers(dest='command')

    server = subparsers.add_parser('server', description= \
        'Starts program in server mode')

    server.add_argument('--host', '-a', dest='host',
        default='', type=str, metavar='<host>',
        help='server hostname or IPv4 address')

    server.add_argument('--port', '-p', dest='port',
        default=29618, type=int, metavar='<port>',
        help='TCP port to bind to')

    server.add_argument('--max_clients', '-c', dest='max_clients',
        default=3, type=int, metavar='<max clients>',
        help='simultaneous clients')

    client = subparsers.add_parser('client', description= \
        'Starts program in client mode')

    client.add_argument('--host', '-a', dest='host',
        default='', type=str, metavar='<host>',
        help='server hostname or IPv4 address')

    client.add_argument('--port', '-p', dest='port',
        default=29618, type=int, metavar='<port>',
        help='server TCP port')

    client.add_argument('--audio_file', '-f', dest='audio_file',
        default='sample.wav', type=str, metavar='<remote audio file>',
        help='relative location of file in remote server')

    client.add_argument('--part_size', '-s', dest='part_size',
        default='500', type=int, metavar='<part size>')

    client.add_argument('--parts_in_window', '-w', dest='parts_in_window',
        default='50', type=int, metavar='<parts in window>')

    client.add_argument('--mid_start', '-b', dest='mid_start',
        default='0', type=int, metavar='<mid start>')

    client.add_argument('--mid_end', '-e', dest='mid_end',
        default='10', type=int, metavar='<mid end>')

    # Parse arguments =========================================================#

    p = parser.parse_args()

    if p.command == 'server':
        server_main((p.host, p.port), p.max_clients)

    elif p.command == 'client':
        client_main((p.host, p.port), p.audio_file, p.part_size,
            p.parts_in_window, p.mid_start, p.mid_end)

    else:
        logging.error('Unrecognized command')
        exit(-1)