import os
import string
import random
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor

import yaml
import torch
from torch import Tensor
import librosa
import numpy as np
from model import RawNet

# HOST = "192.168.1.9"
# HOST = "192.168.1.32"
HOST = "192.168.185.163"
PORT = 9999
# MODEL_PATH = "/home/huytq/study/DATN/codes/2021/LA/Baseline-RawNet2/models/model_DF_CCE_100_32_0.0001_flac/epoch_1.pth"
# MODEL_PATH = "/home/huytq/study/DATN/codes/2021/LA/Baseline-RawNet2/models/model_DF_CCE_100_32_0.0001/epoch_6.pth"
# MODEL_PATH = "/home/huytq/study/DATN/pre-trained/pre_trained_DF_RawNet2.pth"
MODEL_PATH = "./epoch_1.pth"
CLOSE_CONN_CODE = b"<DNE>"
SINGLE_FILE_CODE = b"<SGL>"
MANY_FILES_CODE = b"<MNY>"
END_FILE_CODE = b"<END>"

# init server
def init(host, port):
    print("initializing...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    return server

# proceed classification requests from client
def proceed_request(conn, addr, model, device):
    while True:
        # check if client continues sending file
        cmd = conn.recv(5)
        print(cmd.decode(), addr)
        if cmd != CLOSE_CONN_CODE:
            # create audio file
            file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            file = open('./audio/' + file_name + '.m4a', "wb")
            file_bytes = b""

            # receive file's data
            done = False
            while not done:
                if file_bytes[-5:] == END_FILE_CODE:
                    done = True
                else:
                    data = conn.recv(1024)
                    file_bytes += data

            file.write(file_bytes[:-5])
            print(file_bytes[-5:].decode(), addr)
            file.close()

            # convert m4a to wav
            # try:
            #     subprocess.check_call(["ffmpeg", "-loglevel", "warning", "-i", 
            #                            "./audio/{}.m4a".format(file_name), 
            #                            "-ar", "16000", 
            #                            "./audio/{}.wav".format(file_name)])
            # except subprocess.CalledProcessError as e:
            #     print(e.output)
            #     conn.send("1".encode())
            #     continue

            # convert m4a to flac
            try:
                subprocess.check_call(["ffmpeg", "-loglevel", "warning", "-i", 
                                       "./audio/{}.m4a".format(file_name), 
                                       "-sample_fmt", "s16", 
                                       "-ar", "16000", 
                                       "-ac", "1", 
                                       "./audio/{}.flac".format(file_name)])
            except subprocess.CalledProcessError as e:
                print(e.output)
                conn.send("1".encode())
                continue

            # predict and send result
            # predict(cmd, model, device, file_name + ".wav", conn)
            predict(cmd, model, device, file_name + ".flac", conn)

            # remove received files
            os.remove('./audio/' + file_name + '.m4a')
            os.remove('./audio/' + file_name + '.flac')
        else:
            # done sending, close connected socket
            conn.close()
            print("**closed**", addr)

def operate(server):

    model, device = init_model(MODEL_PATH)
    thread_pool = ThreadPoolExecutor(5)

    print("server is running...")
    while True:
        conn, addr = server.accept()
        print("**accepted**", addr)
        thread_pool.submit(proceed_request, conn, addr, model, device)

def init_model(model_path):
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader = yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RawNet(parser1['model'], device)
    model =(model).to(device)

    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()

    return model, device

# pad audio file
def pad(x, max_len = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_audio(file_name, offset = 0):
    cut = 64600 # take ~4 sec audio (64600 samples)
    X, fs = librosa.load('./audio/' + file_name, sr = 16000)
    X = X[offset:]
    print(f"Offset = {offset}; Length = {X.shape[0]}")
    if X.shape[0] < 32000: return None
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    return x_inp

def predict(mode, model, device, file_name, conn):

    result = 1
    offset = 0

    while True:
        audio = load_audio(file_name, offset)
        if audio is None: break
        audio = audio[None, :]
        print(audio.shape)
        # print(audio)
        audio = audio.to(device)
        output = model(audio)
        print(output)
        _, pred = output.max(dim=1)
        print(pred)
        pred_class = int(pred.item())
        if mode == SINGLE_FILE_CODE:
            conn.send(str(pred_class).encode())
        elif pred_class == 0:
            result = 0
        offset += 32000
    
    if mode == SINGLE_FILE_CODE:
        conn.send(str(2).encode())
    else:
        conn.send(str(result).encode())

if __name__ == "__main__":
    server = init(HOST, PORT)
    operate(server)
