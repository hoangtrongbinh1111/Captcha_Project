import os
from socket import socket
import time
import threading
from threading import Thread
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
# from fastapi_sockeimtio import SocketManager
import socketio
import json
import pickle
import uvicorn
import asyncio
from train import train
from test import test
from inference import infer
from tool import *

SOCKET_BACKEND_URL = 'http://192.168.0.159:6790'
PORT = 5676

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncClient(logger=True, engineio_logger=True)

'''
    API 
'''


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to Core Detect URL Phising server1 API!"}

'''
    Server Fast API and SocketIO
'''


@app.on_event('startup')
async def startup():
    await sio.connect(SOCKET_BACKEND_URL)


@sio.event
async def connect():
    print('connection established')


async def start_training(data):
    async for response in train(data["config"], data["data_root"], data["train"], data["test"], data["num_epochs"], data["batch_size"], data["max_lr"], data["export"], data["checkpoint"]):
        await sio.emit(f'receive_training_process', json.dumps({
            "response": response,
            "labId": data["labId"],
            "trainId": data["trainId"]
        }))
        await sio.sleep(0.1)


async def start_testing(data):
    response = await test(data["config"], data["data_root"], data["test"], data["max_lr"], data["weight"])
    await sio.emit(f'receive_testing_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "testId": data["testId"]
    }))
    await sio.sleep(0.1)


async def start_infering(data):
    response = await infer(data["img"], data["config"], data["weight"])
    await sio.emit(f'receive_infering_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "inferId": data["inferId"]
    }))
    await sio.sleep(0.1)


async def start_comparing_phase_testing(data):
    responseComparing = await test(data["test_data_dir"], data["labId"], None, data["model_type"], data["sample_model_dir"])
    await sio.emit(f'receive_comparing_process_phase_testing', json.dumps({
        "responseComparing": responseComparing,
        "labId": data["labId"],
        "compareId": data["compareId"]
    }))
    await sio.sleep(0.1)


async def start_comparing_phase_infering(data):
    responseComparing = await infer(data["sample_path"], data["labId"], None, data["model_type"], data["sample_model_dir"])
    await sio.emit(f'receive_comparing_process_phase_infering', json.dumps({
        "responseComparing": responseComparing,
        "labId": data["labId"],
        "compareId": data["compareId"]
    }))
    await sio.sleep(0.1)


async def start_reviewing_dataset(data):
    # trainX, valX, trainY, valY, tokenizer, embedding_matrix = get_train_data(
    #     data["data_dir"], data["val_size"])
    # model_dir = f'./modelDir/{data["labId"]}/log_train/{data["model_type"]}'

    # # Get tokenizer from file
    # tokenizer_file = open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb')
    # tokenizer = pickle.load(tokenizer_file)
    # tokenizer_file.close()
    # testX, testY = get_test_data(data["test_data_dir"], tokenizer)

    # await sio.emit(f'receive_reviewing_dataset_process', json.dumps({
    #     "response": {
    #         "numTrain": trainX.shape[0],
    #         "numVal": valX.shape[0],
    #         "numTest": testX.shape[0],
    #     },
    #     "labId": data["labId"],
    #     "datasetId": data["datasetId"],
    # }))
    await sio.sleep(0.1)


@sio.on("start_training")
async def start_training_listener(data):
    Thread(target=await start_training(data)).start()
    # sio.start_background_task(start_training, data)
    # asyncio.create_task(start_training(data))


@sio.on("start_testing")
async def start_testing_listener(data):
    Thread(target=await start_testing(data)).start()


@sio.on("start_infering")
async def start_infering_listener(data):
    Thread(target=await start_infering(data)).start()


@sio.on("start_reviewing_dataset")
async def start_reviewing_dataset_listener(data):
    Thread(target=await start_reviewing_dataset(data)).start()


@sio.on("start_comparing_phase_testing")
async def start_comparing_listener(data):
    Thread(target=await start_comparing_phase_testing(data)).start()


@sio.on("start_comparing_phase_infering")
async def start_comparing_infering_listener(data):
    Thread(target=await start_comparing_phase_infering(data)).start()


@sio.event
async def disconnect():
    print('disconnected from server')

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True,
                debug=True, ws_ping_interval=99999999, ws_ping_timeout=99999999)
