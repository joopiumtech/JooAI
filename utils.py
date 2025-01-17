import time

def stream_data(message: str):
    for word in message.split(" "):
        yield word + " "
        time.sleep(0.02)