import zmq

def listen():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # Try 127.0.0.1 explicitly
    socket.connect("tcp://127.0.0.1:5555")
    socket.subscribe("")

    print("Listening on tcp://127.0.0.1:5555...")
    while True:
        try:
            topic, msg = socket.recv_multipart()
            print(f"RECEIVED: {topic.decode()} | {len(msg)} bytes")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)

if __name__ == "__main__":
    listen()