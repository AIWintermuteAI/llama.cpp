#!/usr/bin/env python3
import socket
import json
import sys
import os
import argparse
import cv2

class VisionClient:
    def __init__(self, socket_path):
        self.socket_path = socket_path
        self.request_id = 0

    def _connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        return sock

    def _send_request(self, request):
        sock = self._connect()
        try:
            # Send request
            sock.sendall(json.dumps(request).encode('utf-8'))

            # Receive response
            response = sock.recv(65536).decode('utf-8')
            return json.loads(response)
        finally:
            sock.close()

    def next_id(self):
        self.request_id += 1
        return self.request_id

    def initialize(self):
        request = {
            "id": self.next_id(),
            "init": {
                "model_path": "dummy",
            }
        }
        return self._send_request(request)

    def clear_kv_cache(self):
        request = {
            "id": self.next_id(),
            "clear_kv_cache": {}
        }
        return self._send_request(request)

    def infer(self, image_path, prompt=None, n_predict=64):
        if prompt is None:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img_placement>\nDescribe the image in one short sentence.<|im_end|>\n<|im_start|>assistant\n"

        request = {
            "id": self.next_id(),
            "infer": {
                "image_path": image_path,
                "prompt": prompt,
                "n_predict": n_predict
            }
        }
        return self._send_request(request)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Client")
    parser.add_argument("--socket", type=str, default = "/tmp/vision.sock", help="Path to the UNIX socket")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument("--prompt", type=str, help="Prompt for the model")
    parser.add_argument("--n_predict", type=int, default=64, help="Number of tokens to predict")
    parser.add_argument("--use_tts", action="store_true", help="Use piper-tts to speak the generated text")
    args = parser.parse_args()

    socket_path = args.socket
    image_path = args.image_path

    # Verify paths exist
    if not os.path.exists(socket_path):
        print(f"Error: Socket {socket_path} does not exist. Make sure the server is running.")
        sys.exit(1)
    if image_path:
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist.")
            sys.exit(1)

    if args.use_tts:
        print("Initializing TTS...")
        # Download the model if it does not exist
        tts_model_path = "~/.config/piper-tts/en_US-lessac-medium.onnx"
        if not os.path.exists(os.path.expanduser(tts_model_path)):
            print("Downloading the TTS model...")
            os.system("echo 'Init' | piper --model en_US-lessac-medium --output_file .tmp.wav --data-dir ~/.config/piper-tts --download-dir ~/.config/piper-tts")

    client = VisionClient(socket_path)

    # Initialize the model
    print("Initializing model...")
    response = client.initialize()
    if not response["success"]:
        print(f"Failed to initialize: {response['error']}")
        sys.exit(1)
    print("Model initialized successfully!")
    continuos = False

    if not image_path:
        cap = cv2.VideoCapture(0)
        image_path = ".image.png"
        continuos = True

    while True:
        if continuos:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Cam read error")
                    break
                cv2.imshow('Preview', frame)
                key = cv2.waitKey(1)
                if key == ord('c'):
                    cv2.imwrite(image_path, frame)
                    print(f"Image captured and saved to {image_path}")
                    break
                elif key == ord('q'):
                    print("Quitting...")
                    cv2.destroyAllWindows()
                    cap.release()
                    sys.exit(0)

        # Run inference
        print("\nRunning inference...")
        image_abspath= os.path.abspath(image_path)
        respose = client.clear_kv_cache()
        if not response["success"]:
            print(f"Failed to clear KV cache: {response['error']}")
            sys.exit(1)
        response = client.infer(image_abspath, args.prompt, args.n_predict)
        if not response["success"]:
            print(f"Inference failed: {response['error']}")
            sys.exit(1)

        print("\nGenerated text:")
        print(response["result"]["text"])
        if args.use_tts:
            print("Speaking the generated text...")
            # use piper-tts to speak the generated text
            os.system(f"echo \"{response['result']['text']}\" | piper --model {tts_model_path} --output_file .tmp.wav")
            # check the os and play the audio
            if sys.platform == 'linux':
                os.system("aplay .tmp.wav")
            elif sys.platform == 'darwin':
                os.system("afplay .tmp.wav")
            else:
                print("Unsupported OS")
            os.remove(".tmp.wav")

        if not continuos:
            break

    if continuos:
        cap.release()
        os.remove(image_path)