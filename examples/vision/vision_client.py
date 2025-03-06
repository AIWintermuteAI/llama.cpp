#!/usr/bin/env python3
import socket
import json
import sys
import os

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
        
    def initialize(self, model_path, n_gpu_layers=0):
        request = {
            "id": self.next_id(),
            "init": {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers
            }
        }
        return self._send_request(request)
        
    def infer(self, image_path, prompt=None, n_predict=64):
        if prompt is None:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img_placement>\nwhat do you see?<|im_end|>\n<|im_start|>assistant\n"
            
        request = {
            "id": self.next_id(),
            "infer": {
                "image_path": image_path,
                "prompt": prompt,
                "n_predict": n_predict
            }
        }
        return self._send_request(request)

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <socket_path> <model_path> <image_path>")
        sys.exit(1)
        
    socket_path = sys.argv[1]
    model_path = sys.argv[2]
    image_path = sys.argv[3]
    
    # Verify paths exist
    if not os.path.exists(socket_path):
        print(f"Error: Socket {socket_path} does not exist. Make sure the server is running.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        sys.exit(1)
    
    client = VisionClient(socket_path)
    
    # Initialize the model
    print("Initializing model...")
    response = client.initialize(model_path)
    if not response["success"]:
        print(f"Failed to initialize: {response['error']}")
        sys.exit(1)
    print("Model initialized successfully!")
    
    # Run inference
    print("\nRunning inference...")
    response = client.infer(image_path)
    if not response["success"]:
        print(f"Inference failed: {response['error']}")
        sys.exit(1)
        
    print("\nGenerated text:")
    print(response["result"]["text"])

if __name__ == "__main__":
    main() 