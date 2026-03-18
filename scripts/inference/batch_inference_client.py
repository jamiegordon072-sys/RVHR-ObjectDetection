"""
This Script can be used for debugging batch_infernece_server.py
Run batch_infernece_server (ensuring PORT = 65432) and wait for prompt "Waiting for connection from client..."
Then run batch_inference_client in parallel
"""

import socket
import json

HOST = "127.0.0.1"
PORT = 65432  # Match with server

data = {
    "Command":"ANALYSE",
    "Image Paths":[
        "C:\\Workspace\\Rail Tech\\RV-HR\\Data\\CPH 032026\\260303-025143_M3_T2_LeftRail\\M3_T2_021-00009_20260303_00007_200,670km.jpg",
        "C:\\Workspace\\Rail Tech\\RV-HR\\Data\\CPH 032026\\260303-025143_M3_T2_LeftRail\\M3_T2_021-00009_20260303_00014_200,684km.jpg"
        ],
        "Database Path":"C:\\Workspace\\Rail Tech\\RV-HR\\Data\\CPH 032026\\260303-025143_M3_T2.db",
        "Model Path":"C:\\Workspace\\Rail Tech\\RV-HR\\Code\\RVHR 1.0.10\\Analysis\\best.pt"
        }

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

message = json.dumps(data).encode("utf-8")
client.sendall(message)

response = client.recv(1024).decode("utf-8")
print("📥 Response from server:", response)
client.close()