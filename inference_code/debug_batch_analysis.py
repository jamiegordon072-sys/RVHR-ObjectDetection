import socket
import json

HOST = "127.0.0.1"
PORT = 65432  # Match with server

data = {
    "Command":"ANALYSE",
    "Image Paths":[
        "C:\\Workspace\\Rail Tech\\RV-HR\\Data\\Test_Microcorrugation_Removal\\250304-023503_M3_T1_LeftRail\\M3_T1_021-00009_20250304_00158_103,854km.jpg",
        "C:\\Workspace\\Rail Tech\\RV-HR\\Data\\Test_Microcorrugation_Removal\\250304-023503_M3_T1_LeftRail\\M3_T1_021-00009_20250304_00186_103,910km.jpg"
        ],
        "Database Path":"C:\\Workspace\\Rail Tech\\RV-HR\\Data\\Test_Microcorrugation_Removal\\250304-023503_M3_T1.db",
        "Model Path":"C:\\Workspace\\Rail Tech\\RV-HR\\Code\\RVHR 1.0.10\\Analysis\\best.pt"
        }

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

message = json.dumps(data).encode("utf-8")
client.sendall(message)

response = client.recv(1024).decode("utf-8")
print("📥 Response from server:", response)
client.close()