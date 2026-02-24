import bpy
import socket
import json
import time
import io
import traceback
from contextlib import redirect_stdout

# Configuration
HOST = '127.0.0.1'
PORT = 9876

class F1BlenderWorker:
    def __init__(self):
        self.host = HOST
        self.port = PORT
        
    def get_scene_info(self):
        """Returns scene data including AABB (Dimensions)."""
        objs = []
        for obj in bpy.data.objects:
            # We round to 4 decimal places to keep the JSON small but precise (0.1mm)
            info = {
                "name": obj.name,
                "type": obj.type,
                "location": [round(obj.location.x, 4),
                             round(obj.location.y, 4),
                             round(obj.location.z, 4)],
                "dimensions": [round(obj.dimensions.x, 4),
                               round(obj.dimensions.y, 4),
                               round(obj.dimensions.z, 4)],
                "rotation_deg": [round(rad * 57.2958, 2) for rad in obj.rotation_euler] # Convert to degrees for LLM
            }
            objs.append(info)
        return objs

    def execute_code(self, code):
        """Directly executes the code and captures output."""
        namespace = {"bpy": bpy}
        capture_buffer = io.StringIO()
        try:
            with redirect_stdout(capture_buffer):
                exec(code, namespace)
            return {"executed": True, "result": capture_buffer.getvalue()}
        except Exception as e:
            return {"executed": False, "error": str(e)}

    def handle_request(self, data):
        try:
            request = json.loads(data.decode('utf-8'))
            cmd_type = request.get("type")
            params = request.get("params", {})
            
            if cmd_type == "get_scene_info":
                return {"status": "success", "result": self.get_scene_info()}
            elif cmd_type == "execute_code":
                return {"status": "success", "result": self.execute_code(params.get("code", ""))}
            else:
                return {"status": "error", "message": f"Unknown command: {cmd_type}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

def run_worker():
    worker = F1BlenderWorker()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    s.setblocking(False)
    
    print(f"F1-Blender Hybrid Worker listening on {PORT}...")
    
    while True:
        try:
            conn, addr = s.accept()
            conn.settimeout(10.0)
            data = conn.recv(8192)
            if data:
                response = worker.handle_request(data)
                conn.sendall(json.dumps(response).encode('utf-8'))
            conn.close()
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Worker Error: {e}")
        
        time.sleep(0.05) # High frequency for snappy AI responses

if __name__ == "__main__":
    run_worker()
