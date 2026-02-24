# blender_client.py
import socket
import json
import logging

logger = logging.getLogger("BlenderClient")

class BlenderMCPClient:
    def __init__(self, host="localhost", port=9876):
        self.host = host
        self.port = port

    def send_command(self, command_type: str, params: dict = None):
        """Formats and sends the JSON packet to the Blender worker."""
        payload = {
            "type": command_type,
            "params": params or {}
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(15)  # Blender might take a second to process complex mesh code
                s.connect((self.host, self.port))
                s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
                
                # Receive the response
                data = s.recv(1024 * 1024)
                if not data:
                    return {"status": "error", "message": "No response"}
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to communicate with Blender: {e}")
            return {"status": "error", "message": str(e)}
