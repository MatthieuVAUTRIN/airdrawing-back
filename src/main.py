import base64

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket

from src.classes.airdrawing import AirDrawing

app = FastAPI()


air_drawing = AirDrawing()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            # convert base64 to numpy array
            frame_data = base64.b64decode(data["frame"])
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Shape: (H, W, 3)

            # process frame
            draw_color = list(data["draw_color"])
            erase_color = (0, 0, 255)
            processed_frame = air_drawing.process_frame(frame, draw_color, erase_color)

            # convert processed frame back to base64 (bettr for sending over websocket)
            _, buffer = cv2.imencode(".jpg", processed_frame)
            encoded_frame = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_json({"frame": encoded_frame})

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


@app.post("/clear-canvas")
async def clear_canvas():
    air_drawing.canvas = None
    return {"message": "Canvas cleared"}
