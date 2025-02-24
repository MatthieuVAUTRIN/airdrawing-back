import cv2
import numpy as np
from fastapi import FastAPI, WebSocket

from src.classes.airdrawing import AirDrawing
from src.models.models import ColorData

app = FastAPI()


air_drawing = AirDrawing()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()

            image_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # shape: (H, W, 3)

            # process frame
            erase_color = (0, 0, 255)
            processed_frame = air_drawing.process_frame(frame, erase_color)

            # encode frame to jpg for lower resolution
            _, buffer = cv2.imencode(
                ".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            )

            await websocket.send_bytes(buffer.tobytes())

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


@app.post("/clear-canvas")
async def clear_canvas():
    air_drawing.canvas = None
    return {"message": "Canvas cleared"}


@app.post("/change-color")
async def change_color(color_data: ColorData):
    air_drawing.draw_color = color_data.color
    return {"message": "Color changed"}
