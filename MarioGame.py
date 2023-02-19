import IPython
# from google.colab import output
from PIL import Image
import base64
import cv2
import numpy as np
import io

class MarioGame:
  def __init__(self, width=640, height=480, update_interval=500, callback=None, autoplay=False):
    self.width = width
    self.height = height
    self.update_interval = update_interval
    self.running = False
    self.current_frame = None
    self.callback = callback
    self.autoplay = autoplay

  def jsCode(self):
    return IPython.display.Javascript(
      """
        const frame = document.createElement("iframe");
        frame.width={width};
        frame.height={height};
        document.querySelector("#output-area").appendChild(frame);

        const frame_doc = frame.contentDocument;
        const frame_window = frame.contentWindow;

        const frame_script = frame_doc.createElement("script");
        const frame_style = document.createElement("style");
        frame_style.type = "text/css"
        frame_style.innerText = "#data_display, div.text {display:none;} body {margin:0px !important;}";

        frame_script.src = "https://supermarioplay.com/game/all.js?v=2.0.0.0";
        frame_script.onload = () => {
          frame_doc.head.appendChild(frame_style)
          frame_window.is_mobile=false; 
          frame_window.FullScreenMario();
          const canvas = frame_doc.querySelector("canvas");
          var n = 0;
          setInterval(
            () => { 
              const prefix = "data:image/png;base64,";
              image_data = canvas.toDataURL().substring(prefix.length); 
              google.colab.kernel.invokeFunction('updateFrame', [image_data, n++], {});

              if ({autoplay}) {
                frame_window.triggerKeyboardEvent(frame_doc.body, 68, "keydown"); // go rigth
              }
            }, 
            {update_interval}
          );
        }
        frame_doc.body.appendChild(frame_script);
      """
    )

  def runGame(self):
    if self.running:
      return
    self.running = True
    js_code = self.jsCode()
    js_code.data = js_code.data.replace(
        "{width}", str(self.width)
    ).replace(
        "{height}", str(self.height)
    ).replace(
        "{update_interval}", str(self.update_interval)
    ).replace(
        "{autoplay}", str(self.autoplay).lower()
    )
    display(js_code)
    output.register_callback("updateFrame", self.__updateFrame)

  def __updateFrame(self, new_frame, n):
    self.current_frame = new_frame
    if self.callback:
      self.callback(new_frame, n)