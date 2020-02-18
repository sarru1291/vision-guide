import imagezmq
image_hub = imagezmq.ImageHub()
msg, image_np = image_hub.recv_image()
print(msg)