from network import LoRa
import socket
import machine
import time

# initialise LoRa in LORA mode
# Please pick the region that matches where you are using the device:
# Asia = LoRa.AS923
# Australia = LoRa.AU915
# Europe = LoRa.EU868
# United States = LoRa.US915
# more params can also be given, like frequency, tx power and spreading factor
lora = LoRa(mode=LoRa.LORA, region=LoRa.EU868,frequency=868000000,bandwidth=LoRa.BW_125KHZ, sf=7, preamble=8,coding_rate=LoRa.CODING_4_5)

# create a raw LoRa socket
s = socket.socket(socket.AF_LORA, socket.SOCK_RAW)

t = 0
while t < 1000000:
    time.sleep(0.1)
    # send some data
    s.setblocking(True)
    #s.send('Hello')
    s.send(bytes([0x00])) #1 bytes
    print("s")

    # get any data received...
    #s.setblocking(False)
    #data = s.recv(64)
    #print(data)

    t = t+1
