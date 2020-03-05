import cv2
import pickle
import socket
import struct

TCP_IP = '127.0.0.1'
TCP_PORT = 9501
#video_file = 'some_file.MP4'

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # establishing a tcp connection
sock.bind((TCP_IP, TCP_PORT))
sock.listen(5)

while True:
	(client_socket, client_address) = sock.accept() # wait for client
	print ('connection established with ' +str(client_address))
	cap = cv2.VideoCapture(0)
	#pos_frame = cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
	pos_frame = cap.get(1)
	while True:
		flag, frame = cap.read()
		if flag:
			frame = pickle.dumps(frame)
			size = len(frame)
			p = struct.pack('I', size)
			frame = p + frame
			client_socket.sendall(frame)
		else:
			cap.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
	  
		#if cap.get(cv2.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.CV_CAP_PROP_FRAME_COUNT):
		if cap.get(1) == cap.get(7):
			size = 10
			p = struct.pack("I", size)
			client_socket.send(p)
			client_socket.send('')
			break