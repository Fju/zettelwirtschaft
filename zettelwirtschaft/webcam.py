import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

width, height = int(cap.get(3)), int(cap.get(4))


box_size = 256


while True:	
	ret, frame = cap.read()
	
	# drawing box
	cv2.rectangle(frame, ((width - box_size) / 2, (height - box_size) / 2), ((width + box_size) / 2, (height + box_size) / 2), (255, 0, 0))

	cv2.imshow('frame', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
