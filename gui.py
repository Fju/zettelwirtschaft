import opencv2 as cv2

class UserInterface(object):
	"""
	d = DatasetBuilder('labels', 'data')
	p = 0
	while(1):
		print(p)
		img = d.getBoxes(p)
		cv2.imshow('image', img)
		key = cv2.waitKey(0)
		if key == 1113939:
			p += 1
		elif key == 1113937:
			p -= 1
		elif key == 1048603:
			break
		
		max_p = len(d.data)
		p = (p + max_p) % max_p

	cv2.destroyAllWindows()
	"""
	def __init__(self, net, dataset):
		self.net = net
		self.dataset = dataset

	def show(self, image)
