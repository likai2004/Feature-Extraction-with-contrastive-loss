import torchvision.transforms as transforms

class para_cfg(object):
	def __init__(self):

		self.CUDA = True
		self.mGPUs = True

		self.OPTIM = 'Adam'
		self.MOMENTUM = 0.9
		self.LR = 5e-4
		self.WEIGHT_DECAY = 1e-6
		self.MAX_EPOCH = 5000
		self.BATCH_SIZE = 1 # query_size
		self.NUM_WORKERS = 1
		self.CLASSES = 10

		self.QUERY_CLASSES = 10
		self.QUERY_SIZE = self.BATCH_SIZE
		self.NEG_NUM = 3
		self.POS_NUM = 10
		self.NEG_NUM_FOR_LOSS = self.CLASSES * self.POS_NUM * 6 # 30:180=1:6

		self.LOSS_MARGIN = 0.7

		self.START_EPOCH = 0
		self.RESUME = True
		self.RESUME_PATH = './checkpoint/ckpt-50-1874.pth'
		self.CHECKPOINT_PATH = './checkpoint'

		self.TRAIN_transforms = transforms.Compose([
			transforms.Resize((227, 227)),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])
		self.USE_TFBOARD = True
		self.LOG_NAME = 'MNIST'

	def print_para(self):
		for i in sorted(self.__dict__):
			print ("{0}:{1}".format(i, self.__dict__[i]))