from .conditional_gan_model import ConditionalGAN

def create_model(opt):
	model = None
	if opt.model == 'test':
		assert (opt.dataset_mode == 'single')
		from .test_model import TestModel
		model = TestModel()
	else:
		model = ConditionalGAN()
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
