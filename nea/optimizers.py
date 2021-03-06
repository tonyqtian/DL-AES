import keras.optimizers as opt

def get_optimizer(args):

# 	clipvalue = 0
# 	clipnorm = 10
# 	clipnorm = 0

	if args.algorithm == 'rmsprop':
# 		optimizer = opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
# 		optimizer = opt.RMSprop(lr=0.01)
		if args.learning_rate > 0:
			optimizer = opt.RMSprop(lr=args.learning_rate)
		else:
			optimizer = opt.RMSprop()
	elif args.algorithm == 'sgd':
# 		optimizer = opt.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clipnorm, clipvalue=clipvalue)
		if args.learning_rate > 0:
			optimizer = opt.SGD(lr=args.learning_rate)
		else:
			optimizer = opt.SGD()
	elif args.algorithm == 'adagrad':
# 		optimizer = opt.Adagrad(lr=0.01, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
		if args.learning_rate > 0:
			optimizer = opt.Adagrad(lr=args.learning_rate)
		else:
			optimizer = opt.Adagrad()
	elif args.algorithm == 'adadelta':
# 		optimizer = opt.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
		if args.learning_rate > 0:
			optimizer = opt.Adadelta(lr=args.learning_rate)
		else:
			optimizer = opt.Adadelta()
	elif args.algorithm == 'adam':
# 		optimizer = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
		if args.learning_rate > 0:
			optimizer = opt.Adam(lr=args.learning_rate)
		else:
			optimizer = opt.Adam()
	elif args.algorithm == 'adamax':
# 		optimizer = opt.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
		if args.learning_rate > 0:
			optimizer = opt.Adamax(lr=args.learning_rate)
		else:
			optimizer = opt.Adamax()
	
	return optimizer
