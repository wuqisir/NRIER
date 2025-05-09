import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'ml_100k')
parser.add_argument('--model',
	type = str,
	help = 'model used for training. options: GMF, MLP, NeuMF-end',
	default = 'NeuMF-end')
parser.add_argument('--drop_rate',
	type = float,
	help = 'drop rate',
	default = 0.1)
parser.add_argument('--num_gradual',
	type = int,
	default =2000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--alpha',
	type = float,
	default = 0.2,
	help='hyperparameter in loss function')
parser.add_argument('--exponent',
	type = float,
	default = 1,
	help='exponent of the drop rate {0.5, 1, 2}')
parser.add_argument("--lr",
	type=float,
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout",
	type=float,
	default=0.0,
	help="dropout rate")
parser.add_argument("--batch_size",
	type=int,
	default=2048,
	help="batch size for training")
parser.add_argument("--epochs",
	type=int,
	default=150,
	help="training epoches")
parser.add_argument("--eval_freq",
	type=int,
	default=2000,
	help="the freq of eval")
parser.add_argument("--top_k",
	type=list,
	default=[5, 20],
	help="compute metrics@top_k")
parser.add_argument("--factor_num",
	type=int,
	default=32,
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
	type=int,
	default=[64,128,64,32],
	help="number of layers in MLP model")
parser.add_argument("--num_ng",
	type=int,
	default=1,
	help="sample negative items for training")
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--gpu",
	type=str,
	default="1",
	help="gpu card ID")
parser.add_argument('--temp_rate',
					nargs='?',
					default=[0.05, 0.1, 0.2],
					help="temperature 𝜏")

args = parser.parse_args()