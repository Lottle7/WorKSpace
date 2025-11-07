import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--city', default="NY", help="city name, can be Chi or NY")
parser.add_argument('--task', default="crime", help="task name, can be crime or checkIn or serviceCall")
parser.add_argument('--text_output_dim',  type=int,   default=256)
parser.add_argument('--lr',  type=float,  default=0.08)
parser.add_argument('--epochs',         type=int,    default=2000)
parser.add_argument('--model',     type=str,    default="llama")
parser.add_argument('--device',     type=str,    default="0")

args = parser.parse_args()