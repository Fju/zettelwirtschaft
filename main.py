from tensorflow.python.client import device_lib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-name', default='default', help='foo help')
parser.add_argument('-ld', '--list-devices', action='store_true')
parser.add_argument('-c', '--use_checkpoint', action='store_true')

args = parser.parse_args()

print(args)
if args.list_devices:
	print(device_lib.list_local_devices())

def main():



if __name__ == '__main__':
	main()
