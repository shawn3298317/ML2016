import numbers
import argparse

def read_n_sort(fn, fn_out, cols):

	lines = []
	with open(fn, 'r') as f:
		for line in f:
			l = sorted([float(x) for x in line.split()])
			l = [str(x) for x in l]
			lines.append(", ".join(l))

	# return lines

	with open(fn_out, 'w') as f:
		f.write(lines[cols])

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('cols', type = int, help='input cols number')
	parser.add_argument('O', type = str, help='output text name')
	args = parser.parse_args()
	
	read_n_sort("prob1/hw0_data.dat", args.O, args.cols)
	# cmd = raw_input("\n>>Please enter column number:")
	# while(isinstance(int(cmd), numbers.Integral)):
	# 	if int(cmd) >= len(lines):
	# 		print "Out of range!"
	# 	else:
	# 		print lines[int(cmd)]
	# 	cmd = raw_input("\n>>Please enter column number:")