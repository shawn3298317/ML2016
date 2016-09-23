import cv2
import argparse

def process_img(fn, fn_out):

	img = cv2.imread(fn)
	img = cv2.flip(img, -1)

	# cv2.imshow("img",img)
	# cv2.waitKey()

	cv2.imwrite(fn_out, img)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('I', type = str, help='input image file name')
	parser.add_argument('O', type = str, help='output image file name')
	args = parser.parse_args()
	# process_img(str(args.I),str(args.O))
	process_img(args.I,args.O)