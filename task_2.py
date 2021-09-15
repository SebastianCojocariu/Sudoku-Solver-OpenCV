import utils
from utils import *

# Arguments available
def parse_args():
	parser = argparse.ArgumentParser(description='Task2')
	parser.add_argument('--image_path', type=str, default=None,
	                    help='Path to an image on which to apply task2 (absolute or relative path)')
	parser.add_argument('--save_path', type=str, default=None,
	                    help='Path where to save the output of the algorithm (absolute or relative path)')
	parser.add_argument('--path_dir', type=str, default="./dataset/images/jigsaw/",
	                    help='Path to the directory where we want to apply task2 (absolute or relative path)')
	parser.add_argument('--save_dir', type=str, default='./dataset/predictions/jigsaw/',
	                    help='Path where to save the directory where we want to apply task2 (absolute or relative path)')
	parser.add_argument('--no_file', type=str, default=None,
	                    help='Apply the algorithm on the image specified by this number, that is located on path_dir. The output is saved on save_dir location')
	parser.add_argument('--verbose', type=str, default='0',
	                    help='Print intermediate output from the algorithm. Choose 0/1')
	args = parser.parse_args()
	return args


# Returns the number of white pixels measured on a fixed portion between 2 bounding boxes 
def get_no_white_pixels_from_intersection_line(sudoku_image, bbox1, bbox2):
	mid1, w1, h1 = bbox1
	mid2, w2, h2 = bbox2

	m_w, m_h = (w1 + w2) // 2, (h1 + h2) // 2

	# Split the line between centers into 3 equal parts, by 2 points
	x1 = int(mid1[0] + 1 / 3 * (mid2[0] - mid1[0]))
	y1 = int(mid1[1] + 1 / 3 * (mid2[1] - mid1[1]))

	x2 = int(mid1[0] + 2 / 3 * (mid2[0] - mid1[0]))
	y2 = int(mid1[1] + 2 / 3 * (mid2[1] - mid1[1]))

	# Check the direction: We enlarge this line either on Ox or Oy to get a fair amount o pixels
	points = []
	if abs(x1 - x2) / m_w < abs(y1 - y2) / m_h:
		for dx in range(-m_w // 4, m_w // 4 + 1):
			rr, cc = line(x1 + dx, y1, x2 + dx, y2)
			for i in range(len(rr)):
				points.append((rr[i], cc[i]))
	else:
		for dy in range(-m_h // 4, m_h // 4 + 1):
			rr, cc = line(x1, y1 + dy, x2, y2 + dy)
			for i in range(len(rr)):
				points.append((rr[i], cc[i]))

	# calculate the number of pixels
	no_white_pixels = 0
	for (r, c) in points:
		if r >= 0 and r < sudoku_image.shape[0] and c >= 0 and c < sudoku_image.shape[1]:
			if sudoku_image[r][c] >= 200:
				no_white_pixels += 1

	return no_white_pixels


# Returns the sudoku structure based on the thicknes of the lines that compose it.
def get_sudoku_structure(sudoku_image, bboxes, no_cells=9):
	result = {}
	values_stored = []
	for i in range(no_cells):
		for j in range(no_cells):
			# check the neighbours and store the number of pixels between every 2 adjacent bounding boxes
			for (dx, dy) in ((-1, 0), (0, -1), (0, 1), (1, 0)):
				if i + dx >= 0 and i + dx < no_cells and j + dy >= 0 and j + dy < no_cells:
					res = get_no_white_pixels_from_intersection_line(sudoku_image=sudoku_image, bbox1=bboxes[i][j], bbox2=bboxes[i + dx][j + dy])
					result[(i, j), (i + dx, j + dy)] = res
					values_stored.append(res)

	# compute some metrics for the threshold
	mean = np.mean(values_stored)
	std = np.std(values_stored)
	
	# filter based on mean and std
	result = {k: result[k] for k in result if result[k] >= mean + std // 3}
	
	return result

# Function that compute the sudoku structure based on a Depth-First-Search approach
def dfs(i, j, number_matrix, sudoku_structure, visited, result, curr_area_no, no_cells):
	if not (i >= 0 and i < no_cells and j >= 0 and j < no_cells):
		return

	visited[(i, j)] = True
	result[i][j] = "{}{}".format(curr_area_no, number_matrix[i][j]) 
	
	# apply DFS for the neighbours, if possible
	for (dx, dy) in ((-1, 0), (0, -1), (0, 1), (1, 0)):
		if i + dx >= 0 and i + dx < no_cells and j + dy >= 0 and j + dy < no_cells and (i + dx, j + dy) not in visited and ((i, j), (i + dx, j + dy)) not in sudoku_structure:
			dfs(i + dx, j + dy, number_matrix, sudoku_structure, visited, result, curr_area_no, no_cells)


# The entire logic behind Task2. 
def task2(image_path, save_path=None, verbose=0):
	# read the image
	image = cv2.imread(image_path)
	
	# convert the image to black and white
	image_black_and_white = utils.transform_to_black_and_white(image=image, ksize=55, C=21, k_blur=11, k_cross=9)
	
	# get the sudoku's number matrix + grid structure images through a perpective transformation.
	sudoku_image_only_numbers, sudoku_grid_image = utils.split_lines_from_numbers(image_black_and_white=image_black_and_white, width=2000, height=2000)

	# split the image into equal bounding boxes
	bboxes = utils.split_into_bboxes(image=sudoku_image_only_numbers)
	
	# extract the numbers in a matrix form
	number_matrix = utils.extract_numbers_from_image(sudoku_image=sudoku_image_only_numbers, bboxes=bboxes, verbose=verbose)
	
	# extract the grid adjacency matrix
	sudoku_structure = get_sudoku_structure(sudoku_image=sudoku_grid_image, bboxes=bboxes, no_cells=9)

	# compute the regions
	visited, curr_area_no, no_cells = {}, 1, 9
	solution = [[""] * no_cells for _ in range(no_cells)]
	for i in range(no_cells):
		for j in range(no_cells):
			if (i, j) not in visited:
				dfs(i=i, j=j, number_matrix=number_matrix, sudoku_structure=sudoku_structure, visited=visited, result=solution, curr_area_no=curr_area_no, no_cells=no_cells)
				curr_area_no += 1
	
	if verbose:
		print("\n##### Sudoku Structure #####")
		for key in sudoku_structure:
			print(key)
		
		print("\n##### Number matrix #####")
		for x in number_matrix:
			print(x)

		print("\n##### Result matrix #####")
		for x in result:
			print(x)

	string_to_write_in_file = "\n".join(["".join(solution[row_idx]) for row_idx in range(len(solution))])	
	if save_path != None and save_path != "":
		with open(save_path, "w+") as f:
			f.write(string_to_write_in_file)
		print("The output was saved at location: {}!".format(save_path))

	# return the solution generated by the underlying algorithm
	return solution, string_to_write_in_file


if __name__ == "__main__":
	args = parse_args()
	verbose = ord(args.verbose) - ord('0')

	if args.image_path != None:
		try:
			task2(image_path=args.image_path, save_path=args.save_path, verbose=verbose)
		except:
			raise Exception("An exception occured during the execution of task2!")

	else:
		os.makedirs(args.save_dir, exist_ok=True)
		
		if args.no_file != None:
			try:
				image_path = os.path.join(args.path_dir, "{}.jpg".format(args.no_file))
				save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(args.no_file))
				print("Processing the image located at {}".format(image_path))
				task2(image_path=image_path,
				 	  verbose=verbose, 
				 	  save_path=save_path)
			except:
				raise Exception("An exception occured during the execution of task2 for the image located at: {}!".format(image_path))
		
		else:
			for no_file in range(1, 16):
				try:
					image_path = os.path.join(args.path_dir, "{}.jpg".format(no_file))
					save_path = os.path.join(args.save_dir, "{}_predicted.txt".format(no_file))
					print("Processing the image located at {}".format(image_path))
					task2(image_path=image_path,
				 	  	  verbose=verbose, 
				 	      save_path=save_path)
				except:
					raise Exception("An exception occured during the execution of task2 for the image located at: {}!".format(image_path))
	

