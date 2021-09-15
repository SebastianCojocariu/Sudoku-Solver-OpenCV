import utils
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Arguments available
def parse_args():
	parser = argparse.ArgumentParser(description='Task3')
	parser.add_argument('--image_path', type=str, default=None,
	                    help='Path to an image on which to apply task3 (absolute or relative path)')
	parser.add_argument('--template_own_path', type=str, default="./own_template.jpg",
	                    help='Path to the own template (absolute or relative path)')
	parser.add_argument('--template_given_path', type=str, default="./dataset/images/cube/template.jpg",
	                    help='Path to the given template (absolute or relative path)')
	parser.add_argument('--save_path_text', type=str, default=None,
	                    help='Path where to save the text output of the algorithm (absolute or relative path)')
	parser.add_argument('--save_path_image', type=str, default=None,
	                    help='Path where to save the image output of the algorithm (absolute or relative path)')
	parser.add_argument('--path_dir', type=str, default="./dataset/images/cube/",
	                    help='Path to the directory where we want to apply task3 (absolute or relative path)')
	parser.add_argument('--save_dir', type=str, default='./dataset/predictions/cube/',
	                    help='Path where to save the directory where we want to apply task3 (absolute or relative path)')
	parser.add_argument('--no_file', type=str, default=None,
	                    help='Apply the algorithm on the image specified by this number, that is located on path_dir. The output is saved on save_dir location')
	parser.add_argument('--verbose', type=str, default='0',
	                    help='Print intermediate output from the algorithm. Choose 0/1')
	args = parser.parse_args()
	return args


# Run a MNIST model against each digit (after centering the same digits). A heuristic approach is used based on the confidence of the model
# (several candidates per digit are sampled based on that).
def apply_model(model_path, image, no_cells=9, verbose=0):
	def show_matrix(matrix):
		for i in range(len(matrix)):
			print(matrix[i])
		print()
	# split the image with numbers into bounding boxes
	bboxes = utils.split_into_bboxes(image=image, no_cells=no_cells)
	# load the NN Model trained on MNIST
	onnx_model = prepare(onnx.load(model_path))

	possible_matrix = [[0] * no_cells for _ in range(no_cells)]
	images = []

	# loop over all the digits and predict a number (or more based on the model's confidence)
	for i in range(no_cells):
		for j in range(no_cells):
			# select the current image
			(row_left, row_right), (col_left, col_right) = utils.convert_bbox(image=image, bbox=bboxes[i][j])
			curr_image = image[row_left: row_right, col_left: col_right]

			curr_image = cv2.resize(curr_image, (28,28))
			curr_image = utils.center_digit(image=curr_image)
			
			images.append(curr_image)
			
			curr_image = curr_image.astype(np.float32) / 255
			
			# Resize the image dimension to fulfill model's needs (28 * 28)
			curr_image = np.reshape(curr_image, (1, 1, 28, 28))

			# Get the predictions (for digits from 0 to 9 inclusively)
			output = onnx_model.run(curr_image)
			output = np.squeeze(output)

			# Get probabilities
			probabilities = special.softmax(output) # remove the prediction for 0 from equation

			# Get the indexes with maximum probabilities
			argmax_indeces = probabilities.argsort()[::-1]
			
			# If the model predicted 0 to be the most probable (0 cannot be for sure as sudoku doesnt contain 0's)
			# Append the following 4 digits as candidates
			if argmax_indeces[0] == 0:
				possible_matrix[i][j] = argmax_indeces[1:5]
			else:
				# If the model predicted with a higher confidence than 95%, than we are pretty sure about the result and save it
				if probabilities[argmax_indeces[0]] >= 0.95:
					possible_matrix[i][j] = [argmax_indeces[0]]
				
				# if it is not the case, but is greater than 75% for the first prediction, we append the first 3 digit predictions (removing 0 if necessary)
				elif probabilities[argmax_indeces[0]] >= 0.75:
					possible_matrix[i][j] = [x for x in argmax_indeces if x != 0][:3]
				# for the reamaining cases, we append the first 4 digit predictions (removing 0 if necessary)
				else:
					possible_matrix[i][j] = [x for x in argmax_indeces if x != 0][:4]

	correct_sudokus = []

	# trim the sudoku possibilities to make backtracking as lightweighting as possible
	corrected_possible_matrix = trim_sudoku_possibilities(possible_matrix=copy.deepcopy(possible_matrix))
	if corrected_possible_matrix is None:
		show_matrix(possible_matrix)
		raise Exception("Cannot correct errors.")
	
	possible_matrix = corrected_possible_matrix
	
	# backtrack to find the desired solution
	backtracking_sudoku(i=0, j=0, current_matrix=np.uint8(np.zeros((no_cells, no_cells))), possible_matrix=possible_matrix, result=correct_sudokus)
	
	if verbose:
		print("#### Possible solutions ####\n")
		show_matrix(possible_matrix)
		print("##############################\n")
		for i, sudoku in enumerate(correct_sudokus):
			print("#### {} Solution ####\n".format(i))
			show_matrix(sudoku)
			print("##############################\n")

		utils.show_images(images_list=images, nrows=no_cells, ncols=no_cells)	
	
	# return the sudoku matrix
	final_sudoku = correct_sudokus[0].tolist()
	final_sudoku = [[str(element) for element in line] for line in final_sudoku]
	return final_sudoku

# Enforce the conditions for sudoku
def check_sudoku_solver(matrix, no_cells=9):
	matrix = np.uint8(np.asarray(matrix))
	
	# horizontal checking
	for i in range(no_cells):
		visited = {}
		for j in range(no_cells):
			if matrix[i][j] in visited:
				return False
			visited[matrix[i][j]] = True

	# vertical checking
	for j in range(no_cells):
		visited = {}
		for i in range(no_cells):
			if matrix[i][j] in visited:
				return False
			visited[matrix[i][j]] = True

	# box checking
	step = int(math.sqrt(no_cells))
	for i in range(step):
		for j in range(step):
			x, y = i * step, j * step
			visited = {}
			for elem in matrix[x: x + step, y: y + step].flatten():
				if elem in visited:
					return False
				visited[elem] = True

	return True


# Check second condition (the line condition adjiaceny from the requirement)
def check_second_condition(matrices, no_cells=9):
	# matrices[0] -> matrices[1]
	for i in range(no_cells):
		if matrices[0][no_cells - 1][i] != matrices[1][0][i]:
			return False

	# matrices[0] -> matrics[2]
	for i in range(no_cells):
		if matrices[0][i][no_cells - 1] != matrices[2][0][no_cells - i - 1]:
			return False

	# matrices[1] -> matrics[2]
	for i in range(no_cells):
		if matrices[1][i][no_cells - 1] != matrices[2][i][0]:
			return False

	return True


# the first trim algorithm before starting the backtracking to reduce the complexity
def trim_sudoku_possibilities(possible_matrix):
	possible_matrix = copy.deepcopy(possible_matrix)
	no_cells = len(possible_matrix)

	# horizontal trim
	for i in range(no_cells):
		fixed = {}
		for j in range(no_cells):
			if len(possible_matrix[i][j]) == 1:
				if possible_matrix[i][j][0] in fixed:
					return None
				fixed[possible_matrix[i][j][0]] = True
		for j in range(no_cells):
			if len(possible_matrix[i][j]) > 1:
				possible_matrix[i][j] = [x for x in possible_matrix[i][j] if x not in fixed]
				if len(possible_matrix[i][j]) == 0:
					return None

	# vertical trim
	for j in range(no_cells):
		fixed = {}
		for i in range(no_cells):
			if len(possible_matrix[i][j]) == 1:
				if possible_matrix[i][j][0] in fixed:
					return None
				fixed[possible_matrix[i][j][0]] = True
		for i in range(no_cells):
			if len(possible_matrix[i][j]) > 1:
				possible_matrix[i][j] = [x for x in possible_matrix[i][j] if x not in fixed]
				if len(possible_matrix[i][j]) == 0:
					return None

	# box checking trim
	step = int(math.sqrt(no_cells))
	for i in range(step):
		for j in range(step):
			x, y = i * step, j * step
			fixed = {}
			for k_x in range(step):
				for k_y in range(step):
					if len(possible_matrix[x + k_x][y + k_y]) == 1:
						if possible_matrix[x + k_x][y + k_y][0] in fixed:
							return None
						fixed[possible_matrix[x + k_x][y + k_y][0]] = True
			for k_x in range(step):
				for k_y in range(step):
					if len(possible_matrix[x + k_x][y + k_y]) != 1:
						possible_matrix[x + k_x][y + k_y] = [x for x in possible_matrix[x + k_x][y + k_y] if x not in fixed]
						if len(possible_matrix[x + k_x][y + k_y]) == 0:
							return None

	return possible_matrix


# Backtracking based on possible candidates per digit
def backtracking_sudoku(i, j, current_matrix, possible_matrix, result):
	no_cells = len(possible_matrix)
	
	if not (i >= 0 and i < no_cells and j >= 0 and j < no_cells):
		if check_sudoku_solver(current_matrix, no_cells=no_cells):
			result.append(np.uint8(current_matrix.copy()))
	else:
		for possible_elem in possible_matrix[i][j]:
			current_matrix[i][j] = possible_elem
			if i % 2 == 0:
				if j == no_cells - 1:
					backtracking_sudoku(i + 1, j, current_matrix, possible_matrix, result)
				else:
					backtracking_sudoku(i, j + 1, current_matrix, possible_matrix, result)
			
			else:
				if j == 0:
					backtracking_sudoku(i + 1, j, current_matrix, possible_matrix, result)
				else:
					backtracking_sudoku(i, j - 1, current_matrix, possible_matrix, result)


# Converts image to black and white
def convert_to_black(image):
	image = image.copy()
	# Add blur
	image = cv2.GaussianBlur(image, (3, 3), 0)
	# Convert from RGB to gray
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Convert to black and white image
	image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)
	
	return image


# Get the largest component based on flood fill.
def get_largest_component_task3(image):
	image = image.copy()
	
	# start floodfilling for each component and update the maximum component found so far
	# for this part we use a value different from 0 and 255 (64 was chosen for simplicity)
	flood_fill_image = np.float32(image.copy())
	max_area, contour_sudoku = 0, None
	for i in range(flood_fill_image.shape[0]):
		for j in range(flood_fill_image.shape[1]):
			if flood_fill_image[i][j] >= 200:
				(area, _, _, _) = cv2.floodFill(flood_fill_image, None, seedPoint=(j, i), newVal=64)
				if area > max_area:
					seedPoint = (j, i)
					max_area = area

	# floodfill again the largest contour with white
	cv2.floodFill(flood_fill_image, None, seedPoint=seedPoint, newVal=255)
	
	# make all the remaining components black through floodfilling.
	for i in range(flood_fill_image.shape[0]):
		for j in range(flood_fill_image.shape[1]):
			if flood_fill_image[i][j] == 64 and (j, i) != seedPoint:
				flood_fill_image[i][j] = 0
	
	return np.uint8(flood_fill_image)


# Takes a black and white image, find the largest contour using floodfill (from get_largest_component)
# Then, it detects the sudoku contour based on the outer_grid determined.  Next, we will remove the outer
# grid and we will map the countour twice: one time for the numbers inside the grid, and one time for the
# grid lines (while applying a perspective transformation: see transform_perspective function)
def split_lines_from_numbers(image_black_and_white, width=450, height=450):
	outer_grid_image = get_largest_component_task3(image=image_black_and_white)
	sudoku_contour_points = utils.detect_sudoku_contour(image=outer_grid_image)

	new_sudoku_image = utils.remove_image_from_image(remove_image=outer_grid_image, from_image=image_black_and_white)

	numbers_image = utils.transform_perspective(image=new_sudoku_image, contour=sudoku_contour_points, width=width, heigth=height)
	grid_image = utils.transform_perspective(image=outer_grid_image, contour=sudoku_contour_points, width=width, heigth=height)

	return numbers_image, grid_image, outer_grid_image, sudoku_contour_points


# Function that return the 3 sudokus + their contours on the original image
def get_all_three_sudoku(image_path, width, height):
	image = cv2.imread(image_path)
	
	image = convert_to_black(image)

	sudoku_image_only_numbers1, sudoku_grid_image1, only_outer_grid1, sudoku_contour_points1 = split_lines_from_numbers(image, width, height)
		
	# remove the outer grid for the first sudoku
	image = utils.remove_image_from_image(remove_image=only_outer_grid1, from_image=image)
	sudoku_image_only_numbers2, sudoku_grid_image2, only_outer_grid2, sudoku_contour_points2 = split_lines_from_numbers(image, width, height)
	
	# remove the outer grid for the second sudoku
	image = utils.remove_image_from_image(remove_image=only_outer_grid2, from_image=image)
	sudoku_image_only_numbers3, sudoku_grid_image3, only_outer_grid3, sudoku_contour_points3 = split_lines_from_numbers(image, width, height)

	return (sudoku_image_only_numbers1, sudoku_contour_points1), (sudoku_image_only_numbers2, sudoku_contour_points2), (sudoku_image_only_numbers3, sudoku_contour_points3)


# Maps an image's contour onto a bbox from a template_image
def transform_perspective_onto_cube_face(image, template_image, contour, bbox):
	assert contour is not None, "contour passed is None"
	image, contour = image.copy(), contour.copy()

	contour = np.float32(contour)
	contour = np.squeeze(contour)
	
	# make the background black
	aux = np.zeros(image.shape).astype(image.dtype)
	cv2.fillPoly(aux, np.int32([contour]), color=(255, 255, 255))
	image = cv2.bitwise_and(image, aux)

	image = np.float32(image)
	M, _ = cv2.findHomography(contour, bbox, method=0)
	image = cv2.warpPerspective(image, M, (template_image.shape[1], template_image.shape[0]))
	
	return np.uint8(image)


# Maps an image as in the testing onto an own cube template (this will later be mapped on the desired template automatically)
def construct_cube_on_own_template(initial_image_path, template_image_path, contour_list, verbose=0):
	initial_image = cv2.imread(initial_image_path)
	template_image = cv2.imread(template_image_path)

	# own_template_coordinates
	bbox1 = np.array([[275, 4], [562, 87], [295, 238], [5, 161]], dtype = "float32")
	bbox2 = np.array([[5, 161], [295, 238], [296, 542], [6, 465]], dtype = "float32")
	bbox3 = np.array([[295, 238], [562, 87], [565, 387], [297, 542]], dtype = "float32")

	bbox_mapping = [bbox1, bbox2, bbox3]
	images = []
	for i in range(len(contour_list)):
		image = transform_perspective_onto_cube_face(image=initial_image, template_image=template_image, contour=contour_list[i], bbox=bbox_mapping[i])
		images.append(image)

	# combine the outputs to form the cube
	final_image = np.full(images[0].shape, 0).astype(images[0].dtype)
	final_image = cv2.bitwise_or(final_image, images[0])
	final_image = cv2.bitwise_or(final_image, images[1])
	final_image = cv2.bitwise_or(final_image, images[2])

	aux = np.full(final_image.shape, 255).astype(final_image.dtype)
	cv2.fillPoly(aux, np.int32(bbox_mapping), color=(0, 0, 0))
	final_image = cv2.bitwise_or(final_image, aux)

	images.append(final_image)
	
	if verbose:
		utils.show_images(images_list=images, nrows=1, ncols=4)
	
	return final_image


# Find the transformation from our own template to a given template using a template matching algorithm
# After that, it applies the tranformation matrix M onto the own_template_with_sudoku
def template_matching(own_template, given_template, own_template_with_sudoku):
	own_template = own_template.copy()
	given_template = given_template.copy()
	own_template_with_sudoku = own_template_with_sudoku.copy()

	sift = cv2.SIFT_create()
	keypoints1, descriptors1 = sift.detectAndCompute(own_template, None)
	keypoints2, descriptors2 = sift.detectAndCompute(given_template, None)

	matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50}).knnMatch(descriptors1, descriptors2, k=2)
	candidates = [a for a, b in matches if a.distance < 0.7 * b.distance]

	reconstructed_image = own_template_with_sudoku
	if len(candidates) > 7:
		source_points = np.float32([keypoints1[a.queryIdx].pt for a in candidates]).reshape(-1,1,2)
		destination_points = np.float32([keypoints2[a.trainIdx].pt for a in candidates]).reshape(-1,1,2)
		
		M, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
		
		height, width = own_template.shape[:2]
		points = np.float32([[0,0], [0,height - 1], [width - 1,height - 1], [width - 1,0]]).reshape(-1,1,2)
		destination = cv2.perspectiveTransform(points, M)
		reconstructed_image = cv2.warpPerspective(own_template_with_sudoku, M, (given_template.shape[1], given_template.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

	return reconstructed_image


# The entire logic behind Task3.
def task3(image_path, save_path_text, save_path_image, template_given_path, template_own_path, model_path="./mnist_model.onnx", verbose=0):
	# get the sudoku images + their corresponding contour points
	(s1_image, s1_contour_points), (s2_image, s2_contour_points), (s3_image, s3_contour_points) = get_all_three_sudoku(image_path=image_path, width=1000, height=1000)

	# get the digits in matrix form
	matrix1 = apply_model(model_path=model_path, image=s1_image, no_cells=9, verbose=verbose)
	matrix2 = apply_model(model_path=model_path, image=s2_image, no_cells=9, verbose=verbose)
	matrix3 = apply_model(model_path=model_path, image=s3_image, no_cells=9, verbose=verbose)

	# find their position on the cube faces
	for permutation in list(itertools.permutations([[matrix1, s1_contour_points], [matrix2, s2_contour_points], [matrix3, s3_contour_points]])):
		matrices = [permutation[i][0] for i in range(len(permutation))]
		contours = [permutation[i][1] for i in range(len(permutation))]
		if check_second_condition(matrices=matrices):
			break
	[matrix1, matrix2, matrix3] = matrices
	
	first_matrix = "\n".join(["".join(matrix1[row_idx]) for row_idx in range(len(matrix1))])
	second_and_third_matrices = "\n".join(["".join(matrix2[row_idx]) + " " + "".join(matrix3[row_idx]) for row_idx in range(len(matrix1))])
	string_to_write_in_file = first_matrix + "\n\n" + second_and_third_matrices

	if save_path_text != None and save_path_text != "":
		with open(save_path_text, "w+") as f:
			f.write(string_to_write_in_file)
		print("The text output was saved at location: {}!".format(save_path_text))

	cube_image_on_own_template = construct_cube_on_own_template(initial_image_path=image_path, 
								  			    				template_image_path=template_own_path, 
								   							 	contour_list=contours,
								   							 	verbose=verbose)

	own_template = cv2.imread(template_own_path)
	given_template = cv2.imread(template_given_path)
	final_cube = template_matching(own_template=own_template, given_template=given_template, own_template_with_sudoku=cube_image_on_own_template)

	if save_path_image != None and save_path_image != "":
		cv2.imwrite(save_path_image, final_cube)
		print("The image output was saved at location: {}!".format(save_path_image))


if __name__ == "__main__":	
	args = parse_args()
	verbose = ord(args.verbose) - ord('0')

	if args.image_path != None:
		try:
			task3(image_path=args.image_path,
			 	  save_path_text=args.save_path_text,
			 	  save_path_image=args.save_path_image,
			 	  template_given_path=args.template_given_path,
			 	  template_own_path=args.template_own_path, 
			 	  verbose=verbose)
		except:
			raise Exception("An exception occured during the execution of task3!")

	else:
		os.makedirs(args.save_dir, exist_ok=True)
		
		if args.no_file != None:
			try:
				image_path = os.path.join(args.path_dir, "{}.jpg".format(args.no_file))
				save_path_text = os.path.join(args.save_dir, "{}_predicted.txt".format(args.no_file))
				save_path_image = os.path.join(args.save_dir, "{}_predicted.jpg".format(args.no_file))
				print("Processing the image located at {}".format(image_path))
				task3(image_path=image_path,
				 	  save_path_text=save_path_text,
				 	  save_path_image=save_path_image,
				 	  template_given_path=args.template_given_path,
				 	  template_own_path=args.template_own_path, 
				 	  verbose=verbose)
			except:
				raise Exception("An exception occured during the execution of task3 for the image located at: {}!".format(image_path))
		
		else:
			for no_file in range(1, 6):
				try:
					image_path = os.path.join(args.path_dir, "{}.jpg".format(no_file))
					save_path_text = os.path.join(args.save_dir, "{}_predicted.txt".format(no_file))
					save_path_image = os.path.join(args.save_dir, "{}_predicted.jpg".format(no_file))
					print("Processing the image located at {}".format(image_path))
					task3(image_path=image_path,
					 	  save_path_text=save_path_text,
					 	  save_path_image=save_path_image,
					 	  template_given_path=args.template_given_path,
					 	  template_own_path=args.template_own_path, 
					 	  verbose=verbose)
				except:
					raise Exception("An exception occured during the execution of task3 for the image located at: {}!".format(image_path))
	

	