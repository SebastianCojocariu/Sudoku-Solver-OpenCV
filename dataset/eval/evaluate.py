import os

def evaluate_results_task1(predictions_path,ground_truth_path,verbose=0):
    total_correct = 0
    for i in range(1, 21):
        try:
            filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
            p = open(filename_predictions,"rt")        
            filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
            gt = open(filename_ground_truth,"rt")
            correct_flag = 1
            for row in range(1,10):
                p_line = p.readline()
                gt_line = gt.readline()
                #print(p_line)
                #print(gt_line)
                if (p_line[:10] != gt_line[:10]):
                    correct_flag = 0
            p.close()
            gt.close()
            if verbose:
                print("Task 1 - Classic Sudoku: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
            
            total_correct = total_correct + correct_flag
            
        except:
            continue
    
    points = total_correct * 0.05
    return total_correct, points

def evaluate_results_task2(predictions_path,ground_truth_path,verbose = 0):
    total_correct = 0
    for i in range(1, 16):
        try:
            filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
            p = open(filename_predictions,"rt")        
            filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
            gt = open(filename_ground_truth,"rt")
            correct_flag = 1
            for row in range(1,10):
                p_line = p.readline()
                gt_line = gt.readline()
                #print(p_line)
                #print(gt_line)
                if (p_line[:19] != gt_line[:19]):
                    correct_flag = 0        
            p.close()
            gt.close()
            
            if verbose:
                print("Task 2 - Jigsaw Sudoku: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
            
            total_correct = total_correct + correct_flag
            
        except:
            continue
    
    points = total_correct * 0.05
    return total_correct, points


def evaluate_results_task3(predictions_path,ground_truth_path, verbose = 0):
    total_correct = 0
    for i in range(1, 6):
        try:
            filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
            p = open(filename_predictions,"rt")        
            filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
            gt = open(filename_ground_truth,"rt")
            correct_flag = 1
            for row in range(1,10):
                p_line = p.readline()
                gt_line = gt.readline()
                #print(p_line)
                #print(gt_line)
                if (p_line[:10] != gt_line[:10]):
                    correct_flag = 0
            
            p_line = p.readline()
            gt_line = gt.readline()
            
            for row in range(1,10):
                p_line = p.readline()
                gt_line = gt.readline()
                #print(p_line)
                #print(gt_line)
                if (p_line[:20] != gt_line[:20]):
                    correct_flag = 0
            p.close()
            gt.close()
            
            if verbose:
                print("Task 3 - Sudoku Cube: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
            
            total_correct = total_correct + correct_flag
        except:
            continue
    
    points = total_correct * 0.05
    return total_correct,points 


verbose = 1

#change this on your machine
predictions_path_root = "../predictions/"
ground_truth_path_root = "../images/"

#task1
predictions_path = predictions_path_root + "classic"
ground_truth_path = ground_truth_path_root + "classic"
total_correct_task1, points_task1 = evaluate_results_task1(predictions_path,ground_truth_path,verbose)
#print(total_correct_task1,points_task1)

#task2
predictions_path = predictions_path_root + "jigsaw"
ground_truth_path = ground_truth_path_root + "jigsaw"
total_correct_task2, points_task2 = evaluate_results_task2(predictions_path,ground_truth_path,verbose)
#print(total_correct_task2,points_task2)

#task3
predictions_path = predictions_path_root + "cube"
ground_truth_path = ground_truth_path_root + "cube"
total_correct_task3, points_task3 = evaluate_results_task3(predictions_path,ground_truth_path,verbose)
#print(total_correct_task3,points_task3)

print("Task 1 = ", points_task1, "\nTask 2 = ",points_task2, "\nTask 3 = ", points_task3, "\nTo add points from Task 3 results  + to add 0.5 points ex officio")