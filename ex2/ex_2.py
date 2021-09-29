import numpy as np
import scipy
import sys

train_x_input_path, train_y_output_lines_path, test_input_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(train_x_input_path) as train_x_input:
    train_x_input_lines = train_x_input.readlines()
with open(train_y_output_lines_path) as train_y_output:
    train_y_output_lines = train_y_output.readlines()
with open(test_input_path) as test_input:
    test_input_lines = test_input.readlines()
len_train_x = len(train_x_input_lines)

K = 10
# adding classes to the same array
train_x_array1 = train_x_input_lines[0].split(",")
parameters_class_Num = len(train_x_array1) + 1
train_input_output_x_y_array = np.zeros((len_train_x, parameters_class_Num))

index = 0  # goes from 0 - 355
len_trainx_sole_line = 0
# running every line in text:
for train_x_input_line in train_x_input_lines:
    train_x_array1 = train_x_input_line.split(",")
    len_trainx_sole_line = len(train_x_array1) - 1
    if train_x_array1[len_trainx_sole_line] == 'W\n':
        train_x_array1[len_trainx_sole_line] = 0
    else:
        train_x_array1[len_trainx_sole_line] = 1
    for i in range(len(train_x_array1)):
        train_input_output_x_y_array[index][i] = train_x_array1[i]
    index += 1
index = 0
for train_y_output_line in train_y_output_lines:
    class_num = train_y_output_line.split("\n")
    class_num = class_num[0]
    train_input_output_x_y_array[index][len_trainx_sole_line + 1] = class_num
    index += 1  # goes from 0 - 355(whole training len)

# chanig W,R the same for test:
test_len_sole_line = test_input_lines[0].split(",")
test_array = np.zeros((len(test_input_lines), len(test_len_sole_line)))
indexline = 0
# running every line in test than change w,r to 0 or 1:
for test_input_line in test_input_lines:
    test_splitted_line = test_input_line.split(",")
    test_len_sole_line = len(test_splitted_line) - 1
    if test_splitted_line[test_len_sole_line] == 'W\n':
        test_splitted_line[test_len_sole_line] = 0
    else:
        test_splitted_line[test_len_sole_line] = 1
    for i in range(len(test_splitted_line)):
        test_array[indexline][i] = test_splitted_line[i]
    indexline += 1

column_min_max_arr = np.zeros((2, 12))
check = 1  # using in def for creating max min only with train values


# x: input data in numpy array return normalized x per column, test normalized per train column*
def normalize_train(x, len_sole_line, check):
    min_val = 0
    max_val = 1
    index_len_line = 0
    index_column = 0
    len_all_lines = len(x)
    if check == 1:
        for ind_col in range(len(column_min_max_arr[0])):
            column_min_max_arr[0][ind_col] = np.max(x[:, ind_col])
            column_min_max_arr[1][ind_col] = np.min(x[:, ind_col])

    while index_len_line != len_all_lines:  # run over all lines
        while len_sole_line != index_column:  # run over all columns
            max_num = column_min_max_arr[0][index_column]
            min_num = column_min_max_arr[1][index_column]
            x[index_len_line][index_column] = ((x[index_len_line][index_column] - min_num) / (max_num - min_num)) * (
                    max_val - min_val) + min_val
            index_column += 1
        index_len_line += 1
        index_column = 0
    return x


normalize_train(train_input_output_x_y_array, len_trainx_sole_line, check)

check = 0
normalize_train(test_array, test_len_sole_line, check)

counter_tested_points = int(len(test_array))  # =8/10 of lines, 2/10 goes to validation
LEN_TESTED_P = counter_tested_points
counter_train_points = int(len(train_input_output_x_y_array))
LEN_TRAIN_P = counter_train_points

twelve_column_i = int((len(test_splitted_line)))
column_counter = 0
oclid_dis_counter = 0
oclidi_class_mat = np.zeros((counter_train_points, 2))  # left = oclidian dis, right = class specipi
K_mat_result_per_pointpoint = np.zeros((2, K))
k_lowest_oclid = 0
oclid_i = 0
temp_class = 4  # check if class is ok..
iter_K_sum_classes = 0
sum_classes0 = 0
sum_classes1 = 0
sum_classes2 = 0
class_mat_tested_pointes_KNN = np.zeros((LEN_TESTED_P, 1))
# for each tested point, oclid dis, than sort, than class by k sorted
while counter_tested_points != 0:
    while counter_train_points != 0:
        while column_counter < twelve_column_i:
            oclid_dis_counter += (test_array[-counter_tested_points + LEN_TESTED_P][column_counter] -
                                  train_input_output_x_y_array[LEN_TRAIN_P - counter_train_points][column_counter]) ** 2
            column_counter += 1
        column_counter = 0
        # oclid_dis_counter = np.sqrt(oclid_dis_counter)

        oclidi_class_mat[LEN_TRAIN_P - counter_train_points][0] = oclid_dis_counter
        oclidi_class_mat[LEN_TRAIN_P - counter_train_points][1] = \
            train_input_output_x_y_array[LEN_TRAIN_P - counter_train_points][twelve_column_i]
        oclid_dis_counter = 0
        counter_train_points -= 1
    # here sort K:
    while k_lowest_oclid != K:
        while oclid_i != LEN_TRAIN_P:
            # at the start only:
            if temp_class == 4:
                temp_oclid_num = oclidi_class_mat[oclid_i][0]
                temp_class = oclidi_class_mat[oclid_i][1]
                temp_oclid_index = oclid_i
                continue

            if temp_oclid_num > oclidi_class_mat[oclid_i][0]:
                temp_oclid_num = oclidi_class_mat[oclid_i][0]
                temp_class = oclidi_class_mat[oclid_i][1]
                temp_oclid_index = oclid_i
                continue
            oclid_i += 1
        oclid_i = 0
        K_mat_result_per_pointpoint[0][k_lowest_oclid] = temp_oclid_num
        K_mat_result_per_pointpoint[1][k_lowest_oclid] = temp_class
        # עדכון גודל מינימאלי למקסימום, + איפוס משתנים טמפורריים
        temp_class = 4
        temp_oclid_num = 9999999999999999999.9
        oclidi_class_mat[temp_oclid_index][0] = 9999999999999999999.9
        k_lowest_oclid += 1
    k_lowest_oclid = 0

    oclidi_class_mat = np.zeros((LEN_TRAIN_P, 2))

    # checking class by most or lowest if equal
    while iter_K_sum_classes < K:
        if K_mat_result_per_pointpoint[1][iter_K_sum_classes] == 0:
            sum_classes0 += 1
        elif K_mat_result_per_pointpoint[1][iter_K_sum_classes] == 1:
            sum_classes1 += 1
        else:
            sum_classes2 += 1
        iter_K_sum_classes += 1
    iter_K_sum_classes = 0
    if sum_classes0 >= sum_classes1 and sum_classes0 >= sum_classes2:
        class_mat_tested_pointes_KNN[LEN_TESTED_P - counter_tested_points][0] = 0
    elif sum_classes1 >= sum_classes2:
        class_mat_tested_pointes_KNN[LEN_TESTED_P - counter_tested_points][0] = 1
    else:
        class_mat_tested_pointes_KNN[LEN_TESTED_P - counter_tested_points][0] = 2

    sum_classes0 = 0
    sum_classes1 = 0
    sum_classes2 = 0

    counter_train_points = LEN_TRAIN_P
    counter_tested_points -= 1

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  PERCEPTRON:  @@@@@@@@@@@@@@@@@@@@@@@

epochs = 1000
eta = 0.2
w = np.zeros((12, 3))
X_train = np.zeros((1, 12))  # point 12 dim
y_Hat = 0.0

train_x_input_lines
train_y_output_lines
test_input_lines

# preparing the data:
train_x_array1 = train_x_input_lines[0].split(",")
parameters_class_Num = len(train_x_array1)
trainX_input_Arr = np.zeros((len_train_x, parameters_class_Num))
trainY_input_Arr = np.zeros((1, len_train_x))

index = 0  # goes from 0 - 355
len_trainx_sole_line = 0
# running every line in text:
for train_x_input_line in train_x_input_lines:
    train_x_array1 = train_x_input_line.split(",")
    len_trainx_sole_line = len(train_x_array1) - 1
    if train_x_array1[len_trainx_sole_line] == 'W\n':
        train_x_array1[len_trainx_sole_line] = 0
    else:
        train_x_array1[len_trainx_sole_line] = 1
    for i in range(len(train_x_array1)):
        trainX_input_Arr[index][i] = train_x_array1[i]
    index += 1
index = 0
for train_y_output_line in train_y_output_lines:
    class_num = train_y_output_line.split("\n")
    class_num = class_num[0]
    trainY_input_Arr[0][index] = class_num
    index += 1  # goes from 0 - 355(whole training len)

# chanig W,R the same for test:
test_len_sole_line = test_input_lines[0].split(",")
test_lines = np.zeros((len(test_input_lines), len(test_len_sole_line)))
indexline = 0
# running every line in test than change w,r to 0 or 1:
for test_input_line in test_input_lines:
    test_splitted_line = test_input_line.split(",")
    test_len_sole_line = len(test_splitted_line) - 1
    if test_splitted_line[test_len_sole_line] == 'W\n':
        test_splitted_line[test_len_sole_line] = 0
    else:
        test_splitted_line[test_len_sole_line] = 1
    for i in range(len(test_splitted_line)):
        test_lines[indexline][i] = test_splitted_line[i]
    indexline += 1

trainX_input_Arr
trainY_input_Arr
test_lines

# doing normalization:
normalize_train(trainX_input_Arr, len_trainx_sole_line, check)
normalize_train(test_lines, len_trainx_sole_line, check)

# for validation, getting 80% of trainX and trainY array:
full_length_train = len(trainX_input_Arr)
length_for_validation_train_points_array = int(full_length_train * (2 / 10))  # =2/10 goes to validation
length_for_learning_train_points_array = int(full_length_train * (8 / 10))  # =8/10 of lines to train

twenty_percent_trainX_Validationx = np.zeros((length_for_validation_train_points_array, 12))
eighty_percent_trainX = np.zeros((length_for_learning_train_points_array, 12))
twenty_percent_train_classY_Validaion = np.zeros((length_for_validation_train_points_array, 1))
eighty_percent_trainY = np.zeros((length_for_learning_train_points_array, 1))

columnX = 12
columnsY = length_for_learning_train_points_array
# filling training arrays:
for eighty_lines_percentX in range(length_for_learning_train_points_array):
    for i1 in range(columnX):
        eighty_percent_trainX[eighty_lines_percentX][i1] = trainX_input_Arr[eighty_lines_percentX][i1]
for i2 in range(length_for_learning_train_points_array):
    eighty_percent_trainY[i2][0] = trainY_input_Arr[0][i2]

# filling from 284 to the end:
ind_val = 0
while ind_val < length_for_validation_train_points_array:
    for i3 in range(columnX):
        twenty_percent_trainX_Validationx[ind_val][i3] = \
        trainX_input_Arr[full_length_train - length_for_validation_train_points_array + ind_val][i3]
    ind_val += 1
for i4 in range(length_for_validation_train_points_array):
    twenty_percent_train_classY_Validaion[i4][0] = trainY_input_Arr[0][
        full_length_train - length_for_validation_train_points_array + i4]
ind_val = 0

W = np.ones((3, 12))  # weights, start with 1's in order the eta will change it (instead of zeroes)
W_best = np.ones((3, 12))
W_best_2 = np.ones((3, 12))


def shuffle(Xset, Yset):
    assert len(Xset) == len(Yset)
    permutation = np.random.permutation(len(Xset))
    return Xset[permutation], Yset[permutation]


changin_eta_for_this_num_of_epoech = 50
whole_percentage_train = 0
whole_percentage_val = 0
check_w = 0
for e in range(epochs):
    if e % changin_eta_for_this_num_of_epoech == 0:
        eta *= 0.9
    eighty_percent_trainX, eighty_percent_trainY = shuffle(eighty_percent_trainX, eighty_percent_trainY)
    for x, y in zip(eighty_percent_trainX, eighty_percent_trainY):
        # predict
        k = (np.dot(W, x))
        y_hat = int(np.argmax(k))
        # update
        the_num_class = int(y[0])
        if the_num_class != y_hat:
            W[the_num_class, :] = W[the_num_class, :] + eta * x
            z = eta * x
            W[y_hat, :] = W[y_hat, :] - z
    # success rate:
    max_num_eighty_percent_success = length_for_learning_train_points_array
    max_num_twenty_percent_success = length_for_validation_train_points_array
    success_train_counter = 0
    success_val_counter = 0

    for x, y in zip(eighty_percent_trainX, eighty_percent_trainY):
        predicted_class_train = int(np.argmax(np.dot(W, x)))
        the_class_train = int(y)
        if predicted_class_train == the_class_train:
            success_train_counter += 1
    for x, y in zip(twenty_percent_trainX_Validationx, twenty_percent_train_classY_Validaion):
        predicted_class_val = int(np.argmax(np.dot(W, x)))
        the_class_val = int(y)
        if predicted_class_val == the_class_val:
            success_val_counter += 1
    #whole_percentage_train += 100 * success_train_counter / max_num_eighty_percent_success
    #whole_percentage_val += 100 * success_val_counter / max_num_twenty_percent_success
    success_train_percentage = 100 * success_train_counter / max_num_eighty_percent_success
    success_val_percentage = 100 * success_val_counter / max_num_twenty_percent_success
    if success_train_percentage >= 75 and success_val_percentage >= 80 and success_train_percentage <= 80 and success_val_percentage <= 85:
        W_best = W
    if success_train_percentage >= 80 and success_val_percentage >= 85 and success_train_percentage <= 85 and success_val_percentage <= 90:
        W_best_2 = W
        check_w = 3
    # print("success_train_percentage= ",success_train_percentage,"\nsuccess_val_percentage= ",success_val_percentage,"\n")
    success_train_counter = 0
    success_val_counter = 0
if check_w == 3:
    W_best = W_best_2
# print("whole_percentage_train= ",whole_percentage_train/1000, "\nwhole_percentage_train= ",whole_percentage_val/1000)
len_test = len(test_lines)
mat_perceptron_result = np.zeros((len_test, 1))
test_line_filler = 0
for in_test in test_lines:
    z = np.dot(W_best, in_test)
    result_per_point = int(np.argmax(z))
    mat_perceptron_result[test_line_filler][0] = int(result_per_point)
    test_line_filler += 1
test_line_filler = 0
# print(mat_perceptron_result)
# print(class_mat_tested_pointes_KNN)
# print(mat_perceptron_result-class_mat_tested_pointes_KNN)

# PA ALGORITHM @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

"""
המימוש של PA דומה מאוד לשל PERCEPTRON, ההבדל הוא שאת ETA מחליף TAU שהחישוב שלו הוא
max(0, (1.0 - np.dot(w_y, x) + np.dot(w_y_hat, x))) / (2 * ((np.linalg.norm(x)) ** 2)) 
ושכאשר מחפשים את ה yhat על ידי הכפלת כל וקטור בW בx ולקיחת האינדקס של הוקטור ב W שהביא לתוצאה המקסימלית,
צריך קודם לכן להוציא את הוקטור שמייצג את y האמיתי בW מW.
"""
TAU = 0
check_w_pa = 0
W_PA = np.ones((3, 12))
W_best_PA = np.ones((3, 12))
for e in range(epochs):
    eighty_percent_trainX, eighty_percent_trainY = shuffle(eighty_percent_trainX, eighty_percent_trainY)
    for x, y in zip(eighty_percent_trainX, eighty_percent_trainY):
        # predict
        k = (np.dot(W_PA, x))
        y_hat = int(np.argmax(k))
        the_num_class = int(y[0])
        TAU = max(0, (1.0 - np.dot(W_PA[the_num_class][:], x) + np.dot(W_PA[y_hat][:], x))) / (
                    2 * ((np.linalg.norm(x)) ** 2))
        # update
        if the_num_class != y_hat:
            z = TAU * x
            W_PA[the_num_class, :] = W_PA[the_num_class, :] + z
            W_PA[y_hat, :] = W_PA[y_hat, :] - z
    # success rate:
    max_num_eighty_percent_success = length_for_learning_train_points_array
    max_num_twenty_percent_success = length_for_validation_train_points_array
    success_train_counter = 0
    success_val_counter = 0

    for x, y in zip(eighty_percent_trainX, eighty_percent_trainY):
        predicted_class_train = int(np.argmax(np.dot(W_PA, x)))
        the_class_train = int(y)
        if predicted_class_train == the_class_train:
            success_train_counter += 1
    for x, y in zip(twenty_percent_trainX_Validationx, twenty_percent_train_classY_Validaion):
        predicted_class_val = int(np.argmax(np.dot(W_PA, x)))
        the_class_val = int(y)
        if predicted_class_val == the_class_val:
            success_val_counter += 1
    #whole_percentage_train += 100 * success_train_counter / max_num_eighty_percent_success
    #whole_percentage_val += 100 * success_val_counter / max_num_twenty_percent_success
    success_train_percentage = 100 * success_train_counter / max_num_eighty_percent_success
    success_val_percentage = 100 * success_val_counter / max_num_twenty_percent_success
    if success_train_percentage >= 70 and success_val_percentage >= 78 and success_train_percentage <= 80 and success_val_percentage <= 90:
        W_best_PA = W_PA
    if success_train_percentage >= 80 and success_val_percentage >= 85 and success_train_percentage <= 85 and success_val_percentage <= 90:
        W_best_PA_2 = W_PA
        check_w_pa = 4
    # print("success_train_percentage= ",success_train_percentage,"\nsuccess_val_percentage= ",success_val_percentage,"\n")
    success_train_counter = 0
    success_val_counter = 0
if check_w_pa == 4:
    W_best_PA = W_best_PA_2
# print("whole_percentage_train= ",whole_percentage_train/1000, "\nwhole_percentage_train= ",whole_percentage_val/1000)
len_test = len(test_lines)
mat_PA = np.zeros((len_test, 1))
test_line_filler = 0
for in_test in test_lines:
    z = np.dot(W_best_PA, in_test)
    result_per_point = int(np.argmax(z))
    mat_PA[test_line_filler][0] = int(result_per_point)
    test_line_filler += 1
test_line_filler = 0
# print(mat_PA-class_mat_tested_pointes_KNN)
last_len_counter = len(class_mat_tested_pointes_KNN)
last_counter = 0
while last_counter != last_len_counter:
    # print("knn:",int(class_mat_tested_pointes_KNN[last_counter][0]), ", perceptron:",int(mat_perceptron_result[last_counter][0]),", pa: ",int(mat_PA[last_counter][0]))
    print('knn:%d, perceptron: %d, pa: %d' % (
    int(class_mat_tested_pointes_KNN[last_counter][0]), int(mat_perceptron_result[last_counter][0]),
    int(mat_PA[last_counter][0])))
    last_counter += 1
