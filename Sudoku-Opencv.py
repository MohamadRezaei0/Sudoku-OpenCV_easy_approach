#importing the required libraries
import cv2
import numpy as np
import pytesseract


def put_number(image, table, ans):

    """ Function to compare unsolved table
        with a solved sudoku table and the answer
        will be written on the image

    Arguments:
        image {numpy.ndarray} --
        table {numpy.array} -- unsolved array
        ans {numpy.array} -- solved array

    Returns:
        numpy.ndarray -- final image
    """
    image = cv2.resize(image, (450, 450))
    width = image.shape[0]
    height = image.shape[1]
    w = width//9
    h = height//9
    c = 0

    for i in range(0, 9):
        for j in range(0, 9):
            if table[i][j] == 0:
                x = j*w
                y = i*h
                cv2.putText(image, str(ans[i][j]), (int((x+w/3)), int((y+h/1.5)))\
                    ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image
# -----------------------------------------------------------
def solve(board):

    find = find_empty(board)

    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if valid(board, i, (row, col)):
            board[row][col] = i
            if solve(board):
                return True
            board[row][col] = 0
    return False

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                return (i, j)

    return None
# -----------------------------------------------------------
def valid(board, num, pos):
    # check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True
# -----------------------------------------------------------
def table_analyze(image):

    """The function divide image into 9x9
        equal parts.
      Google OCR engine (Pytesseract) reads every section.

    Arguments:
        image {numpy.ndarray} -- prepared image

    Returns:
        list -- intger table values
    """

    image = cv2.resize(image, (900, 900))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    width = image.shape[0]
    height = image.shape[1]
    w = width//9
    h = height//9

    list = []
    i = 0
    for x in range(0, width, w):
        for y in range(0, height, h):

            number_image = thresh[x + 10:x+w - 10, y + 10:y+h - 10]

            number = pytesseract.image_to_string(number_image, lang='eng',\
            config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')

            print('index[{}] = {}'.format(i, number))
            try:
                number = int(number)
            except:
                number = 0

            list.append(number)
            i = i + 1

    return list
# -----------------------------------------------------------
def arrange(n, size):

    """align 4 corners for undistort

    Arguments:
        n {list} -- x, y lsit of 4 corners
        size {int} -- size of dst

    Returns:
        src
        dst
    """
    # alignment of src and dst --> [[left],[top left], [top right][right]]
    points = np.float32([
        [n[0], n[1]],
        [n[2], n[3]],
        [n[4], n[5]],
        [n[6], n[7]]
    ])
    sum_list = []
    res = points.copy()

    for m in points:
        sum = 0
        sum = m[0] + m[1]
        sum_list.append(sum)
    sum_list.sort()

    for i in range(0, len(sum_list)):
        for point in points:
            if point[0] + point[1] == sum_list[i]:
                if i == 0:
                    res[1] = [point[0], point[1]]
                elif i == 1:
                    res[2] = [point[0], point[1]]
                elif i == 2:
                    res[0] = [point[0], point[1]]
                elif i == 3:
                    res[3] = [point[0], point[1]]
                else:
                    print('somthing went wrong')
    size = 250 # optional

    dst = np.float32([
        [0, size],
        [0, 0],
        [size ,0],
        [size, size]
    ])
    return res, dst
# -----------------------------------------------------------
def preprocess(image):

    """
        Function to apply preprocessing on image includes
        denoising and finding contours to
        undistort the image and remove useless parts
        of the image!
    """

    # Resize for more control over the image.
    image = cv2.resize(image, (500, 600))
    # no need for colouring the details
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=3, templateWindowSize=6, searchWindowSize=21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    canny = cv2.Canny(thresh, 120, 150, apertureSize=3)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Largest area belongs to largest rectangle in the image.
    # Largest rectangle actually is sudoku table border
    largest_area = 0
    largest_cnt = contours[0]
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, .01*cv2.arcLength(cnt, True), True)
        # Rectangle
        if len(approx == 4):
            area = cv2.contourArea(cnt)
            if largest_area <= area:
                largest_area = area
                largest_cnt = approx
    # finding 4 corners of the table.
    n = largest_cnt.ravel()
    # size of dst image
    size = 250
    src, dst = arrange(n, size)
    # undistort the image
    m = cv2.getPerspectiveTransform(src, dst)
    undistorted = cv2.warpPerspective(image, m, (size, size))

    print('preprocess completed')
    return undistorted
# -----------------------------------------------------------
"""
    main   |  |  |
           V  V  V
"""
# getting the image from the user as input
path = input('Please enter the sudku table path:')
image = cv2.imread(path, cv2.IMREAD_COLOR)

image = preprocess(image)
table = table_analyze(image) # reading sudoku table

ans = np.array(table).reshape((9, 9)).tolist()
table = np.array(table).reshape((9, 9)).tolist()

solve(ans)
# casting ans and table to the numpy.array to present answers on image and presenting a better print.
ans = np.array(ans)
table = np.array(table)
print(table)
print(ans)

image = put_number(image, table, ans)
cv2.imshow('my image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()