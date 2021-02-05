import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

# processing, blur, thresh, and invert colors returns the processed image
def process(img, skip_dilate=False):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.GaussianBlur(grayImg.copy(), (9, 9), 0)
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed = cv2.bitwise_not(processed, processed)

    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        processed = cv2.dilate(processed, kernel)
    return processed

# return an array with outer corners
def corners(img):

    #finds external contours
    extContours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extContours = extContours[0] if len(extContours) == 2 else extContours[1]
    extContours = sorted(extContours, key=cv2.contourArea, reverse=True)

    for c in extContours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            return approx

# corners in order
def orderCorners(corners):

    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    topR, topL, bottomL, bottomR = corners[0], corners[1], corners[2], corners[3]
    return topR, topL, bottomL, bottomR


# Crop the image
def perspectiveTransform(image, corners):
    ordered_corners = orderCorners(corners)
    topR, topL, bottomL, bottomR = ordered_corners

    # width of cropped image
    widthA = np.sqrt(((bottomR[0] - bottomL[0]) ** 2) + ((bottomR[1] - bottomL[1]) ** 2))
    widthB = np.sqrt(((topR[0] - topL[0]) ** 2) + ((topR[1] - topL[1]) ** 2))
    width = max(int(widthA), int(widthB))

    # height of cropped image
    heightA = np.sqrt(((topR[0] - bottomR[0]) ** 2) + ((topR[1] - bottomR[1]) ** 2))
    heightB = np.sqrt(((topL[0] - bottomL[0]) ** 2) + ((topL[1] - bottomL[1]) ** 2))
    height = max(int(heightA), int(heightB))

    # new cropped dimensions
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    # Return the transformed image
    return cv2.warpPerspective(image, grid, (width, height))

# returns array of the individual cells
def createImageGrid(img):
    grid = np.copy(img)
    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9

    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    grid = cv2.bitwise_not(grid, grid)

    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])

    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

    try:
        for i in range(9):
            for j in range(9):
                np.os.remove("C:/coding/imageFiles/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("C:/coding/imageFiles/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])


    return finalgrid


def center(img, size, margin=20, background=0):
    h, w = img.shape[:2]

    def centrePad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centrePad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centrePad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def predict(img_grid):
    image = img_grid.copy()

    image = cv2.resize(image, (28, 28))

    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    model = load_model("C:/coding/Pycharm/openCvPractice/model.h5")
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)

    return pred.argmax()

def drawSolution(board):
    # convert the board to string
    solution = "------------\n"
    for i in range(9):
        solution = solution + "| "
        for j in range(0,9,3):
            solution = solution + str(board[i][j]) + str(board[i][j+1]) + str(board[i][j+2]) + " | "
        solution = solution + "\n"
        if (i+1)%3 == 0:
            solution = solution + "------------\n"



    blank = np.zeros((512, 512, 3), np.uint8)
    y0, dy = 75, 25
    for i, line in enumerate(solution.split('\n')):
        y = y0 + i*dy
        cv2.putText(blank, line, (75, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('BLANK', blank)
    cv2.waitKey(0)

def confirmGrid(board):
    printBoard(board)
    change = " "
    while (change != "done"):
        change = input("Enter row, col, num to input in one string, enter done when finished: ")
        if (change == "done"):
            break
        else:
            board[int(change[0])][int(change[1])] = int(change[2])


#main function that processes th image
def extract():

    img = cv2.imread("C:/coding/imageFiles/sudoku6.jpg")
    img = cv2.rotate(img, 2)
    processed_sudoku = process(img)
    sudoku = corners(processed_sudoku)
    transformed = perspectiveTransform(img, sudoku)
    transformed = cv2.flip(transformed, 1)
    cropped = 'C:/coding/imageFiles/croppedSudoku6.jpg'
    cv2.imwrite(cropped, transformed)
    transformed = cv2.resize(transformed, (450, 450))
    sudoku = createImageGrid(transformed)
    return sudoku


# extract the number from the cell and predict the value using predict function
def extract_number_image(img_grid):
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):

            image = img_grid[i][j]
            image = cv2.resize(image, (28, 28))
            gray = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

            # Find contours
            cnts = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                if (x < 3 or y < 3 or h < 3 or w < 3):
                    continue
                ROI = gray[y:y + h, x:x + w]
                ROI = center(ROI, 120)

                num = predict(ROI)
                tmp_sudoku[i][j] = num

    return tmp_sudoku


#solve sudoku

def printBoard(board):
    for i in range(9):
        for j in range(0, 9, 3):
            print(board[i][j], board[i][j+1], board[i][j+2], "|", end=" ")
        if (i+1)%3 == 0:
            print(" ")
            print("-----------------------")
        else:
            print(" ")
    print(" ")

def findFreeSpot(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)

    return False

def isCorrect(board, num, row, col):
    # check if num is in row
    for i in range(9):
        if board[row][i] == num:
            return False

    # check if num is in col
    for j in range(9):
        if board[j][col] == num:
            return False

    # check if num is in same box
    boxRow = row//3
    boxColumn = col//3

    for i in range(boxRow*3, boxRow*3 + 3):
        for j in range(boxColumn*3, boxColumn*3 + 3):
            if board[i][j] == num:
                return False



    return True


def solve(board):
    if findFreeSpot(board) == False:
        return True

    tmpRow, tmpCol = findFreeSpot(board)

    for cellNum in range(1,10):
        if isCorrect(board, cellNum, tmpRow, tmpCol):
            board[tmpRow][tmpCol] = cellNum
            if solve(board):
                return True
            board[tmpRow][tmpCol] = 0


    return False

def main():

    sudokuBoard = extract_number_image(extract())
    confirmGrid(sudokuBoard)



    if (solve(sudokuBoard)):
        drawSolution(sudokuBoard)
    else:
        print("NO SOLUTIONS")

main()






