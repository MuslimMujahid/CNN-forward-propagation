import numpy as np
import cv2
import pickle

def loadImage(filepath: str, size=None):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)

    if (size):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img


def showImage(img: np.ndarray) -> None:
    cv2.imshow("image", img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def conv2D(X: np.ndarray, kernel: np.ndarray, padding: int, stride: tuple, bias) -> np.ndarray:
    # padding
    X = np.pad(X, padding, mode='constant')
    x_height, x_width = X.shape
    k_height, k_width = kernel.shape

    f_height, f_width = ((x_height-k_height) //
                         stride[0] + 1, (x_width-k_width)//stride[1] + 1)
    feature_map = np.zeros([f_height, f_width], dtype=int)

    for i in range(f_height):
        for j in range(f_width):
            feature_map[i, j] = np.sum(np.multiply(X[i*stride[1] : i*stride[1]+k_width,
                                                     j*stride[0] : j*stride[0]+k_width], kernel[:, :]))

    return feature_map


def conv3D(X: np.ndarray, kernel: np.ndarray, padding: int, stride: tuple, bias) -> np.ndarray:
    return np.add(
        conv2D(X[2, :, :], kernel[2, :, :], padding, stride, bias),
        np.add(
            conv2D(X[0, :, :], kernel[0, :, :], padding, stride, bias),
            conv2D(X[1, :, :], kernel[1, :, :], padding, stride, bias)))

    return feature_map

def save(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, 'rb'))

# 10 Fold Cross Validation Function
def get_tp_fp_tn_fn(y_actual, y_pred):
    TP = []
    FP = []
    TN = []
    FN = []
    
    for positive in (list(set(y_actual))):
        TP_temp = 0
        FP_temp = 0
        TN_temp = 0
        FN_temp = 0
        for i in range(len(y_pred)):
            if y_actual[i]==y_pred[i]==positive:
                TP_temp += 1
            if y_pred[i]==positive and y_actual[i]!=positive:
                FP_temp += 1
            if y_pred[i]!=positive and y_actual[i]!=positive:
                TN_temp += 1
            if y_pred[i]!=positive and y_actual[i]==positive:
                FN_temp += 1

        TP.append(TP_temp)
        FP.append(FP_temp)
        TN.append(TN_temp)
        FN.append(FN_temp)

    return(TP, FP, TN, FN)

def acc_result(y_actual, y_pred):
    TP, FP, TN, FN = get_tp_fp_tn_fn(y_actual, y_pred)
    idx = list(set(y_actual))
    acc = []
    total = 0
    for i in range(len(TP)):
        denom = TP[i] + FP[i] + TN[i] + FN[i]
        if denom != 0:
          res = (TP[i] + TN[i])/ denom
          acc.append(res)
          total += res
    total = total / (len(TP))
    return total

def precision_result(y_actual, y_pred):
    TP, FP, TN, FN = get_tp_fp_tn_fn(y_actual, y_pred)
    idx = list(set(y_actual))
    precision = []
    for i in range(len(TP)):
        denom = TP[i] + FP[i]
        if denom != 0:
          res = TP[i] / denom
        else:
          res = 0
        precision.append(res)
        print(str(idx[i]) + " : " + str(res))
    return precision

def recall_result(y_actual, y_pred):
    TP, FP, TN, FN = get_tp_fp_tn_fn(y_actual, y_pred)
    idx = list(set(y_actual))
    recall = []
    for i in range(len(TP)):
        denom = TP[i] + FN[i]
        if denom != 0:
          res = TP[i] / denom
        else:
          res = 0
        recall.append(res)
        print(str(idx[i]) + " : " + str(res))
    return recall

def f1_result(y_actual, y_pred):
    TP, FP, TN, FN = get_tp_fp_tn_fn(y_actual, y_pred)
    idx = list(set(y_actual))
    f1 = []
    for i in range(len(TP)):
        denom = (2 * TP[i]) + FP[i] + FN[i]
        if denom != 0:
          res = (2 * TP[i]) / denom
        else:
          res = 0
        f1.append(res)
        print(str(idx[i]) + " : " + str(res))
    return f1

def confusion_matrix(y_actual, y_pred): 
  possible_val = []
  for val in y_actual:
    if val not in possible_val:
      possible_val.append(val)
  for val in y_pred:
    if val not in possible_val:
      possible_val.append(val)
  dic = {}
  possible_val.sort()
  for i in range(len(possible_val)):
    dic[possible_val[i]] = i
  mat = [[0 for j in range(len(possible_val))]
    for i in range(len(possible_val))] 
  for i in range(len(y_actual)):
    mat[dic[y_actual[i]]][dic[y_pred[i]]] += 1 
  return mat