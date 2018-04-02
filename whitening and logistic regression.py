# import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression

#For whitening
def J (w, faces, labels, alpha = 0.):
    J =0.5 * (labels - faces.dot(w)).T.dot(labels - faces.dot(w))+ 0.5 * alpha * w.dot(w)
    return J

def gradJ (w, faces, labels, alpha = 0.):
    J = -faces.T.dot(labels-faces.dot(w)) +alpha * w
    return J

def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    rate = 0.6
    w = np.random.randn(trainingFaces.shape[1])
    cost = J(w,trainingFaces,trainingLabels,alpha)+1
    i=1
    while(abs(cost-J(w,trainingFaces,trainingLabels,alpha))>0.5):
        cost = J(w,trainingFaces,trainingLabels,alpha)
        w -= rate * gradJ(w,trainingFaces,trainingLabels,alpha)
        print(J(w,trainingFaces,trainingLabels,alpha))
        i+=1
    print i
    return w

def whiteningPara(trainingFaces):
    cov = np.dot(trainingFaces.T, trainingFaces) + 0.01 * np.eye(trainingFaces.shape[1])
    val, vec = np.linalg.eigh(cov)
    white_para = np.dot(vec, np.diag(1. / np.sqrt(val)))
    return white_para

def whitening(faces, para):
    return np.dot(faces, para)

def whitenMethod(trainingFaces, trainingLabels, testingFaces, testingLabels):
    para = whiteningPara(trainingFaces)
    whitenedTrainF = whitening(trainingFaces, para)
    whitenedTestF = whitening(testingFaces, para)
    w = gradientDescent(whitenedTrainF, trainingLabels, whitenedTestF, testingLabels)
    print ("Training cost: {}".format(J(w, whitenedTrainF, trainingLabels)))
    print ("Testing cost:  {}".format(J(w, whitenedTestF, testingLabels)))
    return w

#For cross entropy
def sigmod(w, faces):
    z = faces.dot(w)
    yhat = 1. / (1. + np.exp(-z))
    return yhat

def JCE(w, faces, labels, alpha = 0.):
    yhat = sigmod(w, faces)
    m = faces.shape[0]
    yhat = yhat.clip(1e-5, 1.-1e-5)
    yhat_1, yhat_0 = np.log(yhat), np.log(1. - yhat)
    cost = (-1./m)*(labels.T.dot(yhat_1) + (1 - labels).T.dot(yhat_0)) + 0.5 * alpha * w.T.dot(w)
    return cost

def gradJCE (w, faces, labels, alpha = 0.):
    yhat = sigmod(w, faces)
    m = faces.shape[0]
    grad = (1./m)*faces.T.dot(yhat - labels) + alpha * w
    return grad

def gradientDescentCE (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0., lr = 0.01, reportcost = 0):
    w = np.random.randn(trainingFaces.shape[1])
    cost_pre = JCE(w, trainingFaces, trainingLabels, alpha)
    count  = 0
    while cost_pre:
        w = w - lr * gradJCE(w, trainingFaces, trainingLabels)
        cost = JCE(w, trainingFaces, trainingLabels, alpha)
        count += 1
        print "training cost is {}".format(cost_pre)
        if np.abs((cost_pre - cost)) < 5e-6:
            print "optimal training cost is {}".format(cost)
            print "iteration time is {}".format(count)
            break
        else:
            cost_pre = cost
    return w
    # TODO implement this!

def CEmethod (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 0.
    w = gradientDescentCE(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha,  lr= 2e-1, reportcost = 1)
    return w

# def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels):
#     w = np.linalg.solve((trainingFaces.T.dot(trainingFaces)),trainingFaces.T.dot(trainingLabels))
#     return w
#
# def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
#     return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)
#
# def method3 (trainingFaces, trainingLabels, testingFaces, testingLabels):
#     alpha = 1e3
#     return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)

# def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
#     print ("Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha)))
#     print ("Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha)))

def reportCostsCE (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print ("Training cost: {}".format(JCE(w, trainingFaces, trainingLabels, alpha)))
    print ("Testing cost:  {}".format(JCE(w, testingFaces, testingLabels, alpha)))

# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
# def detectSmiles (w):
#     # Given the image captured from the web camera, classify the smile
#     def classifySmile (im, imGray, faceBox, w):
#         # Extract face patch as vector
#         face = imGray[faceBox[1]:faceBox[1]+faceBox[3], faceBox[0]:faceBox[0]+faceBox[2]]
#         face = cv2.resize(face, (24, 24))
#         face = (face - np.mean(face)) / np.std(face)  # Normalize
#         face = np.reshape(face, face.shape[0]*face.shape[1])
#
#         # Classify face patch
#         yhat = w.dot(face)
#         print yhat
#
#         # Draw result as colored rectangle
#         THICKNESS = 3
#         green = 128 + (yhat - 0.5) * 255
#         color = (0, green, 255 - green)
#         pt1 = (faceBox[0], faceBox[1])
#         pt2 = (faceBox[0]+faceBox[2], faceBox[1]+faceBox[3])
#         cv2.rectangle(im, pt1, pt2, color, THICKNESS)
#
#     # Starting video capture
#     vc = cv2.VideoCapture()
#     vc.open(0)
#     faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # TODO update the path
#     while vc.grab():
#         (tf,im) = vc.read()
#         im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))  # Divide resolution by 2 for speed
#         imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         k = cv2.waitKey(30)
#         if k >= 0 and chr(k) == 'q':
#             print "quitting"
#             break
#
#         # Detect faces
#         faceBoxes = faceDetector.detectMultiScale(imGray)
#         for faceBox in faceBoxes:
#             classifySmile(im, imGray, faceBox, w)
#         cv2.imshow("WebCam", im)
#
#     cv2.destroyWindow("WebCam")
#     vc.release()

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")
    whitened_w = whitenMethod(trainingFaces, trainingLabels, testingFaces, testingLabels)
    ce_w = CEmethod(trainingFaces, trainingLabels, testingFaces, testingLabels)
    reportCostsCE (ce_w, trainingFaces, trainingLabels, testingFaces, testingLabels)

    # reportCosts(whitened_w,trainingFaces, trainingLabels, testingFaces, testingLabels)
    # w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
    # w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)
    # w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)

    # for w in [ w1, w2, w3 ]:
    #     reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)

    proba_train = sigmod(ce_w,trainingFaces)
    proba_test = sigmod(ce_w,testingFaces)
    likelihood_training = np.power(2, (-JCE(ce_w, trainingFaces, trainingLabels)))
    likelihood_testing = np.power(2, (-JCE(ce_w, testingFaces, testingLabels)))
    model = LogisticRegression(C=9999999999, fit_intercept=False)
    model = model.fit(trainingFaces, trainingLabels)
    proba_training = model.predict_proba(trainingFaces)[:, 0]
    proba_testing = model.predict_proba(testingFaces)[:, 0]

    print "training probability:",proba_train.mean(),proba_training.mean(),likelihood_training
    print "Testing probability:",proba_test.mean(),proba_testing.mean(),likelihood_testing

    check = check_grad(JCE, gradJCE, ce_w, trainingFaces, trainingLabels)
    print "check_grad", check


    #detectSmiles(w3)  # Requires OpenCV
