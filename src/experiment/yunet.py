# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

from itertools import product

import numpy as np
import cv2 as cv

class YuNet:
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        """
        __init__ function
        @param modelPath:
        @param inputSize:
        @param confThreshold:
        @param nmsThreshold:
        @param topK:
        @param backendId:
        @param targetId:
        """
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        """
        return the name of the class
        @return:
        """
        return self.__class__.__name__

    def setBackend(self, backendId):
        """
        Set the backend
        @param backendId:
        @return:
        """
        self._backendId = backendId
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setTarget(self, targetId):
        """
        Set the target
        @param targetId:
        @return:
        """
        self._targetId = targetId
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        """
        Set the input size to desired size
        @param input_size:
        @return:
        """
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        """
        Inference on the image using YuNet network
        @param image:
        @return:
        """
        # Forward
        faces = self._model.detect(image)
        return faces[1]