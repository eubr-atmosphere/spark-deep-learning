#!/usr/bin/python2
"""
Copyright 2018 Demet Sude Saplik
Copyright 2018 Danilo Ardagna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys,time,os,shutil,csv
import numpy as np
from subprocess import call
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, LinearSVC
from pyspark.ml.classification import LogisticRegressionModel,RandomForestClassificationModel, DecisionTreeClassificationModel,LinearSVCModel

from pyspark.sql.functions import lit
from sparkdl import readImages, DeepImagePredictor, DeepImageFeaturizer
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext()
spark = SparkSession(sc)


class Predictor:

    def __init__(self):
        self.images_num_sparkdl_run_best = int(os.environ["IMAGES_NUM_SPARKDL_RUN_BEST"])
        self.images_num_sparkdl = int(os.environ["IMAGES_NUM_SPARKDL"])
        self.img_dir = os.environ["IMG_DIR"]
        self.test_dir = os.environ["TEST_DIR"]
        self.hdfs_path = os.environ["HDFS_PATH"]
        self.best_model_path = os.environ["MODEL_PATH"]
        self.best_model_name = self.best_model_path.split("/")[2]

    def read_images(self):

        class_paths = {}
        class_paths_test = {}
        class_img_nums = {}

        dirlist = [ item for item in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, item)) ]

        if os.path.isdir(self.test_dir) == True :
                shutil.rmtree(self.test_dir)

        for directory in dirlist :

            class_paths[directory] = os.path.join(self.img_dir,directory)
            num = self.images_num_sparkdl + self.images_num_sparkdl_run_best
            files = os.listdir(class_paths[directory])[self.images_num_sparkdl:num]

            for eachfilename in files:
                src = os.path.join(class_paths[directory],eachfilename)
                dst = os.path.join(self.test_dir,directory)
                if not os.path.exists(dst):
                    os.makedirs(dst)
                retrn_val = shutil.copy(src, os.path.join(dst,eachfilename))
            class_paths_test[directory] = os.path.join(dst)

        hdfs_path_run = os.path.join(self.hdfs_path, "test")
        exists = os.system("hadoop fs -test -d %s" % (hdfs_path_run))
        if exists == 0 :
            exists = os.system("hadoop fs -rm -r -skipTrash %s" % (hdfs_path_run))
        os.system("hadoop fs -copyFromLocal %s %s" % (self.test_dir, self.hdfs_path))
        for directory in dirlist :
            class_paths_test[directory] = os.path.join(self.hdfs_path,"test",directory)

        test_df = readImages(class_paths_test[dirlist[0]]).withColumn("label", lit(dirlist[0]))
        for class_label in range(1,len(dirlist)) :
            classi_df = readImages(class_paths_test[dirlist[class_label]]).withColumn("label", lit(dirlist[class_label]))
            test_df = test_df.unionAll(classi_df)

        return test_df

    def read_model(self):

        if "LogisticRegression" in self.best_model_path :
            classifier = LogisticRegressionModel.load(self.best_model_path)

        elif "DecisionTree" in self.best_model_path :
            classifier = DecisionTreeClassificationModel.load(self.best_model_path)

        elif "RandomForest" in self.best_model_path :
            classifier = RandomForestClassificationModel.load(self.best_model_path)

        elif "LinearSVC" in self.best_model_path :
            classifier = LinearSVCModel.load(self.best_model_path)


        if "VGG16" in self.best_model_path :
            featurizer_name = "VGG16"

        elif "VGG19" in self.best_model_path :
            featurizer_name = "VGG19"

        elif "InceptionV3" in self.best_model_path :
            featurizer_name = "InceptionV3"

        elif "Xception" in self.best_model_path :
            featurizer_name = "Xception"

        elif "ResNet50" in self.best_model_path :
            featurizer_name = "ResNet50"

        return featurizer_name, classifier

    def predicate(self,featurizer_name, classifier, test_df):

        featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName=featurizer_name)
        predictor = PipelineModel(stages=[featurizer, classifier])
        predictions = predictor.transform(test_df)
        return predictions

    def save_predictions(self,predictions):
        hdfs_predictions_path = os.path.join(self.hdfs_path,"predictions")
        driver_predictions_path = os.path.join(".","results","runs")

        print("hdfs_predictions_path : " + hdfs_predictions_path)
        print("driver_predictions_path : " + driver_predictions_path)

        predictions.select("label","prediction").write.format("csv").mode('overwrite')\
                   .option("header", "true").save(hdfs_predictions_path)

        predictions_tmp = os.path.join(driver_predictions_path,"predictions")
        if os.path.isdir(predictions_tmp) == True :
            shutil.rmtree(predictions_tmp)
            print("predictions_tmp is removed !!")

        os.system("hadoop fs -copyToLocal %s %s " % (hdfs_predictions_path,driver_predictions_path))
        
        driver_predictions_path = os.path.join(driver_predictions_path,"best_model_predictions")
        if os.path.isdir(driver_predictions_path) == False :
            os.makedirs(driver_predictions_path)
            print("best_model_predictions is created !!")

        os.system("cat %s > %s" % (os.path.join(predictions_tmp,"p*")\
                                  ,os.path.join(driver_predictions_path, \
                                               ("run_"+ self.best_model_name + "_with"+ str(self.images_num_sparkdl_run_best)+"images.csv") )))
        
        shutil.rmtree(predictions_tmp)
        print("predictions_tmp is removed !!")

if __name__ == "__main__":

    start_time = time.time()

    ip = Predictor()
    test_images = ip.read_images()
    featurizer_name, classifier = ip.read_model()
    predictions = ip.predicate(featurizer_name, classifier, test_images)
    ip.save_predictions(predictions)

    execution_time = time.time() - start_time
    print("Execution time : " + str(execution_time))

sc.stop()

