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
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, LinearSVC
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

class ImageClassifier:

    def __init__(self):

        self.classifier_pset = {
            "logistic_regression":   ["LOGISTIC_REGPARAM_MIN", "LOGISTIC_REGPARAM_MAX", "LOGISTIC_REGPARAM_NUM",
                                        "LOGISTIC_MAXITER_MIN", "LOGISTIC_MAXITER_MAX", "LOGISTIC_MAXITER_NUM",
                                        "LOGISTIC_ELASTICNETPARAM_MIN", "LOGISTIC_ELASTICNETPARAM_MAX", "LOGISTIC_ELASTICNETPARAM_NUM"],
            "random_forest":         ["RANDOMFOREST_NUMTREES_MIN", "RANDOMFOREST_NUMTREES_MAX", "RANDOMFOREST_NUMTREES_NUM"],
            "linear_svc":            ["LSVC_REGPARAM_MIN", "LSVC_REGPARAM_MAX", "LSVC_REGPARAM_NUM",
                                        "LSVC_MAXITER_MIN", "LSVC_MAXITER_MAX", "LSVC_MAXITER_NUM"]
        }

        self.classifier_params = {}
        self.classifier_params_range = {}
        self.data_params = {}
        self.featurizers = []
        self.classifiers = []
        self.train_pipeline = Pipeline()
        self.best_model = {}
        self.featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features")

    def read_params(self):

        ft = os.environ["FEATURIZER_NAMES"]
        featurizers = ft.split(' ')

        for featurizer in featurizers:
            if featurizer == "inception_v3":
                self.featurizers.append("InceptionV3")
            if featurizer == "vgg16":
                self.featurizers.append("VGG16")
            if featurizer == "vgg19":
                self.featurizers.append("VGG19")
            if featurizer == "xception":
                self.featurizers.append("Xception")
            if featurizer == "resnet50":
                self.featurizers.append("ResNet50")

        cs = os.environ["CLASSIFIER_NAMES"]
        self.classifiers = cs.split(' ')

        print("Featurizers : ")
        print(self.featurizers)
        print("Classifiers : ")
        print(self.classifiers)

        for classifier in self.classifiers:
            for name, params in (self.classifier_pset).iteritems():
                if classifier == name:
                    for param in params:
                        self.classifier_params[param] = float(os.environ[param])

        self.data_params ["images_num"] = int(os.environ["IMAGES_NUM_SPARKDL"]) 
        self.data_params ["train_ratio"] = float(os.environ["TRAIN_DATA_RATIO"])
        self.data_params ["test_ratio"] = float(os.environ["TEST_DATA_RATIO"])
        self.data_params ["fold_num"] = int(os.environ["FOLD_NUM"]) 
        
        self.data_params ["img_dir"] = os.environ["IMG_DIR"]
        self.data_params ["run_dir"] = os.environ["RUN_DIR"]
        self.data_params ["hdfs_path"] = os.environ["HDFS_PATH"]
        self.data_params ["models_path"] = os.environ["MODELS_PATH"]

    def prepare_data(self):

        images_num = self.data_params["images_num"]
        train_ratio = self.data_params["train_ratio"]
        test_ratio = self.data_params["test_ratio"]
        img_dir = self.data_params["img_dir"] 
        run_dir = self.data_params["run_dir"] 
        hdfs_path = self.data_params["hdfs_path"] 

        class_paths = {}
        class_paths_run = {}
        class_img_nums = {}

        dirlist = [ item for item in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, item)) ]

        if os.path.isdir(run_dir) == True :
            shutil.rmtree(run_dir)

        for directory in dirlist :
            class_paths[directory] = os.path.join(img_dir,directory)
            class_img_nums[directory] = len([name for name in os.listdir(class_paths[directory]) if os.path.isfile(os.path.join(class_paths[directory], name))])        

            if images_num > class_img_nums[directory] or images_num == 0:
                files = os.listdir(class_paths[directory])
            else :
                files = os.listdir(class_paths[directory])[:images_num]
            
            for eachfilename in files:
                src = os.path.join(class_paths[directory],eachfilename)
                dst = os.path.join(run_dir,directory)
                if not os.path.exists(dst):
                    os.makedirs(dst)
                retrn_val = shutil.copy(src, os.path.join(dst,eachfilename))

        hdfs_path_run = os.path.join(hdfs_path, "run")
        exists = os.system("hadoop fs -test -d %s" % (hdfs_path_run))
        if exists == 0 :
            exists = os.system("hadoop fs -rm -r -skipTrash %s" % (hdfs_path_run))
        os.system("hadoop fs -copyFromLocal %s %s" % (run_dir, hdfs_path))

        for directory in dirlist :
            class_paths_run[directory] = os.path.join(hdfs_path,"run",directory)

        class0_df = readImages(class_paths_run[dirlist[0]]).withColumn("label", lit(0))
        train_df, test_df = class0_df.randomSplit([train_ratio, test_ratio], seed=1234)
        for class_label in range(1,len(dirlist)) :
            classi_df = readImages(class_paths_run[dirlist[class_label]]).withColumn("label", lit(class_label))
            classi_train, classi_test = classi_df.randomSplit([train_ratio, test_ratio], seed=1234)
            train_df = train_df.unionAll(classi_train)
            test_df = test_df.unionAll(classi_test)

        return train_df, test_df

    def compute_logistic_regression(self):

        regParams = np.linspace(self.classifier_params["LOGISTIC_REGPARAM_MIN"],
                                self.classifier_params["LOGISTIC_REGPARAM_MAX"],
                                self.classifier_params["LOGISTIC_REGPARAM_NUM"])

        maxIters = np.linspace(self.classifier_params["LOGISTIC_MAXITER_MIN"],
                                self.classifier_params["LOGISTIC_MAXITER_MAX"],
                                self.classifier_params["LOGISTIC_MAXITER_NUM"])

        elasticNetParams = np.linspace(self.classifier_params["LOGISTIC_ELASTICNETPARAM_MIN"], 
                                        self.classifier_params["LOGISTIC_ELASTICNETPARAM_MAX"], 
                                        self.classifier_params["LOGISTIC_ELASTICNETPARAM_NUM"]) 

        self.classifier_params_range["logistic_regParams"] = regParams
        self.classifier_params_range["logistic_maxIters"] = maxIters
        self.classifier_params_range["logistic_elasticNetParams"] = elasticNetParams

        lr = LogisticRegression(labelCol="label",featuresCol="features")
        stages = [self.featurizer, lr]
        paramGrid = ParamGridBuilder() \
            .baseOn([self.train_pipeline.stages, stages]) \
            .addGrid(self.featurizer.modelName, self.featurizers) \
            .addGrid(lr.maxIter, maxIters) \
            .addGrid(lr.regParam, regParams) \
            .addGrid(lr.elasticNetParam, elasticNetParams)\
            .build()
        
        return paramGrid

    def compute_random_forest(self):

        numTreess = np.linspace(self.classifier_params["RANDOMFOREST_NUMTREES_MIN"],
                                self.classifier_params["RANDOMFOREST_NUMTREES_MAX"],
                                self.classifier_params["RANDOMFOREST_NUMTREES_NUM"])
    
        self.classifier_params_range["randomForest_numTreess"] = numTreess

        rf = RandomForestClassifier(labelCol="label",featuresCol="features")
        stages = [self.featurizer, rf]
        paramGrid = ParamGridBuilder() \
            .baseOn([self.train_pipeline.stages, stages]) \
            .addGrid(self.featurizer.modelName, self.featurizers) \
            .addGrid(rf.numTrees, numTreess) \
            .build()

        return paramGrid

    def compute_decision_tree(self):

        dt = DecisionTreeClassifier(labelCol="label",featuresCol="features")
        stages = [self.featurizer, dt]
        paramGrid = ParamGridBuilder() \
            .baseOn([self.train_pipeline.stages, stages]) \
            .addGrid(self.featurizer.modelName, self.featurizers) \
            .build()

        return paramGrid

    def compute_linear_svc(self):

        maxIters = np.linspace(self.classifier_params["LSVC_MAXITER_MIN"], 
                                self.classifier_params["LSVC_MAXITER_MAX"], 
                                self.classifier_params["LSVC_MAXITER_NUM"])

        regParams = np.linspace(self.classifier_params["LSVC_REGPARAM_MIN"], 
                                self.classifier_params["LSVC_REGPARAM_MAX"], 
                                self.classifier_params["LSVC_REGPARAM_NUM"])
        
        self.classifier_params_range["linearSVC_maxIters"] = maxIters
        self.classifier_params_range["linearSVC_regParams"] = regParams

        lSVC = LinearSVC(labelCol="label",featuresCol="features")
        stages = [self.featurizer, lSVC]
        paramGrid = ParamGridBuilder() \
            .baseOn([self.train_pipeline.stages, stages]) \
            .addGrid(self.featurizer.modelName, self.featurizers) \
            .addGrid(lSVC.maxIter, maxIters) \
            .addGrid(lSVC.regParam, regParams) \
            .build()

        return paramGrid


    def train_model_with_HP(self, train_data, test_data):
        
        paramGridAll = [] 

        for classifier in self.classifiers :

            if classifier == "logistic_regression" :
                grid = self.compute_logistic_regression()

            elif classifier == "random_forest" :
                grid = self.compute_random_forest()

            elif classifier == "decision_tree" :
                grid = self.compute_decision_tree()
            
            elif classifier == "linear_svc" :
                grid = self.compute_linear_svc()

            paramGridAll.extend(grid)


        fold_num = self.data_params["fold_num"] 
        cv = CrossValidator(estimator=self.train_pipeline,
                            estimatorParamMaps=paramGridAll,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=fold_num) 

        cvModel = cv.fit(train_data)
        bestPipeline = cvModel.bestModel
        predictions = bestPipeline.transform(test_data)

        featurizer = bestPipeline.stages[0].extractParamMap().values()

        for ff in featurizer:
            if ff != "image" and ff != "features" :
                featurizer = ff

        self.best_model["best_featurizer"] = featurizer
        classifier = type(bestPipeline.stages[1]).__name__ 
        classifier_params = bestPipeline.stages[1]._java_obj
    
        if "LogisticRegression" in classifier:
            self.best_model["best_classifier"] = "LogisticRegression"
            self.best_model["best_logistic_regParam"] = classifier_params.getRegParam()
            self.best_model["best_logistic_maxIter"] = classifier_params.getMaxIter()
            self.best_model["best_logistic_elasticNetParam"] = classifier_params.getElasticNetParam()

        if "RandomForest" in classifier:
            self.best_model["best_classifier"] = "RandomForest"
            self.best_model["best_randomForest_numTrees"] = classifier_params.getNumTrees()

        if "DecisionTree" in classifier:
            self.best_model["best_classifier"] = "DecisionTree"

        if "LinearSVC" in classifier:
            self.best_model["best_classifier"] = "LinearSVC"
            self.best_model["best_lSVC_regParam"] = classifier_params.getRegParam()
            self.best_model["best_lSVC_maxIter"] = classifier_params.getMaxIter()

        best_classifier = bestPipeline.stages[1]
        best_classifier.write().overwrite().save(os.path.join(self.data_params["models_path"], self.best_model["best_featurizer"] + "_" + self.best_model["best_classifier"]))

        return predictions


    def calculate_accuracy(self, predictions):

        results = predictions.select(['prediction', 'label'])
        labels = predictions.select(['label'])
        labels_list = labels.collect()
        predictions_values = predictions.select(['prediction'])
        predictions_list = predictions_values.collect()

        labels = np.asarray(labels_list)
        predictions_values = np.asarray(predictions_list)

        mask_correct = (labels == predictions_values)
        correct_predictions = predictions_values[mask_correct]
        TP = np.sum(correct_predictions == 1)
        TN = np.sum(correct_predictions == 0)

        mask_incorrect = (labels != predictions_values)
        incorrect_predictions = predictions_values[mask_incorrect]
        FP = np.sum(incorrect_predictions == 1)
        FN = np.sum(incorrect_predictions == 0)

        accuracy = (TP+TN)/float(TP+TN+FP+FN) if float(TP+TN+FP+FN) else 0
        precision = TP/float(TP+FP) if float(TP+FP) else 0
        recall = TP/float(TP+FN) if float(TP+FN) else 0
        f1_measure = (2*precision*recall)/float(recall+precision) if float(recall+precision) else 0

        aucpr,auroc = 0,0
        if "probability" in predictions.columns :
            results = predictions.select(['probability', 'label'])
            results_collect = results.collect()
            results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
            scoreAndLabels = sc.parallelize(results_list)
            metrics = BinaryClassificationMetrics(scoreAndLabels)
            aucpr = metrics.areaUnderPR  
            auroc = metrics.areaUnderROC 

        return accuracy,precision,recall,f1_measure,aucpr,auroc

    def write_result(self,result,classifier_params_range,best_model) :

        columns = ["C1N","C2N","ExecutionTime(sec)","Accuracy","Precision","Recall","F1-measure","Aucpr","Auroc",
                   "TrainRatio","TestRatio","Featurizers","Classifiers"]
        columns.extend(classifier_params_range)
        columns.extend(best_model)

        with open("sparkdl_results.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(columns)
            wr.writerow(result)

if __name__ == "__main__":

    ic = ImageClassifier()
    ic.read_params()
    train_df, test_df = ic.prepare_data() 
    
    print("Training is Started")
    start_time = time.time()
    predictions = ic.train_model_with_HP(train_df, test_df)
    execution_time = time.time() - start_time
    print("Training is Finished and Model is saved.")

    accuracy,precision,recall,f1_measure,aucpr,auroc = ic.calculate_accuracy(predictions)
    print("Accuracy is calculated !!!")

    results = [ic.data_params["images_num"],ic.data_params["images_num"],
               execution_time,accuracy,precision,recall,f1_measure,aucpr,auroc,
               ic.data_params["train_ratio"],ic.data_params["test_ratio"],
               (ic.featurizers), (ic.classifiers)]
    results.extend(ic.classifier_params_range.values())
    results.extend(ic.best_model.values())
    ic.write_result(results,ic.classifier_params_range.keys(),ic.best_model.keys())

sc.stop()
