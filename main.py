import geoportal_classifier
from las_file_manager import PointCloudManager
import time

# start_time = time.time()
# model = geoportal_classifier.GeoportalClassifier()
# WMII = PointCloudManager("./../punkty/centrum.las")
# model.train_model(WMII, "./../feature/features_centrum.parquet", read=False)
# model.save("./../modele/centrum.joblib")
# model.show_feature_importances()
# print("time:", time.time() - start_time)

model = geoportal_classifier.GeoportalClassifier()
WMII = PointCloudManager("./../punkty/centrum_test/78924_1439098_N-34-78-C-a-3-1-3-3.laz")
model.load("./../modele/centrum.joblib")
y_test, y_pred = model.classify(WMII)
WMII.color_classified_points()
WMII.visualize()
model.show_feature_importances()
geoportal_classifier.show_classification_report(y_test, y_pred)
