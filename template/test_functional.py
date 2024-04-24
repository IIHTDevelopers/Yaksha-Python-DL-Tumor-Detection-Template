import os

from TestUtils import TestUtils


class FuctionalTests():
    def __init__(self, model_path, org_mask, pred_mask, jaccard_cofficient, dice_score):
        self.test_model_exists(model_path)
        self.original_predicted_mask_same_size(org_mask, pred_mask)
        self.metric_value_test(jaccard_cofficient, dice_score)

    def test_model_exists(self, model_path):
        test_obj = TestUtils()
        model_exists = os.path.exists(model_path)
        print(model_exists)
        if model_exists:
            passed  = True
            test_obj.yakshaAssert("TestModelExists", True, "exception")
            print("TestModelExists= Passed")
        else:
            passed = False
            test_obj.yakshaAssert("TestModelExists", False, "exception")
            print("TestModelExists = Failed")
        assert passed
    
    
    def original_predicted_mask_same_size(self, org_mask, pred_mask):
        test_obj = TestUtils()

        if (pred_mask.size == org_mask.size):
            passed = True
            test_obj.yakshaAssert("OriginalPredictedMaskSameSize", True, "functional")
            print("OriginalPredictedMaskSameSize = Passed")
        else:
            passed = False
            test_obj.yakshaAssert("OriginalPredictedMaskSameSize", False, "functional")
            print("OriginalPredictedMaskSameSize = Failed")
        assert passed
    

    def metric_value_test(self, jaccard_cofficient, dice_score):
        test_obj = TestUtils()
        
        if jaccard_cofficient >= 0.8 and dice_score >= 0.8:
            passed = True
            test_obj.yakshaAssert("ModelEvaluation", True, "functional")
            print("ModelEvaluation = Passed\nModel is good")
        else:
            passed = False
            test_obj.yakshaAssert("ModelEvaluation", False, "functional")
            print("ModelEvaluation = Failed\nModel is not good")
        assert passed



def server_tests(model_path, org_mask, pred_mask, jaccard_cofficient, dice_score):
    FuctionalTests(model_path, org_mask, pred_mask, jaccard_cofficient, dice_score)
