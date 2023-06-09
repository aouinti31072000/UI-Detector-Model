# YOLO options
YOLO_V3_WEIGHTS             = "model_data/yolov3.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolov3-tiny.weights"
YOLO_CUSTOM_WEIGHTS         = False # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 640
YOLO_ANCHORS                = [[[49, 46], [123,  61], [127, 151]],
                               [[233, 161], [262, 268], [368, 283]],
                            [[438,  361], [471, 399], [521, 455]]]
# Train options
TRAIN_YOLO_TINY             = False
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)

TRAIN_CLASSES               = "/content/UI-Detector-Model/wireframe1/UI-Detector-3/train/_classes.txt"
TRAIN_ANNOT_PATH            = "/content/UI-Detector-Model/wireframe1/UI-Detector-3/train/_annotations.txt"
TRAIN_LOGDIR                = "/content/UI-Detector-Model/log"
TRAIN_CHECKPOINTS_FOLDER    = "/content/UI-Detector-Model/checkpoints"
TRAIN_MODEL_NAME            = f"yolov3_custom"
TRAIN_LOAD_IMAGES_TO_RAM    = True # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 4
TRAIN_INPUT_SIZE            = 640
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 100

# TEST options
TEST_ANNOT_PATH             = "/content/UI-Detector-Model/wireframe1/UI-Detector-3/test/_annotations.txt"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 640
TEST_DATA_AUG               = False
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45

if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32]    
    # YOLO_ANCHORS            = [[[23, 27],  [37, 58],   [81,  82]], # this line can be uncommented for default coco weights
    YOLO_ANCHORS            =  [[[153, 120],[236 ,150],[250, 224]],
                                [[277, 253],[352, 348], [435, 441]] ]         
# YOLO_ANCHORS            = [
#                                [[30,  61], [62,   45], [59,  119]],
#                                [[116, 90], [156, 198], [373, 326]]]
