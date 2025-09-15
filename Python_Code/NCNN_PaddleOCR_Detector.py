import ncnn
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """
    # Initializes the DBPostProcess with configuration parameters for text detection post-processing
    def __init__(self,
                 thresh,
                 box_thresh,
                 max_candidates,
                 unclip_ratio,
                 use_dilation,
                 score_mode,
                 box_type,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    # Extracts text bounding boxes from a binary probability map by finding contours and filtering them
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
                whose values are binarized as {0, 1}
        '''
        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return boxes, scores

    # Expands a polygon box outward using the unclip ratio to account for text boundaries
    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    # Gets the minimum area rectangle from a contour and returns ordered corner points
    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    # Calculates confidence score for a text box using fast bbox-based averaging method
    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    # Calculates confidence score for a text box using precise polygon-based averaging method
    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


class PPOCRv4Detector:
    # Initializes the NCNN neural network model with PP-OCRv4 detection parameters and settings
    def __init__(self, param_path, bin_path):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        # PP-OCRv4 specific parameters
        self.max_side_len        = 960
        self.det_db_thresh       = 0.3
        self.det_db_box_thresh   = 0.6
        self.det_db_unclip_ratio = 1.5
        self.use_dilation        = False
        
        # Normalization parameters
        self.mean = [0.485*255, 0.456*255, 0.406*255]
        self.std  = [1/(0.229*255), 1/(0.224*255), 1/(0.225*255)]
        
        # Initialize post processor with PP-OCRv4 parameters
        self.postprocessor = DBPostProcess(
            thresh=self.det_db_thresh,
            box_thresh=self.det_db_box_thresh,
            max_candidates=1000,
            unclip_ratio=self.det_db_unclip_ratio,
            use_dilation=self.use_dilation,
            score_mode="slow",
            box_type="quad"
        )
        
        # Debug: print detector params
        print("===== Detector Params =====")
        print(f"  max_side_len: {self.max_side_len}")
        print(f"  det_db_thresh: {self.det_db_thresh}")
        print(f"  det_db_box_thresh: {self.det_db_box_thresh}")
        print(f"  det_db_unclip_ratio: {self.det_db_unclip_ratio}")
        print(f"  use_dilation: {self.use_dilation}")
        print(f"  mean: {self.mean}")
        print(f"  std: {self.std}")

    # Resizes input image while maintaining aspect ratio and ensuring dimensions are multiples of 32
    def resize_image(self, img):
        h, w = img.shape[:2]
        ratio = 1.0
        if max(h, w) > self.max_side_len:
            ratio = float(self.max_side_len) / max(h, w)
            
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        
        if resize_w != w or resize_h != h:
            img = cv2.resize(img, (resize_w, resize_h))
        
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, (ratio_h, ratio_w)

    # Main detection function that processes an image and returns detected text bounding boxes
    def detect(self, image_path):
        if isinstance(image_path, str):
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"Failed to load image: {image_path}")
                return None, [], []
        else:
            img_bgr = image_path
            
        print(f"Original image shape: {img_bgr.shape}")
        ori_h, ori_w = img_bgr.shape[:2]

        # Resize image
        img_resized, (ratio_h, ratio_w) = self.resize_image(img_bgr)
        print(f"Resized image shape: {img_resized.shape}")
        print(f"Resize ratio: H={ratio_h:.3f}, W={ratio_w:.3f}")

        # Prepare input
        h, w = img_resized.shape[:2]
        mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR, w, h)
        mat_in.substract_mean_normalize(self.mean, self.std)
        print("Applied mean/std normalization.")

        # Run inference
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        mat_out = ncnn.Mat()
        ex.extract("out0", mat_out)
        
        # Convert output to numpy
        arr = np.array(mat_out)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        print(f"[DEBUG][Detect] Output shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")

        # Post-process
        prob_map = arr
        binary_map = prob_map > self.det_db_thresh
        print(f"Binary map > thresh ({self.det_db_thresh}): {np.sum(binary_map)} positive pixels")

        # Apply dilation if needed
        if self.use_dilation:
            kernel = np.array([[1, 1], [1, 1]]).astype(np.uint8)
            binary_map = cv2.dilate(binary_map.astype(np.uint8), kernel, iterations=1)
        
        # Get boxes and scores
        boxes, scores = self.postprocessor.boxes_from_bitmap(prob_map, binary_map, ori_w, ori_h)
        
        print(f"Detected {len(boxes)} text boxes.")

        # Convert to numpy array and filter
        if len(boxes) > 0:
            dt_boxes_np = self.filter_tag_det_res(boxes, ori_h, ori_w)
            
            # Sort boxes by position (top to bottom, left to right)
            if len(dt_boxes_np) > 0:
                # First sort by y-coordinate, then by x-coordinate
                sorted_indices = np.lexsort([
                    dt_boxes_np[:, 0, 0],  # x-coordinates
                    dt_boxes_np[:, 0, 1]   # y-coordinates
                ])
                dt_boxes_np = dt_boxes_np[sorted_indices]
                
                # Apply same sorting to scores
                if len(scores) == len(dt_boxes_np):
                    scores = [scores[i] for i in sorted_indices]
        else:
            dt_boxes_np = np.array([])
            scores = []

        # Debug output
        for i, box in enumerate(dt_boxes_np):
            print(f" Box {i}: {box.tolist()}")

        if len(scores):
            print(f"Box scores: min={np.min(scores):.3f} max={np.max(scores):.3f} mean={np.mean(scores):.3f}")

        return img_bgr, dt_boxes_np, scores

    # Filters and validates detected text boxes by removing boxes that are too small or invalid
    def filter_tag_det_res(self, dt_boxes, image_h, image_w):
        img_height, img_width = image_h, image_w
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    # Orders four corner points of a quadrilateral in clockwise order starting from top-left
    def order_points_clockwise(self, pts):
        """
        Sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype=np.float32)
        return rect

    # Clips bounding box coordinates to ensure they stay within image boundaries
    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    # Crops a single text region from image using perspective transformation to get a rectangular crop
    def crop_text_region(self, img, box):
        """
        Crop a text region from the image using perspective transformation.
        """
        box = box.astype(np.float32)
        
        # Calculate width and height of the box
        width = int(max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        ))
        height = int(max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        ))
        
        # Skip invalid boxes
        if width <= 0 or height <= 0:
            return None, (0, 0)
        
        # Define destination points for perspective transform
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], 
                          dtype=np.float32)
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(box, dst_pts)
        
        # Apply perspective transformation
        dst = cv2.warpPerspective(img, M, (width, height),
                                borderMode=cv2.BORDER_REPLICATE,
                                flags=cv2.INTER_CUBIC)
        
        return dst, (width, height)

    # Crops all detected text regions from the image and returns them as a list of cropped images
    def crop_all_text_regions(self, img, dt_boxes):
        """
        Crop all detected text regions from the image.
        """
        cropped_images = []
        dimensions = []
        
        for box in dt_boxes:
            cropped_img, (w, h) = self.crop_text_region(img, box)
            if cropped_img is not None:
                cropped_images.append(cropped_img)
                dimensions.append((w, h))
        
        return cropped_images, dimensions

# NCNN Model path
DETECT_PARAM = "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_det_mobile.ncnn.param"
DETECT_BIN   = "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_det_mobile.ncnn.bin"

#DETECT_PARAM = "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_det_server.ncnn.param"
#DETECT_BIN   = "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_det_server.ncnn.bin"

# main
if __name__ == "__main__":
    image_path = "OCR_PROJECT/PIC_SS2/1747279737511.png"

    detector = PPOCRv4Detector(DETECT_PARAM, DETECT_BIN)
    img_bgr, dt_boxes, scores = detector.detect(image_path)

    if img_bgr is None:
        print("Image not found or failed to load.")
        exit()

    if dt_boxes is None or len(dt_boxes) == 0:
        print("No text detected.")
        cv2.imwrite("detection_result.png", img_bgr)
        exit()

    # Crop all regions
    print("Saving crops and drawing boxes...")
    cropped_images, dimensions = detector.crop_all_text_regions(img_bgr, dt_boxes)
    
    # Save crops
    for idx, (crop, score) in enumerate(zip(cropped_images, scores)):
        #cv2.imwrite(f"debug_crop{idx:02d}.png", crop)
        print(f"Saved debug_crop{idx:02d}.png | conf {score:.3f}")

    # Draw boxes with confidence scores
    vis = img_bgr.copy()
    for box, score in zip(dt_boxes, scores):
        # Draw box
        cv2.polylines(vis, [box.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Add confidence score at top-left corner
        x_min = int(np.min(box[:, 0]))
        y_min = int(np.min(box[:, 1]))
        cv2.putText(vis, f"{score:.2f}", (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("detection_result2.png", vis)
    print(f"Detection done. Saved {len(cropped_images)} crops and detection_result.png")