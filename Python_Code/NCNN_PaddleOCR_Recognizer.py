import ncnn
import cv2
import numpy as np
import os

class PPOCRv4Recognizer:
    # Initializes the NCNN text recognition model with character dictionary and PP-OCRv4 parameters
    def __init__(self, param_path, bin_path, char_dict_path="en_dict.txt"):
        # Load NCNN model
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        # PP-OCRv4 parameters - DO NOT CHANGE THESE VALUES
        self.rec_image_shape = [3, 48, 320]  # [C, H, W]
        self.mean = [127.5, 127.5, 127.5]
        self.std = [1/(0.5 * 255), 1/(0.5 * 255), 1/(0.5 * 255)]

        # Load character dictionary with improved space handling
        self.character, self.char_to_idx, self.idx_to_char = self._load_dict_improved(char_dict_path)
        
        print("===== Recognizer Params =====")
        print(f"  rec_image_shape: {self.rec_image_shape}")
        print(f"  mean: {self.mean}")
        print(f"  std: {self.std}")
        print(f"  Dictionary size: {len(self.character)}")
        print(f"  First 5 chars: {self.character[:5] if len(self.character) >= 5 else self.character}")
        print(f"  Space at index: {self.character.index(' ') if ' ' in self.character else 'Not found'}")
        if 96 in self.idx_to_char:
            print(f"  Character at index 96: '{self.idx_to_char[96]}'")
        print("=============================")

    # Loads character dictionary from file with improved space handling and model compatibility fixes
    def _load_dict_improved(self, char_dict_path):
        """Load dictionary with improved space handling from organized version"""
        character_str = []
        
        if char_dict_path and os.path.exists(char_dict_path):
            print(f"Loading dictionary from: {char_dict_path}")
            try:
                with open(char_dict_path, "rb") as fin:
                    lines = fin.readlines()
                    for line in lines:
                        line = line.decode('utf-8').strip("\n").strip("\r\n")
                        character_str.append(line)
                print(f"Loaded {len(character_str)} characters from file")
            except Exception as e:
                print(f"Error loading dictionary: {e}")
                character_str = []
        else:
            # Use default PP-OCR English dict if file not found
            print("Using default character dict.")
            character_str = self._get_default_dict()
        
        # CRITICAL: Add special char handling from organized version
        # First, handle blank token
        if len(character_str) > 0 and character_str[0] != "blank":
            if "blank" in character_str:
                character_str.remove("blank")
            character_str.insert(0, "blank")
        elif len(character_str) == 0:
            character_str = ["blank"]
        
        # Check if space character exists, if not add it
        if " " not in character_str:
            print("Warning: Space character not found in dictionary, adding it")
            if len(character_str) > 1:
                character_str.insert(1, " ")
            else:
                character_str.append(" ")
        
        # Create mappings
        char_to_idx = {char: idx for idx, char in enumerate(character_str)}
        idx_to_char = {idx: char for idx, char in enumerate(character_str)}
        
        # IMPORTANT: Handle PP-OCRv4 English model expectations
        # Your en_dict.txt has 96 chars, so after adding blank it becomes 97
        expected_size = 97  # PP-OCRv4 English expects this
        
        if len(character_str) != expected_size:
            print(f"Warning: Dictionary size ({len(character_str)}) doesn't match expected ({expected_size})")
            
            # Special handling for 96-char dictionaries (common for PP-OCR models)
            if len(character_str) == 96:
                print("Dictionary has 96 chars. Checking if space needs to be at index 96...")
                # For some models, space is expected at index 96
                if " " in character_str:
                    current_space_idx = character_str.index(" ")
                    if current_space_idx != 96:
                        print(f"Space is at index {current_space_idx}, some models expect it at 96")
                else:
                    # Add space at the end if not present
                    print("Adding space at index 96 for model compatibility")
                    character_str.append(" ")
                    char_to_idx[" "] = 96
                    idx_to_char[96] = " "
            
            # Pad with unknown tokens if too small
            elif len(character_str) < expected_size:
                remaining = expected_size - len(character_str)
                print(f"Padding dictionary with {remaining} tokens")
                
                while len(character_str) < expected_size:
                    if len(character_str) == 96 and " " not in character_str:
                        # Ensure space is at index 96 if we're padding
                        character_str.append(" ")
                        char_to_idx[" "] = 96
                        idx_to_char[96] = " "
                        print("Added space at index 96 for model compatibility")
                    else:
                        unk_token = f"<UNK_{len(character_str)}>"
                        character_str.append(unk_token)
                        char_to_idx[unk_token] = len(character_str) - 1
                        idx_to_char[len(character_str) - 1] = unk_token
            
            # Truncate if too large
            elif len(character_str) > expected_size:
                print(f"Truncating dictionary to {expected_size} characters")
                character_str = character_str[:expected_size]
                # Rebuild mappings
                char_to_idx = {char: idx for idx, char in enumerate(character_str)}
                idx_to_char = {idx: char for idx, char in enumerate(character_str)}
        
        # CRITICAL FIX: PP-OCRv4 English model expects space at index 96!
        # But your dictionary has space at index 95 (after blank is added)
        # The model is predicting index 96, so we need to ensure something is there
        if len(character_str) == 96 and 96 not in idx_to_char:
            print("FIXING: Model expects character at index 96, adding space there")
            character_str.append(" ")  # Add another space at index 96
            char_to_idx[" "] = 96  # This will override the previous space mapping
            idx_to_char[96] = " "
            
            # Update the char_to_idx to handle duplicate space
            # Keep the original space at 95 as well
            if 95 in idx_to_char and idx_to_char[95] == " ":
                # We now have space at both 95 and 96
                print("Space now available at both index 95 and 96")
        
        # Final validation for space position
        space_indices = [i for i, char in enumerate(character_str) if char == " "]
        print(f"Final space positions: indices {space_indices}")
        
        # Debug: Check what's at critical indices
        if 96 in idx_to_char:
            print(f"Character at index 96: '{idx_to_char[96]}' (model is predicting this!)")
        if 95 in idx_to_char:
            print(f"Character at index 95: '{idx_to_char[95]}'")
        
        return character_str, char_to_idx, idx_to_char

    # Provides default character dictionary matching PP-OCR English format when no file is available
    def _get_default_dict(self):
        """Default dictionary matching your en_dict.txt format"""
        chars = []
        # Numbers
        chars.extend([str(i) for i in range(10)])
        # Symbols
        chars.extend([':', ';', '<', '=', '>', '?', '@'])
        # Uppercase letters
        chars.extend([chr(i) for i in range(ord('A'), ord('Z')+1)])
        # More symbols
        chars.extend(['[', '\\', ']', '^', '_', '`'])
        # Lowercase letters
        chars.extend([chr(i) for i in range(ord('a'), ord('z')+1)])
        # More symbols
        chars.extend(['{', '|', '}', '~', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/'])
        # Space at the end (matching your en_dict.txt)
        chars.append(' ')
        return chars

    # Resizes and normalizes input image to the required format for the recognition model
    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = int(imgH * ratio)
        resized_w = min(resized_w, imgW)
        resized_w = max(16, resized_w)
        resized_image = cv2.resize(img, (resized_w, imgH))
        if resized_w < imgW:
            pad = imgW - resized_w
            resized_image = cv2.copyMakeBorder(resized_image, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        # Convert to NCNN Mat and normalize
        mat_in = ncnn.Mat.from_pixels(resized_image, ncnn.Mat.PixelType.PIXEL_BGR, imgW, imgH)
        mat_in.substract_mean_normalize(self.mean, self.std)
        return mat_in

    # Decodes CTC output predictions into readable text using connectionist temporal classification algorithm
    def ctc_decode(self, preds_idx, preds_prob):
        """CTC decoding with space detection fix"""
        char_list = []
        conf_list = []
        
        # Count how many times index 96 is predicted (for debugging)
        count_96 = np.sum(preds_idx == 96)
        if count_96 > 0:
            print(f"[DEBUG] Index 96 predicted {count_96} times in this text")
        
        # Standard CTC decoding
        prev_idx = None
        for i in range(len(preds_idx)):
            cur_idx = int(preds_idx[i])
            cur_prob = float(preds_prob[i])
            
            # Skip blank token (index 0)
            if cur_idx == 0:
                prev_idx = cur_idx
                continue
            
            # Skip consecutive duplicates
            if prev_idx is not None and cur_idx == prev_idx:
                continue
            
            # Handle the character
            if cur_idx < len(self.character):
                char = self.character[cur_idx]
            elif cur_idx == 96 and 96 in self.idx_to_char:
                # Special handling for index 96
                char = self.idx_to_char[96]
            else:
                print(f"[WARNING] Index {cur_idx} out of range (dict size: {len(self.character)})")
                prev_idx = cur_idx
                continue
            
            char_list.append(char)
            conf_list.append(cur_prob)
            prev_idx = cur_idx
        
        text = ''.join(char_list)
        confidence = np.mean(conf_list) if conf_list else 0.0
        return text, confidence

    # Recognizes text from a list of cropped text region images using the NCNN model
    def recognize_text(self, img_list):
        if not img_list:
            return []
        results = []
        for idx, img in enumerate(img_list):
            try:
                mat_in = self.resize_norm_img(img)
                ex = self.net.create_extractor()
                ex.input("in0", mat_in)
                mat_out = ncnn.Mat()
                ex.extract("out0", mat_out)
                preds = np.array(mat_out)
                
                # Reshape if needed
                if preds.ndim == 1 and preds.size % len(self.character) == 0:
                    seq_len = preds.size // len(self.character)
                    preds = preds.reshape(seq_len, len(self.character))
                
                preds_idx = preds.argmax(axis=-1)
                preds_prob = preds.max(axis=-1)
                text, conf = self.ctc_decode(preds_idx, preds_prob)
                results.append((text, conf))
            except Exception as e:
                print(f"[ERROR][Recognize] Failed to process image {idx}: {str(e)}")
                results.append(("", 0.0))
        return results

    # Callable interface that allows the recognizer object to be used as a function
    def __call__(self, img_list):
        return self.recognize_text(img_list)

### Utility function ###

# Analyzes dictionary file structure to understand character mappings and indices
def analyze_dict_file(dict_path):
    """Analyze dictionary file to understand its structure"""
    print(f"\n=== Analyzing Dictionary File: {dict_path} ===")
    
    if not os.path.exists(dict_path):
        print(f"File not found: {dict_path}")
        return
    
    try:
        with open(dict_path, "rb") as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        chars = []
        for i, line in enumerate(lines):
            char = line.decode('utf-8').strip("\n").strip("\r\n")
            chars.append(char)
            
            # Show specific indices
            if i in [0, 1, 95, 96] or (i < 5) or (i >= len(lines) - 5):
                print(f"  [{i:3d}]: '{char}' (ord: {ord(char) if len(char) == 1 else 'multi-char'})")
        
        # Check for space
        if " " in chars:
            space_idx = chars.index(" ")
            print(f"\nSpace character found at index: {space_idx}")
        else:
            print("\nWARNING: No space character in dictionary!")
            
    except Exception as e:
        print(f"Error analyzing dictionary: {e}")

# Post-processes recognized text by optionally removing spaces and punctuation marks
def post_process_text(text, remove_spaces=False, remove_punctuation=False):
    """Post-process recognized text based on configuration"""
    if remove_spaces:
        text = text.replace(' ', '')
    
    if remove_punctuation:
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text  

# Displays image with detection boxes and recognized text overlaid with automatic scaling
def display_results_with_text(img, dt_boxes, rec_results, scores, window_name="OCR Results"):
    """
    Display image with detection boxes and recognized text
    Text size automatically scales based on image and box size
    """
    import cv2
    import numpy as np
    
    # Create a copy to draw on
    vis_img = img.copy()
    
    # Calculate base font scale based on image size
    img_height, img_width = img.shape[:2]
    
    for idx, (box, (text, conf), score) in enumerate(zip(dt_boxes, rec_results, scores)):
        # Draw the box
        box = box.astype(np.int32)
        
        # Auto-scale boundary thickness based on image size
        img_diagonal = np.sqrt(img_width**2 + img_height**2)
        boundary_thickness = max(2, int(img_diagonal / 600))  # Adjust divisor to control thickness
        boundary_thickness = min(boundary_thickness, 10)  # Cap at 8 pixels
        
        cv2.polylines(vis_img, [box], True, (0, 255, 0), boundary_thickness)
        
        # Calculate box dimensions for text scaling
        box_width = max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        )
        box_height = max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        )
        
        # Fixed font scale based on image size only (not box size)
        # All text will be same size, but scales with image resolution
        img_diagonal = np.sqrt(img_width**2 + img_height**2)
        font_scale = img_diagonal / 2000.0  # Adjust divisor to control text size
        font_scale = max(0.4, min(font_scale, 1.5))  # Clamp between 0.4 and 1.5
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(font_scale * 3.5))  # Thickness scales with font
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Find position above the box
        # Use top-left corner of box and move text above it
        x_min = int(np.min(box[:, 0]))
        y_min = int(np.min(box[:, 1]))
        
        # Position text above box with padding
        text_x = x_min
        text_y = y_min - 5  # 5 pixels above box
        
        # Make sure text doesn't go outside image bounds
        if text_y - text_height < 0:
            # If text would be outside top of image, put it inside the box
            text_y = y_min + text_height + 10
        
        if text_x + text_width > img_width:
            # If text is too wide, adjust x position
            text_x = max(0, img_width - text_width - 5)
        
        # Draw the text without background
        cv2.putText(vis_img, text, (text_x, text_y), font, 
                   font_scale, (237, 9, 173), thickness, cv2.LINE_AA)
        
        # Add confidence score next to text
        conf_text = f" ({conf:.2f})"
        conf_scale = font_scale
        conf_thickness = thickness
        # Position confidence right after the text
        conf_x = text_x + text_width
        cv2.putText(vis_img, conf_text, 
                   (conf_x, text_y), 
                   font, conf_scale, (44, 30, 247), conf_thickness, cv2.LINE_AA)
        
        # Print detection and recognition confidence
        print(f"Region {idx}: '{text}' (rec: {conf:.3f}, det: {score:.3f})")
    
    # Display the image with proper window sizing
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Make window resizable
    cv2.imshow(window_name, vis_img)
    
    # Resize window
    # Set window to 1280x960 or smaller if image is smaller
    window_width = min(1280, img_width)
    window_height = min(960, img_height)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    # First analyze dictionary
    dict_path = "en_dict.txt"
    if os.path.exists(dict_path):
        analyze_dict_file(dict_path)
    
    # Use detector and recognizer
    from NCNN_PaddleOCR_Detector import PPOCRv4Detector
    image_path = "OCR_PROJECT/PIC_SS2/1747279737511.png"
    
    detector = PPOCRv4Detector(
        "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_det_mobile.ncnn.param",
        "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_det_mobile.ncnn.bin"
    )
    
    recognizer = PPOCRv4Recognizer(
        "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_rec_mobile.ncnn.param",
        "/home/cira/en_PP-OCRv4/NCNN_server/PP-OCRv4_rec_mobile.ncnn.bin",
        "en_dict.txt"
    )
    
    img_bgr, dt_boxes, scores = detector.detect(image_path)
    if img_bgr is not None and len(dt_boxes) > 0:
        cropped_images, _ = detector.crop_all_text_regions(img_bgr, dt_boxes)
        rec_results = recognizer(cropped_images)
        
        # Post-Process if needed
        processed_results = []
        for text, conf in rec_results:
            processed_text = post_process_text(
                text, 
                remove_spaces=False,  
                remove_punctuation=False
            )
            processed_results.append((processed_text, conf))
        
        # Display results
        display_results_with_text(img_bgr, dt_boxes, processed_results, scores)
        
        # Also print to console
        print("\n=== Recognition Results ===")
        for i, ((text, conf), score) in enumerate(zip(processed_results, scores)):
            print(f"Region {i}: '{text}'")
            print(f"  Recognition confidence: {conf:.3f}")
            print(f"  Detection confidence: {score:.3f}")
    else:
        print("No text detected in image.")