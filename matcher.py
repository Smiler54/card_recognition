import cv2
import numpy as np

class Matcher:
    def __init__(self, main_image, template_image):
        self.main_image = main_image
        self.template_image = template_image

    def match(self, method=cv2.TM_CCOEFF_NORMED):
        if self.main_image is None or self.template_image is None:
            raise ValueError("Images must not be None")

        if self.main_image.shape[0] < self.template_image.shape[0] or \
            self.main_image.shape[1] < self.template_image.shape[1]:
            return ([0, 0], 0)

        if len(self.main_image.shape) == 2 or self.main_image.shape[2] == 1:
            # Grayscale
            result = cv2.matchTemplate(self.main_image, self.template_image, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_pos = min_loc
                match_score = min_val
            else:
                match_pos = max_loc
                match_score = max_val
            return (match_pos, match_score)
        else:
            # Color: match per channel and take the best overall
            best_score = None
            best_pos = None
            for c in range(self.main_image.shape[2]):
                main_c = self.main_image[:, :, c]
                template_c = self.template_image[:, :, c]
                result = cv2.matchTemplate(main_c, template_c, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    score = min_val
                    pos = min_loc
                    is_better = best_score is None or score < best_score
                else:
                    score = max_val
                    pos = max_loc
                    is_better = best_score is None or score > best_score
                if is_better:
                    best_score = score
                    best_pos = pos
            return (best_pos, best_score)