import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.measure import label, regionprops, perimeter
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray, label2rgb
from skimage.util import img_as_ubyte
import math



class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.mask, self.rgb_masked, self.contour, self.alpha_ch = self._mask()
        self.height, self.width = self.mask.shape

    def _mask(self):
        alpha_ch = self.image[..., 3]
        rgb_image = self.image[..., :3]
        ret, thr = cv2.threshold(alpha_ch, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=len)
        part_mask = np.zeros_like(rgb_image)
        cv2.drawContours(part_mask, [contour], -1, (255, 255, 255), -1)
        rgb_masked = np.where(part_mask == 255, rgb_image, 0)
        return thr, rgb_masked, contour, alpha_ch

class ShapeAnalyzer:
    def __init__(self, processor):
        self.processor = processor
        self.contour = processor.contour
        self.label_img = label(processor.mask)
        self.props = max(regionprops(self.label_img), key=lambda prop: prop.area)

    def eccentricity_from_moments(self, moments):
        '''Calculate eccentricity from moments'''
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11'] ** 2
        # Calculation of eccentricity from moments
        numerator = mu20 + mu02 + np.sqrt(4 * mu11 + (mu20 - mu02) ** 2)
        denominator = mu20 + mu02 - np.sqrt(4 * mu11 + (mu20 - mu02) ** 2)
        if denominator == 0:
            return 0
        eccentricity = np.sqrt(1 - (denominator / numerator))
        return eccentricity

    def eccentricity_from_ellipse(self, contour):
        '''Calculate eccentricity based on the fitted ellipse'''
        if len(contour) < 5:
            return 0  # Need at least 5 points to fit an ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        a = max(MA, ma) / 2  # Semi-major axis
        b = min(MA, ma) / 2  # Semi-minor axis
        eccentricity = np.sqrt(1 - (b ** 2) / (a ** 2))
        return eccentricity

    def aspect_ratio_from_ellipse(self, contour):
        '''Calculate aspect ratio from the fitted ellipse'''
        if len(contour) < 5:
            return 1  # Default to 1 if we cannot fit an ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        aspect_ratio = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 1
        return aspect_ratio

    def analyze(self):
        hull = convex_hull_image(self.processor.mask)
        hull_perim = perimeter(hull)
        moments = cv2.moments(self.contour)
        
        shape_features = {
            'aspect_ratio': self.aspect_ratio_from_ellipse(self.contour),
            'solidity': self.props.solidity,
            'convexity': hull_perim / self.props.perimeter,
            'circularity_cioni': (4 * np.pi * self.props.area) / (self.props.perimeter ** 2),
            'circularity_dellino': self.props.perimeter / (2 * np.sqrt(np.pi * self.props.area)),
            'rectangularity': self.props.perimeter / (2 * (self.props.major_axis_length + self.props.minor_axis_length)),
            'compactness': self.props.area / (self.props.major_axis_length * self.props.minor_axis_length),
            'elongation': (self.props.feret_diameter_max ** 2) / self.props.area,
            'roundness': 4 * self.props.area / (np.pi * (self.props.feret_diameter_max ** 2))
        }

        return shape_features

class TextureAnalyzer:
    def __init__(self, processor, props):
        self.processor = processor
        self.props = props
        self.gray_image = rgb2gray(processor.rgb_masked)
        self.ubyte_image = img_as_ubyte(self.gray_image)
        self.patch_size = int(self.props.major_axis_length / 10)
        self.step = int(self.props.major_axis_length / 20)
        self.thetas = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    def _extract_patches(self):
        patches = []
        step_size = max(int(self.props.major_axis_length / 5), 1)
        y_positions = range(0, self.gray_image.shape[0] - self.patch_size, step_size)
        x_positions = range(0, self.gray_image.shape[1] - self.patch_size, step_size)
        for y in y_positions:
            for x in x_positions:
                patch = self.ubyte_image[y:y+self.patch_size, x:x+self.patch_size]
                if np.any(patch):
                    patches.append(patch)
        return patches
    
    def analyze(self):
        patches = self._extract_patches()
        features = {
            'contrast': [],
            'dissimilarity': [],
            'homogeneity': [],
            'ASM': [],
            'energy': [],
            'correlation': [],
        }
        for patch in patches:
            glcm = graycomatrix(patch, distances=[1], angles=self.thetas, levels=256, symmetric=True, normed=True)
            for prop in features.keys():
                features[prop].append(graycoprops(glcm, prop)[0, 0])
        
        # Compute averages
        avg_features = {f'{k}_avg': np.mean(v) for k, v in features.items()}
        return avg_features

class ColorAnalyzer:
    def __init__(self, processor):
        self.processor = processor

    def analyze(self):
        rgb_image = self.processor.rgb_masked
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        color_features = {}
        for i, color in enumerate(['red', 'green', 'blue']):
            channel = rgb_image[..., i].flatten()
            channel_pixels = channel[channel > 0]
            color_features[f'{color}_mean'] = np.mean(channel_pixels)
            color_features[f'{color}_std'] = np.std(channel_pixels)
            color_features[f'{color}_mode'] = stats.mode(channel_pixels, axis=None)[0]
        
        for i, color in enumerate(['hue', 'saturation', 'value']):
            channel = hsv_image[..., i].flatten()
            channel_pixels = channel[channel > 0]
            color_features[f'HSV_{color}_mean'] = np.mean(channel_pixels)
            color_features[f'HSV_{color}_std'] = np.std(channel_pixels)
            color_features[f'HSV_{color}_mode'] = stats.mode(channel_pixels, axis=None)[0]
        
        return color_features
