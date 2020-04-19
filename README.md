# Overview
The project aims to parse DICOM and corresponding Contour files and create models to segment out i-contour on 2D MRI slice:
- i-contour: inner contour that separates the left ventricular blood pool from the heart muscle (myocardium)
- o-contour: outer contour that defines the outer border of the left ventricular heart muscle

Relevant code files:
```
root
 |---pipelines
 |     |---pipeline_dicom_contour.py: This is the main pipeline that perform the parsing and prepare dataset
 |     |---dicom_contour_analysis.py: This is the main analysis for data visualization and image thresholding
 |     |---dicom_contour_cnn_model.py: This is the main pipeline for U-net model training
 |---utils
 |     |---parsing.py: contains provided file/data parsing functions
 |     |---dataset.py: contains ImageData class (for parsing) and data_geneartor (for iterating through data)
 |     |---image_processing.py: contains functions to create, transform and save images
 |     |---metrics.py: contains evaluation metric functions for segmentation task
 |---models
 |     |---u_net.py: contains U-Net model class
 |---tests
       |---unit
            |---utils
                 |---test_parsing.py: contains unittest for utils/parsing.py
                 |---test_dataset.py: contains unittest for utils/dataset.py
                 |---test_image_processing.py: contains unittest for utils/image_processing.py
                 |---test_metrics.py: contains unittest for utils/metrics.py
```

# DATA PROCESSING
- The pipeline only parses DICOM images and i-contour/o-contour files that are matched together. Files that do not have corresponding i-contour/o-contour counterpart are probably not useful at the moment and will not be parsed.
- An example of the updated visualization output is shown below (Red indicates i-contour or blood pool, Blue indicates o-contour or heart muscle):<br/>
<img src="assets/SCD0000101_59.dcm.png" alt="" width="25%"><br/>
 
# SEGMENTATION MODELS
__Otsu thresolding with convex hull post processing__

o-contour annotation are easier to obtain, so let's assume we can use o-contour as input for our computer vision model. A quick prototype is implemented to evaluate the feasibility of using Otsu thresholding approach and convex hull postprocessing to segment i-contour, given the MRI 2D slide and o-contour.

The segmentation result is available at: https://drive.google.com/open?id=12nfiO3uest38Im7x-Ft4bb6GveFfDIF4

A sample output is shown below (Red indicates annotated i-contour, Yellow indicates predicted i-contour):<br/>
<img src="assets/otsu_SCD0000101_99.dcm.png" alt="" width="25%"> <img src="assets/otsu_hull_SCD0000101_99.dcm.png" alt="" width="25%"><br/>
Left - Otsu thresholding, Right - Otsu thresolding with convex hull post processing.<br/>

To evaluate the segmentation result quantitatively, we look at Intersection over Unition (IoU) score and Dice (F1) score:

| Approach  | Mean (Std) of IoU Score | Mean (Std) of Dice Score |
| ------ | -------- | -------- |
| Otsu threshold | 0.769 (0.096) | 0.865 (0.068) |
| Otsu threshold + Convex Hull | 0.827 (0.099) | 0.901 (0.071) |


__U-Net Prototype__

Without o-contour mask, the segmentation of i-contour would be challenging using Otsu thresholding because other areas in an image have similar pixel intensity. This is where convolutional neural network comes to rescue.
A quick prototype is implemented to evaluate the feasibility of using U-Net for i-contour segmentation without o-segmentation. There are 5 patients with 96 matching image-icontour files. For this prototype, we simply train the U-Net model on the first 3 patients (for only 20 epoches), and validate the model on the remaining 2 patients.

The best model weight and segmentation result (without any postprocessing) are available at: https://drive.google.com/drive/folders/1PjrCZzGC2nGci1Fg7VZrbBY4v69ppa6G?usp=sharing

U-Net can potentially outperform Otsu thresholding approach, even without o-contour mask.<br/>
<img src="assets/otsu_hull_SCD0000101_99.dcm.png" alt="" width="25%"> <img src="assets/unet_SCD0000101_99.dcm.png" alt="" width="25%"><br/>
Left - Otsu thresolding + convex hull post-processing, Right - UNet on trainset

Intersection over Unition (IoU) score and Dice (F1) score of U-Net on train and validation sets:

| Dataset  | Mean (Std) of IoU Score | Mean (Std) of Dice Score |
| ------ | -------- | -------- |
| Train set | 0.911 (0.068) | 0.952 (0.041) |
| Validation set | 0.838 (0.097) | 0.908 (0.066) |

Further model tuning and post-processing will likely improve the i-contour mask prediction.
