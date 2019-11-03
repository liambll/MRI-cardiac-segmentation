# dicom-code-challenge

## Part 1: Parse the DICOM images and Contour Files
__Assumption:__
- The pipeline only parse DICOM images and i-contour files that are matched together. Files that do not have corresponding DICOM/i-contour counterpart are probably not useful at the moment and will not be parsed.
- The pipeline keep the parsed output in memory (for subsequent processing) instead of saving the output to storage
- There is no information on how a DICOM image should be matched with a contour file. I make a guess that for a patient, a __XYZ__.dcm DICOM image should be match with IM-0001-0__XYZ__-icontour-manual.txt contour file.

1. How did you verify that you are parsing the contours correctly?
I verify that the contours are parsed correctly by:
- writing unittest for parsing functions to ensure it has the intended output
- saving the generated outputs of DICOM image data and contour binary mask side-by-side and visually comparing the outputs.
There 96 matching DICOM-contour pairs. The generated outputs for these pairs are available here:
A sample output is shown below:
<img src="assets/SCD0000101_68.dcm.png" alt="" width="50%"><br/>

2. What changes did you make to the code, if any, in order to integrate it into our production code base? 
I made the following changes to the code:
- add in input validation for parsing functions, e.g check if file exists, check if a value can be parsed to float
- replace some hard-coded variables with global variables to avoid inconsistency
- perform minor refactoring to handle some strange logics:
  - In _parse_dicom_file_ function, check if a key exist in dcm object instead of trying to run the code and catch exception:
```python
    try:
        intercept = dcm.RescaleIntercept
    except AttributeError:
        intercept = 0.0
    try:
        slope = dcm.RescaleSlope
    except AttributeError:
        slope = 0.0
```
  
  - In _parse_dicom_file_ function, the below logic might miss out required data transformation in cases where intercept or slope is actually zero
```python
    if intercept != 0.0 and slope != 0.0:
                dcm_image = dcm_image*slope + intercept
```  

  - In _poly_to_mask_ function, it might be better to also draw the outline to avoid missing out pixels, altough this probably does not affect the model result. I did not make this change.
```python
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
```
- add unittests (alghough I do not have time to make it comprehensive)
- use logger to log relevant information and error when running the pipeline
 
## Part 2: Model training pipeline
1. Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?
I made the following chages to the pipelines built in Parts 1 to better streamline the pipeline built in Part 2:
- I keep patient/image ID as part of parsed output. For model training and evaluation, we might need to perfom train/test split at patient level instead of at image level. So, we would need to have patient/image ID to perform the split.

2. How do you/did you verify that the pipeline was working correctly?
- I added unittest for data_generator
- I also looked at log to make sure the data generated in each epoch is as intended

3. Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?
I would consider adding the below improvements to the pipeline:
- In case we have many observations to parse, I would use multithreadding to parse data
- In case the data is too large to store in memory, instead of parsing the whole dataset and keep it in memory, I can either:
    - save the parsed data to storage and use a data generator to load data as needed in each training step/epoch
    - (or) use a data generator to parse data as needed in each training step/epoch
- Add integration test