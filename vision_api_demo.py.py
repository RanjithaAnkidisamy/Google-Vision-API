import os, io
from google.cloud import vision
import pandas as pd
from numpy import random



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'servicetokenjson.json'

client = vision.ImageAnnotatorClient()


def faces(content):
    image = vision.types.Image(content=content)
    response1 = client.face_detection(image=image)
    faceAnnotations = response1.face_annotations

    #likehood = ('Unknown', 'Very Unlikely', 'Unlikely', 'Possibly', 'Likely', 'Very Likely')
    

    print('Faces:')
    face = faceAnnotations[0]
        
    df = pd.DataFrame(columns=['Angry', 'Joy', 'Sorrow','Surprised','Headwear','Exposed','Blurred','Confidence'])
    
    df = df.append(
            dict(
                Angry=face.anger_likelihood*20,
                Joy=face.joy_likelihood*20,
                Sorrow=face.sorrow_likelihood*20,
                Surprised=face.surprise_likelihood*20,
                Headwear=face.headwear_likelihood*20,
                Exposed=face.under_exposed_likelihood*20,
                Blurred=face.blurred_likelihood*20,
                Confidence=round(face.detection_confidence,2)*100
                
            ), ignore_index=True)
        
    print(df)





def labels(content):
    image = vision.types.Image(content=content)
    response3 = client.label_detection(image=image)
    labels = response3.label_annotations

    df = pd.DataFrame(columns=['description', 'score'])
    print("Labels:")
    for label in labels:
        df = df.append(
            dict(
                description=label.description,
                score=round(label.score,2)*100,
                
            ), ignore_index=True)
        
    print(df)
    print('')


def object(content):
    image = vision.types.Image(content=content)
    response = client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations

    pillow_image = Image.open(image_path)
    df = pd.DataFrame(columns=['name', 'score'])
    for obj in localized_object_annotations:
        df = df.append(
            dict(
                name=obj.name,
                score=round(obj.score,2)*100
            ),
            ignore_index=True)
        
       

    print(df)
    print('')

def safesearch(content):
    image = vision.types.Image(content=content)
    response = client.safe_search_detection(image=image)
    safe_search = response.safe_search_annotation
  
            
    df = pd.DataFrame(columns=['Adult', 'Spoof', 'Medical','Violence','Racy'])
    
    df = df.append(
            dict(
                Adult=safe_search.adult*20,
                Spoof=safe_search.spoof*20,
                Medical=safe_search.medical*20,
                Violence=safe_search.violence*20,
                Racy=safe_search.racy*20
                
            ), ignore_index=True)
        
    print(df)
    print('')

def properties(content):
    image = vision.types.Image(content=content)
    response2 = client.image_properties(image=image).image_properties_annotation
    dominant_colors = response2.dominant_colors


    df = pd.DataFrame(columns=['Red', 'Green','Blue','Score'])
 
    for color in dominant_colors.colors:
        
        df = df.append(
            dict(
                Red=color.color.red,
                Green=color.color.green,
                Blue=color.color.blue,
                Score=color.score*100

            ), ignore_index=True)

    
    print(df)
    print('')
        


file_name = 'person_crop_2.png'
image_path = f'.\Images\{file_name}'

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()
    faces(content)
    labels(content)
    object(content)
    safesearch(content)
    properties(content)
    




