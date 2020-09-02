# Google-Vision-API
- A demo to use Google’s Vision API cloud service with vision AI in Python
# Vision API Features (Methods)
- faces() detects the facial attributes of an image 
- safesearch() searches for any explicit contents based on these five categories – adult, spoof, medical, violence, and racy and return the likelihoods. 
- labels() analyze an image, detect and extract information of different objects and entities in an image Using the Label Detection feature we can identity general objects, locations, activities, animal species, products, and more
- properties() detects general attributes of an image, such as the dominant colors composed in the image
- objects() identify different objects such as chair, tables, bicycle, door, lamp, etc, in an image
### Prerequisites
1. Create a Google Service Account in Google Cloud Platform.
2. Enable Google Vision API service.
3. Download Token JSON file to your PC.
4. Create a Python Virtual Environment.
```
python -m venv GoogleVisionApiDemo
```
5. Install Google Cloud Python libraries and Vision API library.
6. Note pip version should be greater that >=19
```
pip install google-cloud-vision
```

### Running the project
1. Run vision_api_demo.py to get the facial attributes for an image
```
python vision_api_demo.py
```


