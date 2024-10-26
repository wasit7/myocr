from pathlib import Path
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pprint import pprint
import pandas as pd
import re
from tqdm.auto import tqdm

class Project:
        
    def __init__ (self, project_name, my_class, clear_landing=False, clear_staging=True, clear_cropped=True):
        print('\nInitializing project:')
        # init project
        self.lb_config={
            'project_name':project_name,
            'my_class':my_class,
            'LABEL_STUDIO_URL':'http://localhost:8080',
            'IMAGE_SERVER_URL':'http://localhost:8000/',
            'API_KEY':'OMG!@#$%'
        }
        self.my_class = my_class
        
        
        # Path to the JSON file
        file_path = 'config.json'
        self.project_dir = Path(project_name)
        # Check if the file exists
        if self.project_dir.exists(): 
            # print(1)
            if (self.project_dir/file_path).exists():
                # Try reading the dictionary from the file
                # print(2)
                d=self.data_from_json(self.project_dir/file_path)
                print('Project config:')
                pprint(d)
                self.project_name =d['project_name']
                self.landing_dir =d['landing_dir']
                self.staging_dir =d['staging_dir']
                self.cropped_dir =d['cropped_dir']
                # try:
                #     with open(self.project_dir/file_path, 'r') as f:
                #         project_config = json.load(f)
                #         self.project_name = str(project_config['project_name'])
                #         self.landing_dir = Path(project_config['landing_dir'])
                #         self.staging_dir = Path(project_config['staging_dir'])
                #         self.cropped_dir = Path(project_config['cropped_dir'])
                #     print("Project successfully loaded from file:", loaded_dict)
                # except json.JSONDecodeError:
                #     print("Error: The project config file exists but is not a valid JSON file.")
        else:
            print(f"Create a new project. The file '{file_path}' does not exist.")
            self.project_name = project_name
            self.project_dir.mkdir(parents=True, exist_ok=True)
            
            self.landing_dir = (self.project_dir/f'_landing').resolve()
            self.staging_dir=(self.project_dir/f'_staging').resolve()
            self.cropped_dir=((self.project_dir/f'../../image-server/dataset/images')/self.project_name).resolve()
            
            if clear_landing:
                self.clear_directory(self.landing_dir)
            self.landing_dir.mkdir(parents=True, exist_ok=True)
            print('landing_dir: ', self.landing_dir)
            
            
            if clear_staging:
                self.clear_directory(self.staging_dir)
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            print('staging_dir: ', self.staging_dir)
    
            
            if clear_staging:
                self.clear_directory(self.cropped_dir)
            self.cropped_dir.mkdir(parents=True, exist_ok=True)
            print('cropped_dir: ', self.cropped_dir)
    
            project_config={
                'project_name':self.project_name,
                'landing_dir':self.landing_dir,
                'staging_dir':self.staging_dir,
                'cropped_dir':self.cropped_dir,
            }
            self.data_to_json(project_config,(self.project_dir/file_path))
            print('Now please load input image to landing')

    def connect_label_studio(self):
        self.lb=LabelStudio(**self.lb_config)

    def set_reference_image(self, ref_path='ref.png'):
        # reference_raw = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        reference_raw = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        self.reference_height, self.reference_width, _ = tuple(d*2 for d in reference_raw.shape)
        self.reference_image = cv2.resize(reference_raw, (self.reference_width, self.reference_height))
        return self.reference_image

    def register_image(self,debug=False):
        print('\nRegister_image:')
        width, height = (self.reference_width, self.reference_height)  
        image_files = self.find_all_images(self.landing_dir)
        reference_image =self.reference_image 
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        # Find keypoints and descriptors
        keypoints_reference, descriptors_reference = sift.detectAndCompute(reference_image, None)
        if len(self.find_all_images(self.landing_dir))==0:
            raise Exception('Error: please upload input images to _landing')
            
        for image in self.find_all_images(self.landing_dir):
            image_path=self.landing_dir/image
            print(image_path)
            
            # input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            keypoints_input, descriptors_input = sift.detectAndCompute(input_image, None)
            
            # Use the brute-force matcher to find matches between the keypoints
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors_input, descriptors_reference)
            
            # Sort the matches based on distance (best matches first)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if debug:
                # Draw the matches for visualization (optional)
                matched_image = cv2.drawMatches(input_image, keypoints_input, reference_image, keypoints_reference, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.figure(figsize=(8,8))
                plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
                plt.title('Matched Keypoints')
                plt.show()
        
            # Extract location of good matches
            points_input = np.float32([keypoints_input[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points_reference = np.float32([keypoints_reference[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Estimate the homography matrix
            H, mask = cv2.findHomography(points_input, points_reference, cv2.RANSAC, 5.0)
            
            # Warp the input image to align with the reference image
            # height, width = reference_image.shape
            aligned_image = cv2.warpPerspective(input_image, H, (width, height))
            
            if debug:
                # Show the aligned image
                plt.figure(figsize=(8,8))
                plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
                plt.title('Aligned Answer Sheet')
                plt.show()
            
            # Save the aligned image (optional)
            output_path=self.staging_dir/image.replace('_','')
            print(output_path)
            cv2.imwrite(output_path, aligned_image)
            
    def extract_bb(self, all_bbox, debug=False):
        print('\nExtract_image:')
        image_files = self.find_all_images(self.staging_dir)
         
        for i,image in enumerate(image_files):
            sheet_all_bbox=[]
            image_path = Path(self.staging_dir)/image
            sheet_name = image.split('.')[0] 
            print("image_path: ",image_path)
            print("sheet_name: ",sheet_name)
            image = Image.open(image_path)
            if debug:
                self.draw_all_bbox(image, all_bbox)
            for i in all_bbox:
                j=i.copy()
                j['s']=sheet_name
                self.save_cropped_bbox(image,**j)
                sheet_all_bbox.append(j)
            print('bbox: ', str(sheet_all_bbox)[:100])
            
    def find_all_images(self, directory):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        # directory = Path(directory_path)
        image_files = [
            str(file.relative_to(directory)) 
            for file in directory.rglob('*') 
            if file.suffix.lower() in image_extensions
            and '.ipynb_checkpoints' not in file.parts
        ]            
        return image_files

    def clear_directory(self, directory):
        ### DENGER ZONE!!!
        directory = directory.resolve()
        # Define a set of directories that should never be deleted
        protected_dirs = {Path('/'), Path('/etc'), Path('/usr'), Path.home(), directory.parent}
        # Safety check: Make sure we're not deleting a protected directory
        if directory in protected_dirs:
            raise ValueError(f"Cannot clear the protected directory: {directory}")
        
        # Check if the directory exists and is indeed a directory
        if directory.exists() and directory.is_dir():
            for item in directory.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)  # Remove subdirectory and its contents
                else:
                    item.unlink()  # Remove file
    
    # Function to draw bounding boxes
    def draw_bbox(self, ax, q, c, x, y, box_width, box_height):
        label = f'{q},{c}'
        rect = patches.Rectangle((x, y), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + box_width / 2, y + box_height / 2, label, fontsize=8, color='blue', ha='center', va='center')
    
    def draw_all_bbox(self, image, all_bbox):
        fig, ax = plt.subplots(figsize=(10, 15))
        for i in all_bbox:
            self.draw_bbox(ax, **i)
        # Display the image
        ax.imshow(image)
        # Show the result with bounding boxes
        plt.axis('off')  # Hide the axes for better visualization
        plt.show()

    # Function to save cropped images for each bounding box (b(q, c))
    def save_cropped_bbox(self, image, s, q, c, x, y, box_width, box_height):
        # Crop the bounding box from the image
        cropped_image = image.crop((x, y, x + box_width, y + box_height))
        # Save the cropped image with a name format like 'Q1_a.jpg'
        cropped_image.save(self.cropped_dir/f"B_{s}_{q}_{c}.jpg")

    def data_to_json(self, data, file_path):
        """
        Serializes data and its type to a JSON file.
        
        Parameters:
        - data: dict, the data to serialize.
        - file_path: str or Path, the file path to save the serialized data.
        """
        serialized_data = {}
        
        for key, value in data.items():
            serialized_data[key] = {
                'value': value,
                'type': type(value).__name__
            }
        
        def custom_encoder(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(file_path, 'w') as f:
            json.dump(serialized_data, f, default=custom_encoder, indent=4)
        
        print(f"Data serialized and saved to {file_path}")
    
    
    # Custom function to deserialize data and cast back to original types
    def data_from_json(self,file_path):
        """
        Deserializes data from a JSON file and casts it back to the original types.
        
        Parameters:
        - file_path: str or Path, the file path to load the serialized data.
        
        Returns:
        - dict: the deserialized data with the original types.
        """
        with open(file_path, 'r') as f:
            serialized_data = json.load(f)
        
        deserialized_data = {}
        
        for key, value_dict in serialized_data.items():
            value = value_dict['value']
            value_type = value_dict['type']
            
            if value_type == 'int':
                deserialized_data[key] = int(value)
            elif value_type == 'float':
                deserialized_data[key] = float(value)
            elif value_type == 'str':
                deserialized_data[key] = str(value)
            elif value_type == 'bool':
                deserialized_data[key] = bool(value)
            elif value_type == 'list':
                deserialized_data[key] = list(value)
            elif value_type == 'dict':
                deserialized_data[key] = dict(value)
            elif value_type == 'PosixPath':
                deserialized_data[key] = Path(value)
            else:
                deserialized_data[key] = value
        
        print(f"Data deserialized and cast to original types from {file_path}")
        return deserialized_data

    def get_img(self, directory, image_file):
        file_path=directory/image_file
        image = Image.open(file_path)
        return image

    def get_input_df(self):
        p=self
        try:
            df=pd.DataFrame({'image_file':p.find_all_images(p.cropped_dir)})
            x=df['image_file'].apply(lambda s:re.split('[_.]', s))
            x_df = pd.DataFrame(x.tolist(), columns=['bbox','sheet', 'question', 'choice','extension'])
            df = pd.concat([df, x_df], axis=1)
            df['project_name']=p.project_name
    
            df2=df[['project_name','image_file','sheet','question','choice']].copy()
            df2.loc[:,'PIL_image']=df2['image_file'].apply(lambda x:p.get_img(p.cropped_dir,x))
            df2['image_url']=self.lb_config['IMAGE_SERVER_URL']+self.project_name+'/'+df2['image_file']
            return df2
        except:
            raise Exception("Eror: unable create an input dataframe. please check your input images and cropping process.")

    def create_task(self,forward):
        p=self
        df = p.get_input_df()
        df[['predict', 'confidence']] = df.apply(lambda row: forward(row['PIL_image']), axis=1)
        df = p.get_input_df()
        df[['predict', 'confidence']] = df.apply(lambda row: forward(row['PIL_image']), axis=1)
        df_new_task = df
        df_old_task=p.lb.fetch_labels_from_label_studio()
        new_rows, deleted_rows, modified_rows=p.lb.perform_cdc(df_old_task, df_new_task, ['image_url',])
        new_images=new_rows['image_url'].to_list()
        if len(new_images):
            dfx=df_new_task[df_new_task['image_url'].isin(new_images)]
            df3=dfx.copy()
            
            tqdm.pandas(desc=f"{'create tasks':>20}")
            df3['task_id']=df3['image_url'].progress_apply(p.lb.create_task)
            
            tqdm.pandas(desc=f"{'create annotations':>20}")
            df3['annotation_id'] = df3.progress_apply(
                lambda row: p.lb.add_prediction_to_task(row['task_id'], self.my_class[row['predict']]), 
                axis=1
            )
        else:
            print('No new image found')


import os
import requests
import xml.etree.ElementTree as ET
import textwrap

class LabelStudio:
    def __init__(self, project_name, my_class, LABEL_STUDIO_URL, IMAGE_SERVER_URL, API_KEY):
        self.project_name = project_name
        self.my_class = my_class
        self.LABEL_STUDIO_URL = LABEL_STUDIO_URL
        self.IMAGE_SERVER_URL=IMAGE_SERVER_URL
        self.API_KEY = API_KEY

        self.project_id=self.create_or_get_project(project_name)
        labeling_config = self.get_config(my_class)
        self.set_labeling_config(self.project_id, labeling_config)
        
    # Step 1: Get a list of existing projects
    def get_project_by_name(self, project_name):
        url = f"{self.LABEL_STUDIO_URL}/api/projects"
        headers = {"Authorization": f"Token {self.API_KEY}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    
        projects = response.json()
        for project in projects['results']:
            if project["title"] == project_name:
                return project["id"]
        return None
    
    # Step 2: Create a new project if it doesn't exist
    def create_project(self, project_name):
        url = f"{self.LABEL_STUDIO_URL}/api/projects"
        headers = {"Authorization": f"Token {self.API_KEY}"}
        payload = {"title": project_name}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        return response.json()["id"]
    
    # Step 3: Create or get project function
    def create_or_get_project(self, project_name):
        project_id = self.get_project_by_name(project_name)
        if project_id is not None:
            self.project_id =project_id
            print(f"Project '{project_name}' already exists with ID: {project_id}")
            return project_id
        else:
            project_id = self.create_project(self.project_name)
            self.project_id =project_id
            print(f"Project '{project_name}' created successfully with ID: {project_id}")
            return project_id

    def get_config(self, class_to_label):
        labeling_config = textwrap.dedent('''
            <View>
                <Image name="image" value="$image"/>
                <Choices name="choice" toName="image" showInLine="true">
                    <Choice value="empty"/>
                </Choices>
            </View>
        ''')
                                          
        root = ET.fromstring(labeling_config)
        choices_element = root.find(".//Choices")
        
        for choice in choices_element.findall('Choice'):
            choices_element.remove(choice)
            
        for key, value in class_to_label.items():
            new_choice = ET.SubElement(choices_element, 'Choice')
            new_choice.set('value', value)
    
        modified_xml = "    "+ET.tostring(root, encoding='unicode',)
        
        return modified_xml
        
    
    # Step 3: Set up the labeling configuration
    def set_labeling_config(self, project_id, labeling_config):
        url = f"{self.LABEL_STUDIO_URL}/api/projects/{project_id}"
        headers = {"Authorization": f"Token {self.API_KEY}"}
        payload = {"label_config": labeling_config}
        response = requests.patch(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"{response.json()['label_config']}")
        return 

    def create_task(self, image_url):
        url = f"{self.LABEL_STUDIO_URL}/api/tasks"
        headers = {"Authorization": f"Token {self.API_KEY}"}
        payload = {
            "project": self.project_id,
            "data": {"image": image_url}
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['id']
        
    def add_prediction_to_task(self, task_id, clas_name):
        url = f"{self.LABEL_STUDIO_URL}/api/tasks/{task_id}/annotations"
        headers = {"Authorization": f"Token {self.API_KEY}"}
        payload = {
            "result": [
                {
                    "value": {"choices": [clas_name]},
                    "from_name": "choice",
                    "to_name": "image",
                    "type": "choices"
                }
            ],
            "completed_by": 1  # Admin user
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()['id']


    def fetch_labels_from_label_studio(self):
        url = f"{self.LABEL_STUDIO_URL}/api/projects/{self.project_id}/export"
        headers = {"Authorization": f"Token {self.API_KEY}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        annotations = response.json()
        
        image_paths = []
        label_ids = []
        label_names = []
        task_ids = []
        inverted_my_class =  {value: key for key, value in self.my_class.items()}
        for annotation in annotations:
            task_id = annotation["id"]
            image_url = annotation["data"]["image"]
            label_name = annotation["annotations"][0]["result"][0]["value"]["choices"][0]
    
            task_ids.append(task_id)
            
            # Translate label back to class ID
            class_id = inverted_my_class[label_name]
            
            # image_path = image_url  # Extract image path
            image_paths.append(image_url)
            label_ids.append(class_id)
            label_names.append(label_name)
            # labels.append(label)

        df=pd.DataFrame({
            'task_id':task_ids,
            'class_id':label_ids,
            'class_name': label_names,
            'image_url':image_paths
        })
        df['image_file']=df['image_url'].apply(lambda x: x.replace(self.IMAGE_SERVER_URL+self.project_name+'/',''))
        return df

    def perform_cdc(self, df1, df2, subset_columns=None):
        
        # Identify common columns if subset_columns is not provided
        if subset_columns is None:
            common_columns = df1.columns.intersection(df2.columns)
        else:
            common_columns = df1.columns.intersection(df2.columns).intersection(subset_columns)
    
        # Only consider common columns for comparison
        df1_common = df1[common_columns].reset_index(drop=True)
        df2_common = df2[common_columns].reset_index(drop=True)
    
        # Step 1: Find New Rows (rows in df2 but not in df1 based on the subset columns)
        new_rows = df2_common[~df2_common.isin(df1_common.to_dict(orient='list')).all(axis=1)]
        
        # Step 2: Find Deleted Rows (rows in df1 but not in df2 based on the subset columns)
        deleted_rows = df1_common[~df1_common.isin(df2_common.to_dict(orient='list')).all(axis=1)]
        
        # Step 3: Find Modified Rows (rows with the same values except for some modifications)
        # We take all rows from both DataFrames that are different (i.e., not common)
        modified_rows = pd.concat([df1_common, df2_common]).drop_duplicates(keep=False)
    
        return new_rows, deleted_rows, modified_rows