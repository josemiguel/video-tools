import numpy
import cv2
import boto3
import numpy
import os
import io
import requests
import json
from datetime import datetime
from openai import OpenAI 
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor
from openai.types.chat.completion_create_params import ResponseFormat

def handler(event, context):
    body = json.loads(event['body'])
    video_url = body['video_url']
    local_file_path = '/tmp/video.mp4'

    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        images_frames = split_video_to_images(local_file_path)
        frames = [frame for image_bytes, frame in images_frames]
        images = [image_bytes for image_bytes, frame in images_frames]
        relevant_frames = find_relevant_frames(frames)
        print('relevant frames:', len(relevant_frames))
        content = [None] * len(relevant_frames)
        with ThreadPoolExecutor() as executor:
            for i, idx in enumerate(relevant_frames):
                content_future = executor.submit(process_with_ai, images[idx], i)
                content[i] = content_future.result()

        products = []
        seen = {}
        for part in content:
            for p in part['products']:
                name = p['product_name']
                if name is None:
                    continue

                if not name in seen:
                    seen[name] = True
                    products.append(p)

        return {
            'statusCode': 200,
            'body': json.dumps(products)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing the video: {str(e)}')
        }

def process_with_ai(image_bytes, idx):
    image_url = upload_s3(image_bytes, prefix='video/image_%d' % idx, ext='png', ExtraArgs={
                        "ContentType": 'image/png', 
                        'ACL': 'public-read'
                    })
    return get_info_image_transfer(image_url)

def upload_s3(content, prefix="", ext="", ExtraArgs=None):
    s3 = boto3.client(
        service_name ="s3",
        aws_access_key_id = '',
        aws_secret_access_key = ''
    )
    today = datetime.now().strftime('%Y-%m-%dT%H%m%s')
    upload_filename = '%s_%s.%s' % (prefix, today, ext)
    s3.upload_fileobj(io.BytesIO(content), 'f2b-images', upload_filename, ExtraArgs=ExtraArgs)
    image_url = f"https://<base_url>.s3.amazonaws.com/{upload_filename}"

    return image_url 

def calculate_difference(image1, image2):
    if image1 is None or image2 is None:
        return 100

    def to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    gray1, gray2 = to_gray(image1), to_gray(image2)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=75)
    
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 100

    # Build Annoy index
    dim = des1.shape[1]
    annoy_index = AnnoyIndex(dim, 'euclidean')
    for i, descriptor in enumerate(des1):
        annoy_index.add_item(i, descriptor)

    annoy_index.build(4) 

    def is_good_match(descriptor):
        neighbors = annoy_index.get_nns_by_vector(descriptor, 2, include_distances=True)
        return len(neighbors[1]) > 1 and neighbors[1][0] < 0.7 * neighbors[1][1]

    # Find good matches using concurrent futures
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(is_good_match, des2))

    num_good_matches = sum(results)

    if len(des2) == 0:
        feature_difference = 100
    else:
        feature_difference = min((1 - num_good_matches / len(des2)) * 100, 100)

    return feature_difference

def split_video_to_images(video_path):
    cap = cv2.VideoCapture(video_path)
    images = []
    while True:
        success, current_frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) 
        success, encoded_image = cv2.imencode('.png', frame)
        image_bytes = encoded_image.tobytes()        
        images.append((image_bytes, frame))
    cap.release()
    return images    


def select_well_distributed_points(n, percentage=8, variation=0.3):
    total_points = numpy.arange(n)
    num_points = int(n * (percentage / 100))    
    step = n / num_points
    indices = []

    for i in range(num_points):
        rand_variation = numpy.random.uniform(-variation, variation) * step
        index = int(i * step + rand_variation)
        indices.append(index)
    
    indices = numpy.clip(indices, 0, n-1)
    indices = numpy.unique(indices)  # Ensure no duplicates
    while len(indices) < num_points:
        new_indices = numpy.random.randint(0, n, num_points - len(indices))
        indices = numpy.unique(numpy.concatenate((indices, new_indices)))
    
    return total_points[indices]

def find_relevant_frames(frames):
    num_frames = len(frames)
    sample = select_well_distributed_points(num_frames)
    hacky_distribution = [calculate_difference(frames[sample[i]], frames[sample[i+1]]) 
                                     for i in range(len(sample)-1)]
    p90 = numpy.percentile(hacky_distribution, 90) if hacky_distribution else 10.0
    i, j = 0, 0
    most_relevant = [0]
    print(p90)
    while i < num_frames and j < num_frames:
        begin, end = i, num_frames - 1
        mid, diff = i + 1, None
        while begin <= end:
            mid = (begin + end) // 2
            diff = calculate_difference(frames[i], frames[mid])
            if diff > p90:
                end = mid - 1
            else:
                begin = mid + 1
        print(mid, diff)
        i = mid + 1
        most_relevant.append(mid)
    return most_relevant

def get_info_image_transfer(image_url):
    os.environ["OPENAI_API_KEY"] = "<openai-api-key>"
    client = OpenAI()

    response_format = ResponseFormat(type="json_object")

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Parse the price tags in the image returning a list of jsons per image"},
            {
              "type": "image_url",
              "image_url": {
                "url": image_url,
              },
            },
          ],
        },
        {"role": "system", 
        "content": """extract in the following json format example , 
        
         return a valid json
         
         always lowercase , images must be a price tag
         the only valid units are KG, UN, SC, CX (means: kilo or quilo, unidade, saco, caixa) if the text is long feel free to abbreviate to one of them
         the only valid weight units are KG, G and null 
         examples:
         
         images with single price tag, 3 examples:
         
             {'products' : [{'product_name': 'cebola organica', 'price': 2.1, 'unit': 'KG', 'weight': 1, 'weight_unit: 'KG'}] } 
             
             {'products' : [{'product_name': 'cebola organica', 'price': 2.1, 'unit': 'CX', 'weight': 1, 'weight_unit: 'KG'}] }              
             
             {'products' : [{'product_name': 'cebola organica', 'price': 2.1, 'unit': 'SC', 'weight': 1, 'weight_unit: 'KG'}] }
            
             {'products' :  [{'product_name': 'banana og', 'price': 12, 'unit': 'UN', 'weight': null, 'weight_unit: null}] }
         
         images with multiple price tag, 1 example:
             {'products' :  [{'product_name': 'banana og', 'price': 12, 'unit': 'UN', 'weight': '150', 'weight_unit: 'G'}, 
             {'product_name': 'laranja', 'price': 9, 'unit': 'KG', 'weight': null, 'weight_unit: null}] }
             
             
             {'products' : [{'product_name': 'tomate', 'price': 2.1, 'unit': 'KG', 'weight': 50, 'weight_unit: 'G'}, 
             {'product_name': 'laranja', 'price': 9, 'unit': 'KG', 'weight': null, 'weight_unit: null}] }
             
         if there are no valid price tag image must return {}, can be multiple prices in a single image
         """}
      ],
        response_format=response_format
    )

    content = json.loads(response.choices[0].message.content)
    print(content)
    return content
