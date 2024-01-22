from openai import AzureOpenAI
import os
import requests
from PIL import Image
from dotenv import load_dotenv
import json
from typing import Literal
import gradio as gr

from gga_utils.common import *

load_dotenv()

i18n = I18nAuto()

client = AzureOpenAI(
  azure_endpoint = os.getenv('AZURE_OAI_ENDPOINT'), 
  api_key = os.getenv('AZURE_OAI_KEY'),  
  api_version = os.getenv('API_VERSION')
)

def add_suffix(file_name, suffix):
    '''
    Add a suffix to a file name.
    
    Args:
        file_name (str): The name of the file.
        suffix (str): The suffix to be added.
    
    Returns:
        str: The new file name with the suffix added.
    '''
    # 将文件名拆分为基础名称和扩展名
    base_name, ext = os.path.splitext(file_name)

    # 新的文件名为基础名称加上后缀名和扩展名
    new_name = f"{base_name}{suffix}{ext}"
    return new_name

def get_next_available_name(file_name):
    '''
    Get the next available name to avoid overwriting an existing file
    '''
    i = 1
    while os.path.exists(file_name):
        file_name = add_suffix(file_name, f"({i})")
        i += 1
    return file_name

def generate_dall3_image(prompt:str,
                         size:Literal['1024x1024','1792x1024', '1024x1792'],
                         quality:Literal['standard', 'hd'],
                         style:Literal['natural', 'vivid'],
                         user:str = 'created by GPT-Gradio-Agent from dall-e-3',
                         progress = gr.Progress()
                         ):
    '''
    Generate an image using DALL-E 3
    
    Args:
        prompt (str): The prompt to use for the image generation
        size (Literal['1024x1024','1792x1024', '1024x1792']): The size of the image to generate
        quality (Literal['standard', 'hd']): The quality of the image to generate
        style (Literal['natural', 'vivid']): The style of the image to generate
        user (str, optional): The user to attribute the image to. Defaults to 'created by GPT-Gradio-Agent from dall-e-3'.
    
    Returns:
        image: The generated image
        str: The prompt ACTUALLY used for the image generation (augmented by GPT)
    '''
    file_name = "generated_image.png"

    progress(0, desc=i18n("Starting..."))
    # Generate the image using the DALL-E 3 model
    try:
        result = client.images.generate(
            model="dall-e-3", # the name of your DALL-E 3 deployment
            prompt=prompt,
            n=1,
            size=size,
            quality=quality,
            style=style,
            user=user
        )
    except openai.BadRequestError as err:
        raise gr.Error(i18n(str(err)))

    progress(20, desc=i18n("Requesting Dall-e-3 response..."))

    # Convert the response to a JSON object
    json_response = json.loads(result.model_dump_json())

    # Set the directory for the stored image
    image_dir = os.path.join(os.curdir, r'.\output\pic')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path (note the filetype should be png)
    # 检查名称是否重复，如果重复，则加入数字(i)后缀，并创建新名称
    new_file_name = get_next_available_name(file_name)
    image_path = os.path.join(image_dir, new_file_name)

    progress(50, desc=i18n("Downloading image..."))

    # Retrieve the generated image
    image_url = json_response["data"][0]["url"]  # extract image URL from response
    generated_image = requests.get(image_url).content  # download the image
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    # Display the image in the default image viewer
    image = Image.open(image_path)

    progress(100, desc=i18n("Downloading image..."))

    # Return the image and the prompt used for the image generation
    return image, json_response["data"][0]['revised_prompt']

def open_dir_func():
    os.startfile(filepath=r".\output\pic")