
import yaml
import openai
from box import ConfigBox

def read_config(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    data = ConfigBox(data)
    return data  


def generate_embeddings(text: str, model_name: str):
    
    try:
        response = openai.embeddings.create(
            model=model_name,
            input=text
        )
        
        return response.data[0].embedding
        
    except:
        return None