import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
##########################################################
import json
def dump_json(data,filename:str)->bool:
    try:
        # dir_name = os.path.dirname(filename)
        # os.makedirs(dir_name,exist_ok=True)
        with open(filename,'w', encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(e)
        return False
def load_small_json(json_name:str):
    with open(json_name,'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
############################################################
def err_control(func):
    def wrapper(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            logging.error(e)
            return None
    return wrapper
############################################################
def count_model_parameters(model) -> tuple[int, int]:
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
    
    return total_params, trainable_params