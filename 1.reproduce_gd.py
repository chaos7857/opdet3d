import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from groundingdino.util.inference import load_image
import bisect

from utils.common import logging
##########################
# 模型
##########################

config_file = "cfg/groundingDino_swinT_ogc.yaml"
checkpoint_path = "groundingdino_swint_ogc.pth"

device="cuda" if torch.cuda.is_available() else "cpu"

cfg = SLConfig.fromfile(config_file)
cfg.device = device

model = build_model(cfg)

##########################
# 权重
##########################
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)



##########################
# 
##########################
model.to(device)

image_source, image = load_image("demo/demo.jpg")
caption = "Book.Bottle.Mouse"
box_threshold = 0.01
text_threshold = 0.25
remove_combined: bool = False


def get_phrases_from_posmap(
    posmap: torch.BoolTensor, tokenized: Dict, tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255
):
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")


with torch.no_grad():
    outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

boxes, logits.max(dim=1)[0], phrases
    



##########################
# 处理输出
##########################
logging.info("done.")

print("done.")