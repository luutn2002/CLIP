import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch import linalg as LA
from torchvision.transforms import CenterCrop

from models.clip import clip
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

MAPLE_PROMPT_DEPTH = 9
MAPLE_INPUT_SIZE = (224, 224)
MAPLE_N_CTX = 2
MAPLE_CTX_INIT = "This is a "
MAPLE_PRETRAIN_DIR = './model.pth.tar-5'
MAPLE_POS_EMBED = True
MAPLE_INNER_BATCH = 12

BACKBONE = 'ViT-B/32'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_device(device=DEVICE):
    url = clip._MODELS[BACKBONE]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device)
        state_dict = None
        print("Loading jitted model.")
        
    except RuntimeError:
        state_dict = torch.load(model_path, map_location=device)
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": MAPLE_N_CTX,
                      "pos_embed": MAPLE_POS_EMBED}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = MAPLE_N_CTX
        ctx_init = MAPLE_CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = MAPLE_INPUT_SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert MAPLE_PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = MAPLE_PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        #self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = torch.div(image_features, LA.norm(image_features, dim=-1, keepdim=True))
        text_features = torch.div(text_features , LA.norm(text_features, dim=-1, keepdim=True))
        logits = logit_scale * torch.matmul(image_features ,text_features.t())

        return logits

class CustomCLIP_V2(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = torch.div(image_features, LA.norm(image_features, dim=-1, keepdim=True))
        text_features = torch.div(text_features , LA.norm(text_features, dim=-1, keepdim=True))

        return image_features, text_features

class MaPLeEncoder(nn.Module):
    """
    Return inference result of CLIP's image and text encoder with prompt learner. 
    """
    def __init__(self, classnames, clip_model, freeze_txt_encoder=False):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        if freeze_txt_encoder:
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = False
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = torch.div(image_features, LA.norm(image_features, dim=-1, keepdim=True))
        text_features = torch.div(text_features , LA.norm(text_features, dim=-1, keepdim=True))
        logits_per_image = logit_scale * torch.matmul(image_features ,text_features.t())
        logits_per_text = logit_scale * torch.matmul(text_features ,image_features.t())

        return logits_per_image, logits_per_text

class NonLinearRegressor(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=128):
        super().__init__()

        self.linear_1 = nn.Linear(n_input, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_hidden)
        self.linear_3 = nn.Linear(n_hidden, n_output)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, logits):
        x = self.lrelu(self.linear_1(logits))
        x = self.lrelu(self.linear_2(x))
        return self.linear_3(x)

class MaPLeIQAPredictor(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()

        self.num_clip = len(classnames)

        for i in range(self.num_clip):
            disc = CustomCLIP(classnames[i], clip_model)

            for name, param in disc.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad = False

            setattr(self, 'clipmodel_{}'.format(i), disc)
        if self.num_clip > 1:
            self.regressor = NonLinearRegressor(n_input=self.num_clip, n_output=1)

    def forward(self, image):
        logits_list = []
        for i in range(self.num_clip):
            disc = getattr(self, 'clipmodel_{}'.format(i))
            logits = disc(image)
            logits_list.append(logits[:, 0].unsqueeze(1))
        logits_list = torch.cat(logits_list, dim=1).float()
        if self.num_clip > 1:
            pred_score = self.regressor(logits_list)
            return pred_score, logits_list
        else:
            return logits_list, logits_list

class MaPLeIQAPredictor_V2(nn.Module):
    """
    Added pos embedding removal.
    """
    def __init__(self, classnames, clip_model, maple_state_dict):
        super().__init__()

        self.num_clip = len(classnames)

        for i in range(self.num_clip):
            disc = CustomCLIP(classnames[i], clip_model)
            disc.load_state_dict(maple_state_dict, strict=False)
            for name, param in disc.named_parameters():
                if name in maple_state_dict:
                    param.requires_grad = False
            

            setattr(self, 'clipmodel_{}'.format(i), disc)
        if self.num_clip > 1:
            self.regressor = NonLinearRegressor(n_input=self.num_clip, n_output=1)

    def forward(self, image):
        logits_list = []
        for i in range(self.num_clip):
            disc = getattr(self, 'clipmodel_{}'.format(i))
            logits = disc(image)
            logits_list.append(logits[:, 0].unsqueeze(1))
        logits_list = torch.cat(logits_list, dim=1).float()
        if self.num_clip > 1:
            pred_score = self.regressor(logits_list)
            return pred_score, logits_list
        else:
            return logits_list, logits_list

class ImageBatchReshape(nn.Module):
    """
    Pytorch module used to trim some excess and reshape image to batches of images with resolution different than original

    Parameters:
    output_size(int or tuple of int): resolution of final image size (either an int represent both of height and width or a tuple of 2 int (h x w)). 
    """
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, tuple):
            assert len(output_size) == 2 and all(isinstance(ele, int) for ele in output_size), "Tuple size should be 2 and all element in tuple have to be integer"
        else: assert isinstance(output_size, int), "This function only accept integer or tuple of integer"
        self.output_size = output_size

    def remove_excess(self, image):
        """
        Using pytorch CenterCrop to remove excess around image for easier reshape

        Args:
        image (torch.Tensor): image input as torch.Tensor

        Returns:
        image (torch.Tensor): trimmed image as torch.Tensor
        batch_num (int): number of resulted images
        """
        if isinstance(self.output_size, tuple):
            width_num = int(image.shape[-1]/self.output_size[1])
            height_num = int(image.shape[-2]/self.output_size[0])
            return CenterCrop((self.output_size[0]*height_num, self.output_size[1]*width_num))(image), width_num*height_num
        else:
            width_num = int(image.shape[-1]/self.output_size)
            height_num = int(image.shape[-2]/self.output_size)
            return CenterCrop((self.output_size*height_num, self.output_size*width_num))(image), width_num*height_num

    def reshape(self, image):
        """
        Reshape image to smaller frames

        Args:
        image (torch.Tensor): image input as torch.Tensor

        Returns:
        image (torch.Tensor): reshaped image as torch.Tensor
        """
        assert len(image.shape) <= 4 and len(image.shape) >= 3, "Reshape operation only accept 3 or 4 dimension images" 
        if len(image.shape) == 3:
            if isinstance(self.output_size, tuple):
                image = torch.reshape(image, (-1, image.shape[0], self.output_size[0], self.output_size[1]))
            else: image = torch.reshape(image, (-1, image.shape[0], self.output_size, self.output_size))
        else:
            if isinstance(self.output_size, tuple):
                image = torch.reshape(image, (image.shape[0], -1, image.shape[1], self.output_size[0], self.output_size[1]))
            else: image = torch.reshape(image, (image.shape[0], -1, image.shape[1], self.output_size, self.output_size))
        
        return image

    def forward(self, image):
        image, batch_num = self.remove_excess(image)
        image = self.reshape(image)
        return image


class MaPLeIQAPredictor_V3(nn.Module):
    """
    MaPLe-IQA with image batches splitting

    Parameters:
    
    """
    def __init__(self, classnames, clip_model, input_size=MAPLE_INPUT_SIZE):
        super().__init__()
        self.predictor = MaPLeIQAPredictor(classnames, load_clip_to_device().float())
        self.reshaper = ImageBatchReshape(input_size)

    def forward(self, image):
        result = []
        image = self.reshaper(image)
        if len(image.shape) ==  5:
            for batch in image:
                score = torch.mean(self.predictor(batch)[0]).reshape(1)
                result.append(score)
            result = torch.cat(result, dim=0)
            return result, None
        
        else:
            result = torch.mean(self.predictor(batch)[0]).reshape(1)
            return result, None

class MaPLeIQAPredictor_V4(nn.Module):
    """
    MaPLe-IQA with image batches splitting to multiple features and to regressor.

    Parameters:
    
    """
    def __init__(self, classnames, clip_model, input_size=MAPLE_INPUT_SIZE, num_batch=MAPLE_INNER_BATCH):
        super().__init__()

        self.num_clip = len(classnames)
        self.num_batch = num_batch

        for i in range(self.num_clip):
            disc = CustomCLIP(classnames[i], clip_model)

            for name, param in disc.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad = False

            setattr(self, 'clipmodel_{}'.format(i), disc)
        if self.num_clip > 1:
            self.regressor = NonLinearRegressor(n_input=self.num_clip*self.num_batch, n_output=1)

        self.reshaper = ImageBatchReshape(input_size)

    def forward(self, image):
        image = self.reshaper(image)
        batch_size = image.shape[0]
        logits_list = []
        for i in range(self.num_clip):
            disc = getattr(self, 'clipmodel_{}'.format(i))
            for batch in image:
                logits = disc(batch)
                logits_list.append(logits[:, 0].unsqueeze(1))
        logits_list = torch.cat(logits_list, dim=0)
        logits_list = torch.reshape(logits_list, (batch_size, -1))
        if self.num_clip*self.num_batch > 1:
            pred_score = self.regressor(logits_list)
            return pred_score, logits_list
        else:
            return logits_list, logits_list

scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

class MaPLeIQAPredictor_V5(nn.Module):
    """
    MaPLe-IQA with architecture similar to LIQE(https://github.com/zwx8981/LIQE).

    Parameters:
    
    """
    def __init__(self, classnames, clip_model, input_size=MAPLE_INPUT_SIZE, step=32, num_patch=15):
        super().__init__()

        self.model = CustomCLIP(classnames, clip_model)
        self.model.logit_scale.requires_grad = False
        #for name, param in self.model.named_parameters():
            #if "prompt_learner" not in name:
                #param.requires_grad = False
        
        self.step = step
        self.num_patch = num_patch
        #self.regressor = NonLinearRegressor(n_input=int(num_patch*len(classnames)/batch_size), n_output=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, 224, 224)
    
        sel_step = x.size(0) // self.num_patch
        sel = torch.zeros(self.num_patch)
        for i in range(self.num_patch):
            sel[i] = sel_step * i
        sel = sel.long()
        x = x[sel, ...]
    
        #x = x.view(-1, x.size(2), x.size(3), x.size(4))

        logits_per_image = self.model(x)

        logits_per_image = logits_per_image.view(batch_size, self.num_patch, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)

        logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
        logits_quality = logits_per_image.sum(3).sum(2)

        quality_prediction = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

        return quality_prediction, None

#for some reason training fp16 on cuda create nan gradient but normal fp32
def build_mapleiqa(classnames):
    print("Building custom CLIP")
    model = MaPLeIQAPredictor(classnames, load_clip_to_device().float())
    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    return model.float().to(DEVICE)

def build_mapleiqa_v2(classnames):
    print("Building custom CLIP")
    model_state_dict = torch.load(MAPLE_PRETRAIN_DIR, map_location=DEVICE)['state_dict']
    new_state_dict = model_state_dict.copy()
    to_delete = ['prompt_learner.token_prefix', 'prompt_learner.token_suffix', 'prompt_learner.ctx', 'prompt_learner.compound_prompts_text']

    for param_tensor in model_state_dict:
        for name in to_delete:
            if name in param_tensor:
                del new_state_dict[param_tensor]
    
    model = MaPLeIQAPredictor_V2(classnames, load_clip_to_device().float(), new_state_dict)
    print("Turning off gradients for pretrained MaPLe")
  
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    return model.float().to(DEVICE)

def build_mapleiqa_v3(classnames):
    print("Building custom CLIP")
    model = MaPLeIQAPredictor_V3(classnames, load_clip_to_device().float())
    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    return model.float().to(DEVICE)

def build_mapleiqa_v4(classnames):
    print("Building custom CLIP")
    model = MaPLeIQAPredictor_V4(classnames, load_clip_to_device().float())
    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    return model.float().to(DEVICE)

def build_mapleiqa_v5(classnames):
    print("Building custom CLIP")
    model = MaPLeIQAPredictor_V5(classnames, load_clip_to_device().float())
    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name:
            #Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad_(True)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    #print(f"Parameters to be updated: {enabled}")

    return model.float().to(DEVICE)