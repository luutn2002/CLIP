import copy
import torch
import torch.nn as nn
from torch import linalg as LA
from torchvision.transforms import CenterCrop
from typing import List, Type, Dict
import configparser

from .clip import clip

def build_mapleiqa(classnames:List[str], 
                   model_name:str, 
                   config_dir:str):
    """
    Build MaPLe-IQA models.

    :param classnames: List of classes equivalent to number of CLIP models.
    :param model_name: Name of the model to build, packages include MaPLeIQA,
                    but if you want to modify source code, please add new models
                    to model dictionary.
    :param config_dir: Directory of the .ini config file.

    :return: PyTorch MaPLe-IQA model.
    """
    #Extend other models and add to dictionary
    print("Building custom CLIP")

    config = configparser.ConfigParser()
    config.read(config_dir)

    #When add a new model to source, remember to include it in dict.
    model_dict = {'MaPLeIQA': MaPLeIQA}
    
    architecture = model_dict[model_name]
    model = architecture(classnames, load_clip_to_device(config).float(), config)
    if(config['MODEL_CONFIG'].getboolean('FREEZE_IMAGE_ENCODER')):
        print("Turning off gradients in image encoder")

    if(config['MODEL_CONFIG'].getboolean('FREEZE_TEXT_ENCODER')):
        print("Turning off gradients in text encoder")

    if(not config['MODEL_CONFIG'].getboolean('FREEZE_IMAGE_ENCODER') and not config['MODEL_CONFIG'].getboolean('FREEZE_TEXT_ENCODER')):
        print("Turning on gradients in image and text encoder")
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
    #print(f"Parameters to be updated: {enabled}")

    return model.float().to(config['MODEL_CONFIG']['DEVICE'])

def parse_tuple(input:str):
    return tuple(int(k.strip()) for k in input[1:-1].split(','))

def _get_clones(module:Type[nn.Module], 
                N:int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_device(config:Dict):
    url = clip._MODELS[config['MODEL_CONFIG']['BACKBONE']]
    model_path = clip._download(url)
    device = config['MODEL_CONFIG']['DEVICE']

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
                      "maple_length": int(config['MODEL_CONFIG']['MAPLE_N_CTX']),
                      "pos_embed": config['MODEL_CONFIG']['MAPLE_POS_EMBED']}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class MaPLeIQATextEncoder(nn.Module):
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
    def __init__(self, classnames, clip_model, config):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = config['MODEL_CONFIG'].getint('MAPLE_N_CTX')
        ctx_init = config['MODEL_CONFIG']['MAPLE_CTX_INIT']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = parse_tuple(config['MODEL_CONFIG']['MAPLE_INPUT_SIZE'])[0]
        # Default is 1, which is compound shallow prompting
        assert config['MODEL_CONFIG'].getint('MAPLE_PROMPT_DEPTH') >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = config['MODEL_CONFIG'].getint('MAPLE_PROMPT_DEPTH')  # max=12, but will create 11 such shared prompts
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
        #name_lens = [len(_tokenizer.encode(name)) for name in classnames]
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
        #self.name_lens = name_lens

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
    def __init__(self, classnames, clip_model, config):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(classnames, clip_model, config)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = MaPLeIQATextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = torch.div(image_features, LA.norm(image_features, dim=-1, keepdim=True))
        text_features = torch.div(text_features , LA.norm(text_features, dim=-1, keepdim=True))
        logits = logit_scale * torch.matmul(image_features ,text_features.t())

        return logits

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
    def __init__(self, classnames, clip_model, config):
        super().__init__()

        self.num_clip = len(classnames)

        for i in range(self.num_clip):
            disc = CustomCLIP(classnames[i], clip_model, config)

            for name, param in disc.named_parameters():
                if "prompt_learner" not in name:
                    if "image_encoder" in name and not config['MODEL_CONFIG'].getboolean('FREEZE_IMAGE_ENCODER'): continue
                    elif "text_encoder" in name and not config['MODEL_CONFIG'].getboolean('FREEZE_TEXT_ENCODER'): continue
                    else: param.requires_grad = False

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
            return pred_score
        else:
            return logits_list

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
        image = self.remove_excess(image)
        image = self.reshape(image)
        return image


class MaPLeIQA(nn.Module):
    """
    MaPLe-IQA model. 
    """
    def __init__(self, classnames, clip_model, config):
        super().__init__()
        input_size= parse_tuple(config['MODEL_CONFIG']['MAPLE_INPUT_SIZE'])
        self.predictor = MaPLeIQAPredictor(classnames, clip_model, config)
        self.reshaper = ImageBatchReshape(input_size)

    def forward(self, image):
        result = []
        image = self.reshaper(image)
        if len(image.shape) ==  5:
            for batch in image:
                score = torch.mean(self.predictor(batch)).reshape(1)
                result.append(score)
            result = torch.cat(result, dim=0)
            return result
        
        else:
            result = torch.mean(self.predictor(batch)).reshape(1)
            return result