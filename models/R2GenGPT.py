import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.metrics import f1_score
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import re


class Accuracy(object):
    # Strict accuracy compares text in a strict manner.
    def __init__(self) -> None:
        super().__init__()
        self._correct = 0
        self._total = 0

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean(self, report):
        # Clean answers before comparing them
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                            .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .' 
        return report

    def compute_score(self, ref, hypo):
        for k, v in ref.items():
            assert isinstance(v, list) and len(v) == 1 # sanity check
            assert isinstance(hypo[k], list) and len(hypo[k]) == 1 # sanity check
            true = v[0]
            pred = hypo[k][0]
            # clean up sentences
            true = self.clean(true)
            pred = self.clean(pred)
            if true == pred:
                self._correct += 1
            self._total += 1
        accuracy = self._correct / self._total
        # just in case, clear up the variables to avoid accumulation if same object is used.
        self._correct = 0
        self._total = 0
        return accuracy, 0


class F1_Score(Accuracy):
    # F1 score compares text in a strict manner.
    def __init__(self) -> None:
        super().__init__()
        self._ans2int = {}

    def compute_score(self, ref, hypo):
        # apply same format
        for k, v in ref.items():
            assert isinstance(v, list) and len(v) == 1
            assert isinstance(hypo[k], list) and len(hypo[k]) == 1
            v = self.clean(v[0])
            hypo[k] = self.clean(hypo[k][0])
        all_ans_ref = set([ans for k, v in ref.items() for ans in v])
        self._ans2int = {ans: i for i, ans in enumerate(all_ans_ref)}
        # encode
        ref_encoded = [self._ans2int[v[0]] for _, v in ref.items()]
        hypo_encoded = [self._ans2int[hypo[k][0]] for k, _ in ref.items()]
        f1 = f1_score(ref_encoded, hypo_encoded, average='micro')
        return f1, 0


class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                #device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        elif args.llm_freeze:
            for param in self.llama_model.parameters():
                # Check if parameter dtype is  Half (float16)
                if param.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading Frozen LLAMA Done')
        else:
            for param in self.llama_model.parameters():
                # Check if parameter dtype is  Half (float16)
                if param.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)
            self.embed_tokens = self.llama_model.get_input_embeddings()
            print('Loading Unfrozen LLAMA Done')

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            #(Meteor(), "METEOR"),
            (Cider(), "CIDEr"),
            (Accuracy(), "Accuracy")
        ]
        #if self.args.dataset in ['insegcat', 'ris']: # If dataset only has binary answers, use F1 score too
        #    scorers.append((F1_Score(), "F1_Score"))
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama


    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def prompt_wrap_vqa(self, img_embeds, atts_img, question):
        base_prompt ="""Answer the question below using the context below\nContext: <Img><ImageHere></Img>\nQuestion: <QuestionHere>\nAnswer: """ # prompt for each batch item. Image tokens always have same shape, but question will vary
        batch_size = img_embeds.shape[0] # call it B
        p_before_image, p_after_image = base_prompt.split('<ImageHere>')
        p_before_image_tokens = self.llama_tokenizer(
            p_before_image, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 14]
        p_after_image_before_question, p_after_question = p_after_image.split('<QuestionHere>')
        p_after_image_before_question_tokens = self.llama_tokenizer(
            p_after_image_before_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 7]
        p_after_question_tokens = self.llama_tokenizer(
            p_after_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 5]
        # now get the question tokens using padding
        question_tokens = self.llama_tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(img_embeds.device) # should be e.g. [B, max_question_len_in_batch] = [12, 13]
        # get embeddings
        p_before_image_embeds = self.embed_tokens(p_before_image_tokens.input_ids).expand(batch_size, -1, -1) # size [B, 14, 4096]
        p_after_image_before_question_embeds = self.embed_tokens(p_after_image_before_question_tokens.input_ids).expand(batch_size, -1, -1)  # size [B, 7, 4096]
        p_after_question_embeds = self.embed_tokens(p_after_question_tokens.input_ids).expand(batch_size, -1, -1) # size [B, 5, 4096]
        question_embeds = self.embed_tokens(question_tokens.input_ids) # size [B, max_question_len_in_batch, 4096]
        # put embeddings together following the prompt
        wrapped_prompt_embeds = torch.cat([p_before_image_embeds, img_embeds, p_after_image_before_question_embeds, question_embeds, p_after_question_embeds], dim=1) # should be [B, 14 + 49(from img_embeds) + 4 + max_question_len_in_batch + 4, 4096]
        # this time, we need to combine attention masks from all parts of the prompt
        wrapped_atts_prompt = torch.cat([   p_before_image_tokens.attention_mask.expand(batch_size, -1), 
                                            atts_img, 
                                            p_after_image_before_question_tokens.attention_mask.expand(batch_size, -1),
                                            question_tokens.attention_mask,
                                            p_after_question_tokens.attention_mask.expand(batch_size, -1)], dim = 1 ) # size [B, 14 + 49(from img_embeds) + 4 + max_question_len_in_batch + 4]
        return wrapped_prompt_embeds, wrapped_atts_prompt


    def ours4(self, img_embeds, atts_img, region_embeds, atts_region, question):
        base_prompt ="""Answer the question below using the context below\nContext: <Img><ImageHere></Img>\nQuestion: <QuestionHere>\nAnswer: """ # prompt for each batch item. Image tokens always have same shape, but question will vary
        batch_size = img_embeds.shape[0] # call it B
        p_before_image, p_after_image = base_prompt.split('<ImageHere>')
        p_before_image_tokens = self.llama_tokenizer(
            p_before_image, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 14]
        p_after_image_before_question, p_after_question = p_after_image.split('<QuestionHere>')
        p_after_image_before_question_tokens = self.llama_tokenizer(
            p_after_image_before_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 7]
        p_after_question_tokens = self.llama_tokenizer(
            p_after_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 5]
        # now get the question tokens using padding
        question_tokens = self.llama_tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(img_embeds.device) # should be e.g. [B, max_question_len_in_batch] = [12, 13]
        # get embeddings
        p_before_image_embeds = self.embed_tokens(p_before_image_tokens.input_ids).expand(batch_size, -1, -1) # size [B, 14, 4096]
        p_after_image_before_question_embeds = self.embed_tokens(p_after_image_before_question_tokens.input_ids).expand(batch_size, -1, -1)  # size [B, 7, 4096]
        p_after_question_embeds = self.embed_tokens(p_after_question_tokens.input_ids).expand(batch_size, -1, -1) # size [B, 5, 4096]
        question_embeds = self.embed_tokens(question_tokens.input_ids) # size [B, max_question_len_in_batch, 4096]
        # put embeddings together following the prompt
        wrapped_prompt_embeds = torch.cat([p_before_image_embeds, img_embeds, region_embeds, p_after_image_before_question_embeds, question_embeds, p_after_question_embeds], dim=1) # should be [B, 14 + 49(from img_embeds) + 4 + max_question_len_in_batch + 4, 4096]
        # this time, we need to combine attention masks from all parts of the prompt
        wrapped_atts_prompt = torch.cat([   p_before_image_tokens.attention_mask.expand(batch_size, -1), 
                                            atts_img,
                                            atts_region, 
                                            p_after_image_before_question_tokens.attention_mask.expand(batch_size, -1),
                                            question_tokens.attention_mask,
                                            p_after_question_tokens.attention_mask.expand(batch_size, -1)], dim = 1 ) # size [B, 14 + 49(from img_embeds) + 4 + max_question_len_in_batch + 4]
        return wrapped_prompt_embeds, wrapped_atts_prompt

    def ours2(self, img_embeds, atts_img, region_embeds, atts_region, question):
        # Similar to prompt_wrap_vqa_special1, but this time the region is given in the prompt in a different location (separately)
        base_prompt ="""Answer the question below using the context and region below\nContext: <Img><ImageHere></Img>\nZoom in to region: <Img><RegionHere></Img>\nQuestion: <QuestionHere>\nAnswer: """
        batch_size = img_embeds.shape[0]
        p_before_image, p_after_image = base_prompt.split('<ImageHere>')
        p_before_image_tokens = self.llama_tokenizer(
            p_before_image, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 14]
        p_after_image_before_region, p_after_region = p_after_image.split('<RegionHere>')
        p_after_image_before_region_tokens = self.llama_tokenizer(
            p_after_image_before_region, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_region_before_question, p_after_question = p_after_region.split('<QuestionHere>')
        p_after_region_before_question_tokens = self.llama_tokenizer(
            p_after_region_before_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_question_tokens = self.llama_tokenizer(
            p_after_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        # now get the question tokens using padding
        question_tokens = self.llama_tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(img_embeds.device)
        # get embeddings
        p_before_image_embeds = self.embed_tokens(p_before_image_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_image_before_region_embeds = self.embed_tokens(p_after_image_before_region_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_region_before_question_embeds = self.embed_tokens(p_after_region_before_question_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_question_embeds = self.embed_tokens(p_after_question_tokens.input_ids).expand(batch_size, -1, -1)
        question_embeds = self.embed_tokens(question_tokens.input_ids)
        # put embeddings together following the prompt
        wrapped_prompt_embeds = torch.cat([p_before_image_embeds, img_embeds, p_after_image_before_region_embeds, region_embeds, p_after_region_before_question_embeds, question_embeds, p_after_question_embeds], dim=1)
        # this time, we need to combine attention masks from all parts of the prompt
        wrapped_atts_prompt = torch.cat([   p_before_image_tokens.attention_mask.expand(batch_size, -1), 
                                            atts_img,
                                            p_after_image_before_region_tokens.attention_mask.expand(batch_size, -1),
                                            atts_region,
                                            p_after_region_before_question_tokens.attention_mask.expand(batch_size, -1),
                                            question_tokens.attention_mask,
                                            p_after_question_tokens.attention_mask.expand(batch_size, -1)], dim = 1 )
        return wrapped_prompt_embeds, wrapped_atts_prompt


    def ours3(self, img_embeds, atts_img, region_embeds, atts_region, question):
        # Similar to prompt_wrap_vqa_special1, but this time the region is given in the prompt in a different location (separately)
        base_prompt ="""Answer the question below based on this region <Img><ImageHere></Img> but also consider this context: <Img><RegionHere></Img>\nQuestion: <QuestionHere>\nAnswer: """
        batch_size = img_embeds.shape[0]
        p_before_image, p_after_image = base_prompt.split('<ImageHere>')
        p_before_image_tokens = self.llama_tokenizer(
            p_before_image, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 14]
        p_after_image_before_region, p_after_region = p_after_image.split('<RegionHere>')
        p_after_image_before_region_tokens = self.llama_tokenizer(
            p_after_image_before_region, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_region_before_question, p_after_question = p_after_region.split('<QuestionHere>')
        p_after_region_before_question_tokens = self.llama_tokenizer(
            p_after_region_before_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_question_tokens = self.llama_tokenizer(
            p_after_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        # now get the question tokens using padding
        question_tokens = self.llama_tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(img_embeds.device)
        # get embeddings
        p_before_image_embeds = self.embed_tokens(p_before_image_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_image_before_region_embeds = self.embed_tokens(p_after_image_before_region_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_region_before_question_embeds = self.embed_tokens(p_after_region_before_question_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_question_embeds = self.embed_tokens(p_after_question_tokens.input_ids).expand(batch_size, -1, -1)
        question_embeds = self.embed_tokens(question_tokens.input_ids)
        # put embeddings together following the prompt
        # ! Below, I give region_embeds first, to account for disagreement above. This was easier than modifying all variables above
        wrapped_prompt_embeds = torch.cat([p_before_image_embeds, region_embeds, p_after_image_before_region_embeds, img_embeds, p_after_region_before_question_embeds, question_embeds, p_after_question_embeds], dim=1)
        # this time, we need to combine attention masks from all parts of the prompt
        wrapped_atts_prompt = torch.cat([   p_before_image_tokens.attention_mask.expand(batch_size, -1), 
                                            atts_img,
                                            p_after_image_before_region_tokens.attention_mask.expand(batch_size, -1),
                                            atts_region,
                                            p_after_region_before_question_tokens.attention_mask.expand(batch_size, -1),
                                            question_tokens.attention_mask,
                                            p_after_question_tokens.attention_mask.expand(batch_size, -1)], dim = 1 )
        return wrapped_prompt_embeds, wrapped_atts_prompt

    def complementary_wrap(self, dr_embeds, atts_dr, co_embeds, atts_co, cr_embeds, atts_cr, question):
        base_prompt ="""Answer the question below using the image, context and region below\nImage: <Img><ImageHere></Img>\nContext: <Img><ContextHere></Img>\nRegion: <Img><RegionHere></Img>\nQuestion: <QuestionHere>\nAnswer: """
        batch_size = dr_embeds.shape[0]
        p_before_image, p_after_image = base_prompt.split('<ImageHere>')
        p_before_image_tokens = self.llama_tokenizer(
            p_before_image, return_tensors="pt", add_special_tokens=False).to(dr_embeds.device)
        p_after_image_before_context, p_after_context = p_after_image.split('<ContextHere>')
        p_after_image_before_context_tokens = self.llama_tokenizer(
            p_after_image_before_context, return_tensors="pt", add_special_tokens=False).to(dr_embeds.device)
        p_after_context_before_region, p_after_region = p_after_context.split('<RegionHere>')
        p_after_context_before_region_tokens = self.llama_tokenizer(
            p_after_context_before_region, return_tensors="pt", add_special_tokens=False).to(dr_embeds.device)
        p_after_region_before_question, p_after_question = p_after_region.split('<QuestionHere>')
        p_after_region_before_question_tokens = self.llama_tokenizer(
            p_after_region_before_question, return_tensors="pt", add_special_tokens=False).to(dr_embeds.device)
        p_after_question_tokens = self.llama_tokenizer(
            p_after_question, return_tensors="pt", add_special_tokens=False).to(dr_embeds.device)
        # now get the question tokens using padding
        question_tokens = self.llama_tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(dr_embeds.device)
        # get embeddings
        p_before_image_embeds = self.embed_tokens(p_before_image_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_image_before_context_embeds = self.embed_tokens(p_after_image_before_context_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_context_before_region_embeds = self.embed_tokens(p_after_context_before_region_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_region_before_question_embeds = self.embed_tokens(p_after_region_before_question_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_question_embeds = self.embed_tokens(p_after_question_tokens.input_ids).expand(batch_size, -1, -1)
        question_embeds = self.embed_tokens(question_tokens.input_ids)
        # put embeddings together following the prompt
        wrapped_prompt_embeds = torch.cat([p_before_image_embeds, dr_embeds, p_after_image_before_context_embeds, co_embeds, p_after_context_before_region_embeds, cr_embeds, p_after_region_before_question_embeds, question_embeds, p_after_question_embeds], dim=1)
        # this time, we need to combine attention masks from all parts of the prompt
        wrapped_atts_prompt = torch.cat([   p_before_image_tokens.attention_mask.expand(batch_size, -1), 
                                            atts_dr,
                                            p_after_image_before_context_tokens.attention_mask.expand(batch_size, -1),
                                            atts_co,
                                            p_after_context_before_region_tokens.attention_mask.expand(batch_size, -1),
                                            atts_cr,
                                            p_after_region_before_question_tokens.attention_mask.expand(batch_size, -1),
                                            question_tokens.attention_mask,
                                            p_after_question_tokens.attention_mask.expand(batch_size, -1)], dim = 1 )
        return wrapped_prompt_embeds, wrapped_atts_prompt

    def ours5(self, img_embeds, atts_img, region_embeds, atts_region, question):
        # pp. 54
        base_prompt ="""Answer the question below using the contents of the following region\nRegion: <Img><RegionHere></Img>\nBut also consider the region in context: <Img><ImageHere></Img>\nQuestion: <QuestionHere>\nAnswer: """
        batch_size = img_embeds.shape[0]
        p_before_region, p_after_region = base_prompt.split('<RegionHere>')
        p_before_region_tokens = self.llama_tokenizer(
            p_before_region, return_tensors="pt", add_special_tokens=False).to(img_embeds.device) # size [1, 14]
        p_after_region_before_image, p_after_image = p_after_region.split('<ImageHere>')
        p_after_region_before_image_tokens = self.llama_tokenizer(
            p_after_region_before_image, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_image_before_question, p_after_question = p_after_image.split('<QuestionHere>')
        p_after_image_before_question_tokens = self.llama_tokenizer(
            p_after_image_before_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_question_tokens = self.llama_tokenizer(
            p_after_question, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        # now get the question tokens using padding
        question_tokens = self.llama_tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(img_embeds.device)
        # get embeddings
        p_before_region_embeds = self.embed_tokens(p_before_region_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_region_before_image_embeds = self.embed_tokens(p_after_region_before_image_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_image_before_question_embeds = self.embed_tokens(p_after_image_before_question_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_question_embeds = self.embed_tokens(p_after_question_tokens.input_ids).expand(batch_size, -1, -1)
        question_embeds = self.embed_tokens(question_tokens.input_ids)
        # put embeddings together following the prompt
        wrapped_prompt_embeds = torch.cat([p_before_region_embeds, region_embeds, p_after_region_before_image_embeds, img_embeds, p_after_image_before_question_embeds, question_embeds, p_after_question_embeds], dim=1)
        # this time, we need to combine attention masks from all parts of the prompt
        wrapped_atts_prompt = torch.cat([   p_before_region_tokens.attention_mask.expand(batch_size, -1), 
                                            atts_img,
                                            p_after_region_before_image_tokens.attention_mask.expand(batch_size, -1),
                                            atts_region,
                                            p_after_image_before_question_tokens.attention_mask.expand(batch_size, -1),
                                            question_tokens.attention_mask,
                                            p_after_question_tokens.attention_mask.expand(batch_size, -1)], dim = 1 )
        return wrapped_prompt_embeds, wrapped_atts_prompt


    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        if self.args.vqa:
            if self.args.ours:
                # special case: if args.ours is True and baseline=='complementary', then separate images
                if self.args.baseline == 'complementary':
                    dr_embeds, atts_dr = self.encode_img([image[0]])
                    dr_embeds = self.layer_norm(dr_embeds)
                    co_embeds, atts_co = self.encode_img([image[1]])
                    co_embeds = self.layer_norm(co_embeds)
                    cr_embeds, atts_cr = self.encode_img([image[2]])
                    cr_embeds = self.layer_norm(cr_embeds)
                    img_embeds, atts_img = getattr(self, self.args.ours_version)(dr_embeds, atts_dr, co_embeds, atts_co, cr_embeds, atts_cr, samples['question'])
                else:
                    # if our method, use special wrapping where image an region are included in the prompt
                    region = samples['region']
                    region_embeds, atts_region = self.encode_img(region)
                    region_embeds = self.layer_norm(region_embeds)
                    img_embeds, atts_img = getattr(self, self.args.ours_version)(img_embeds, atts_img, region_embeds, atts_region, samples['question'])
            elif self.args.miccai2023:
                mask = torch.stack(samples['mask'], dim=0).squeeze(2).squeeze(0).to(img_embeds.device) # size [B, 49, 1]
                masked = img_embeds*mask
                img_embeds, atts_img = self.prompt_wrap_vqa(masked, atts_img, samples['question']) # apply mask to image embeddings
            else:
                img_embeds, atts_img = self.prompt_wrap_vqa(img_embeds, atts_img, samples['question'])
        else:
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]] # this is now the answer to the question

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device) # Answer tokens

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        ) # Answer tokens

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        ) # Empty targets with -100 values
        targets = torch.cat([empty_targets, targets], dim=1) # Empty targets with -100 values and answer tokens

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos) # bos token
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}_acc{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr'], eval_res['Accuracy']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    

    def validation_step(self, samples, batch_idx):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        if self.args.vqa:
            if self.args.ours:
                if self.args.baseline == 'complementary':
                    dr_embeds, atts_dr = self.encode_img([image[0]])
                    dr_embeds = self.layer_norm(dr_embeds)
                    co_embeds, atts_co = self.encode_img([image[1]])
                    co_embeds = self.layer_norm(co_embeds)
                    cr_embeds, atts_cr = self.encode_img([image[2]])
                    cr_embeds = self.layer_norm(cr_embeds)
                    img_embeds, atts_img = getattr(self, self.args.ours_version)(dr_embeds, atts_dr, co_embeds, atts_co, cr_embeds, atts_cr, samples['question'])
                else:
                    # if our method, use special wrapping where image an region are included in the prompt
                    region = samples['region']
                    region_embeds, atts_region = self.encode_img(region)
                    region_embeds = self.layer_norm(region_embeds)
                    img_embeds, atts_img = getattr(self, self.args.ours_version)(img_embeds, atts_img, region_embeds, atts_region, samples['question'])
            elif self.args.miccai2023:
                mask = torch.stack(samples['mask'], dim=0).squeeze(2).squeeze(0).to(img_embeds.device) # size [B, 49, 1]
                masked = img_embeds*mask
                img_embeds, atts_img = self.prompt_wrap_vqa(masked, atts_img, samples['question']) # apply mask to image embeddings
            else:
                img_embeds, atts_img = self.prompt_wrap_vqa(img_embeds, atts_img, samples['question'])
        else:
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        # * No adding of the EOS because that one should be present in the predicted text
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        with torch.cuda.amp.autocast(): #Add to avoid error about location of tensors, but then error outofmemory appears

            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=self.hparams.temperature,
            )
            hypo = [self.decode(i) for i in outputs]
            ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
            if self.args.vqa:
                self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["q_id"]}) # use question id if VQA
            else:
                self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
            return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        if self.args.vqa:
            if self.args.ours:
                if self.args.baseline == 'complementary':
                    dr_embeds, atts_dr = self.encode_img([image[0]])
                    dr_embeds = self.layer_norm(dr_embeds)
                    co_embeds, atts_co = self.encode_img([image[1]])
                    co_embeds = self.layer_norm(co_embeds)
                    cr_embeds, atts_cr = self.encode_img([image[2]])
                    cr_embeds = self.layer_norm(cr_embeds)
                    img_embeds, atts_img = getattr(self, self.args.ours_version)(dr_embeds, atts_dr, co_embeds, atts_co, cr_embeds, atts_cr, samples['question'])
                else:
                    # if our method, use special wrapping where image an region are included in the prompt
                    region = samples['region']
                    region_embeds, atts_region = self.encode_img(region)
                    region_embeds = self.layer_norm(region_embeds)
                    img_embeds, atts_img = getattr(self, self.args.ours_version)(img_embeds, atts_img, region_embeds, atts_region, samples['question'])
            elif self.args.miccai2023:
                mask = torch.stack(samples['mask'], dim=0).squeeze(2).squeeze(0).to(img_embeds.device) # size [B, 49, 1]
                masked = img_embeds*mask
                img_embeds, atts_img = self.prompt_wrap_vqa(masked, atts_img, samples['question']) # apply mask to image embeddings
            else:
                img_embeds, atts_img = self.prompt_wrap_vqa(img_embeds, atts_img, samples['question'])
        else:
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        if self.args.vqa:
            self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["q_id"]})
        else:
            self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()