
import os
import json
import re
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
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
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        if 'question' in features:
            question = features['question']
            to_return['question'] = question
        if 'q_id' in features:
            q_id = features['q_id']
            to_return['q_id'] = q_id
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)

class FieldParserRegions(FieldParser):
    def __init__(
            self,
            args
    ):
        super().__init__(args)
        self.baseline = args.baseline
        self.regions = args.regions
        self.digit2word = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', 
                            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

    def get_mask(self, mask_coords, mask_size, mask_type = 'rectangle'):
        # From locvqa repo, changed to return numpy array instead of tensor
        # mask_coords has the format ((y,x), h, w)
        if mask_type == 'ellipse': # requires ellipse regions (DECIDE FROM MASK_COORDS FORMAT)
            mask_ref = Image.new('L', mask_size, 0)
            mask = ImageDraw.Draw(mask_ref)
            mask.ellipse([(mask_coords[0][1], mask_coords[0][0]),(mask_coords[0][1] + mask_coords[2], mask_coords[0][0] + mask_coords[1])], fill=1)
            mask = np.array(mask_ref)
        else:
            mask = np.zeros(mask_size, dtype=np.uint8)
            mask[mask_coords[0][0]:mask_coords[0][0]+mask_coords[1] , mask_coords[0][1]:mask_coords[0][1]+mask_coords[2]] = 1
        return mask

    def draw_region(self, img, coords, r=2, mask_type = 'rectangle'):
        # From locvqa repo, changed to return numpy array instead of tensor
        if mask_type == 'ellipse': # requires ellipse regions
            img_ref = T.ToPILImage()(img)
            ((y,x), h, w) = coords
            draw = ImageDraw.Draw(img_ref)
            draw.ellipse([(x, y),(x + w, y + h)], outline='red')
            img_ref = np.array(img_ref)
            return img_ref
        elif mask_type == 'rectangle': # requires rectangle regions
            ((y,x), h, w) = coords

            for i in range(3):
                img[y-r:y+h+r, x-r:x+r, i] = 0
                img[y-r:y+r, x-r:x+w+r, i] = 0
                img[y-r:y+h+r, x+w-r:x+w+r, i] = 0
                img[y+h-r:y+h+r, x-r:x+w+r, i] = 0

            # set red channel line to red
            img[y-r:y+h+r, x-r:x+r, 0] = 255
            img[y-r:y+r, x-r:x+w+r, 0] = 255
            img[y-r:y+h+r, x+w-r:x+w+r, 0] = 255
            img[y+h-r:y+h+r, x-r:x+w+r, 0] = 255
            return img
        else:
            raise NotImplementedError

    # override parse method
    def parse(self, features):
        to_return = {'id': features['id']} # image name
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        images = []
        masks = []
        for image_path in features['image_path']: # ! although capable of handling multiple images, only one image is used in the dataset
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                array_copy = array.copy()
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                if self.regions:
                    mask = self.get_mask(features['mask_coords'], features['mask_size'], mask_type = features['mask_type']) # get mask
                    mask_copy = mask.copy()
                # Now, depending on baseline, apply mask to image or draw region on image
                if self.baseline == 'crop_region':
                    array = array * np.expand_dims(mask, axis=2) # apply mask to image
                elif self.baseline == 'draw_region' or 'region_in_text_sep_t' in self.baseline: # region_in_text_sep_tX is just to test (see pp. 32)
                    array = self.draw_region(array, features['mask_coords'], mask_type = features['mask_type']) # draw region on image
                elif self.baseline == 'context_only':
                    # multiply array with complement of the mask
                    array = array * (1 - np.expand_dims(mask, axis=2))
                elif self.baseline == 'complementary':
                    # multiply array with complement of the mask
                    array_co = array * (1 - np.expand_dims(mask, axis=2))
                    # also create the crop_region image
                    array_cr = array_copy.copy() * np.expand_dims(mask, axis=2)
                    # concatenate original image from array_copy and crop_region image from array
                if self.baseline != 'complementary':
                    image = self._parse_image(array) # applies vision model pre-processing
                    images.append(image)
                else:
                    image_dr = self._parse_image(self.draw_region(array_copy, features['mask_coords'], mask_type = features['mask_type'])) # draw region on image)
                    image_co = self._parse_image(array_co)
                    image_cr = self._parse_image(array_cr)
                    images.append(image_dr) 
                    images.append(image_co)
                    images.append(image_cr)
                # masks
                # resize mask to 7x7
                mask = T.Resize(7, antialias=None, interpolation = T.InterpolationMode.NEAREST)(torch.from_numpy(np.array(mask, dtype=np.uint8)).unsqueeze(0))
                mask = mask.view(49,-1) # tailored to swin transformer
                masks.append(mask)
        to_return["image"] = images
        to_return["mask"] = masks
        if 'question' in features:
            if self.baseline == 'region_in_text':
                question = features['question_alt']
                to_return['question'] = question
            elif self.baseline == 'region_in_text_sep': # baseline in which digits are separated by spaces
                # try to shorten question. This works only for insegcat and ris because all regions are rectangles, but for DME, modify
                question = re.sub(' +', ' ', "".join([" "+i + " " if i.isdigit() else i for i in features['question_alt']]))  # adding spaces between digits so that model can do better 
                to_return['question'] = question
            elif self.baseline == 'region_in_text_sep_t1': # pp. 32
                # try to shorten question. This works only for insegcat and ris because all regions are rectangles, but for DME, modify
                question = re.sub(' +', ' ', "".join([" "+i + " " if i.isdigit() else i for i in features['question_alt']]))  # adding spaces between digits so that model can do better 
                to_return['question'] = question
            elif self.baseline == 'region_in_text_sep_t2': # pp. 32
                # try to shorten question. This works only for insegcat and ris because all regions are rectangles, but for DME, modify
                question = re.sub(' +', ' ', "".join([" "+i + " " if i.isdigit() else i for i in features['question_alt']]))  # adding spaces between digits so that model can do better 
                # send object to the end of the question
                obj = question.split('is there ')[1].split(' in the region')[0]
                q_no_obj = question.split('in the region')[1].replace(' ?', '')
                question = 'in the region' + q_no_obj + ' is there ' + obj + '?'  
                to_return['question'] = question
            else:
                question = features['question']
                to_return['question'] = question
        if 'q_id' in features:
            q_id = features['q_id']
            to_return['q_id'] = q_id
        if 'mask_coords' in features:
            mask_coords = features['mask_coords']
            to_return['mask_coords'] = mask_coords
        if 'mask_size' in features:
            mask_size = features['mask_size']
            to_return['mask_size'] = mask_size
        to_return['region'] = [self._parse_image(np.zeros_like(array_copy))] # For questions about whole image, return empty region
        if (self.baseline == 'mask' or self.baseline == 'draw_region') and features['question_type'] == 'region':
            if features['mask_type'] == 'ellipse':
                array_copy = array_copy * np.expand_dims(mask_copy, axis=2) # apply mask to image
                region = array_copy[mask_coords[0][0]:mask_coords[0][0]+mask_coords[1] , mask_coords[0][1]:mask_coords[0][1]+mask_coords[2], :]
            else:
                region = array_copy[mask_coords[0][0]:mask_coords[0][0]+mask_coords[1] , mask_coords[0][1]:mask_coords[0][1]+mask_coords[2], :]
            to_return['region'] = [self._parse_image(region)] # same pre-processing as image
        elif self.baseline == 'context_only' and features['question_type'] == 'region':
            array_copy = array_copy * np.expand_dims(mask_copy, axis=2)
            to_return['region'] = [self._parse_image(array_copy)]
        return to_return

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        if args.regions:
            self.parser = FieldParserRegions(args)
        else:
            self.parser = FieldParser(args)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


