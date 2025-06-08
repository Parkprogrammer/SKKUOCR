"""
This code is adapted from https://github.com/JaidedAI/EasyOCR/blob/8af936ba1b2f3c230968dc1022d0cd3e9ca1efbb/easyocr/recognition.py
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import cv2
import re

from .model import Model
from .utils import CTCLabelConverter


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / np.maximum(10, high + low), high, low


def adjust_contrast_grey(img, target: float = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / np.maximum(10, high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(
            np.full(img.shape, 0),
            np.minimum(
                np.full(img.shape, 255),
                img,
            ),
        ).astype(np.uint8)
    return img

def count_digits(text):
    """Count the number of digits in the text."""
    return sum(c.isdigit() for c in text)


def count_special_chars(text: str) -> int:
    """
    문자열에서 숫자, 영문, 한글이 아닌 모든 문자(예: -, ., /, ~ 등)의 개수를 반환합니다.
    """
    pattern = re.compile(r'[^0-9A-Za-z\uAC00-\uD7A3]')
    return len(pattern.findall(text))

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type: str = "right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = (img[:, :, w - 1].unsqueeze(2).expand(
                c,
                h,
                self.max_size[2] - w,
            ))

        return Pad_img


class ListDataset(torch.utils.data.Dataset):

    def __init__(self, image_list: list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, "L")


class AlignCollate(object):

    def __init__(self, imgH: int, imgW: int, adjust_contrast: float):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = True  # Do Not Change
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images = batch

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            # augmentation here - change contrast
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target=self.adjust_contrast)
                image = Image.fromarray(image, "L")

            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors


def recognizer_predict(model, converter, test_loader, opt2val: dict):
    device = opt2val["device"]

    model.eval()
    result = []
    with torch.no_grad():
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)
            inputs = image_tensors.to(device)
            preds = model(inputs)  # (N, length, num_classes)

            # rebalance
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()
            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)
            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            # Select max probabilty (greedy decoding), then decode index to character
            preds_lengths = torch.IntTensor([preds.size(1)] *
                                            batch_size)  # (N,)
            _, preds_indices = preds_prob.max(2)  # (N, length)
            preds_indices = preds_indices.view(-1)  # (N*length)
            preds_str = converter.decode_greedy(preds_indices, preds_lengths)

            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                result.append([pred, confidence_score.item()])

    return result

#숫자 중심
def second_recognizer_predict(model, converter, test_loader, opt2val: dict):
    device = opt2val["device"]

    model.eval()
    result = []
    with torch.no_grad():
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)
            inputs = image_tensors.to(device)
            preds = model(inputs)  # (N, length, num_classes)

            # rebalance
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()
            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)
            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            # Select max probabilty (greedy decoding), then decode index to character
            preds_lengths = torch.IntTensor([preds.size(1)] * batch_size)  # (N,)
            _, preds_indices = preds_prob.max(2)  # (N, length)
            preds_indices = preds_indices.view(-1)  # (N*length)
            preds_str = converter.decode_greedy(preds_indices, preds_lengths)

            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                digit_count = count_digits(pred)
                adjusted_confidence = confidence_score.item() + 0.2 * digit_count
                result.append([pred, adjusted_confidence])

    return result

#특수 기호 중심
def third_recognizer_predict(model, converter, test_loader, opt2val: dict):
    device = opt2val["device"]

    model.eval()
    result = []
    with torch.no_grad():
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)
            inputs = image_tensors.to(device)
            preds = model(inputs)  # (N, length, num_classes)

            # rebalance
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()
            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)
            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            # Select max probabilty (greedy decoding), then decode index to character
            preds_lengths = torch.IntTensor([preds.size(1)] * batch_size)  # (N,)
            _, preds_indices = preds_prob.max(2)  # (N, length)
            preds_indices = preds_indices.view(-1)  # (N*length)
            preds_str = converter.decode_greedy(preds_indices, preds_lengths)

            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                special_count = count_special_chars(pred)
                adjusted_confidence = confidence_score.item() + 0.2 * special_count
                result.append([pred, adjusted_confidence])

    return result


def get_recognizer(opt2val: dict):
    """
    :return:
        recognizer: recognition net
        converter: CTCLabelConverter
    """
    # converter
    vocab = opt2val["vocab"]
    converter = CTCLabelConverter(vocab)

    # recognizer
    recognizer = Model(opt2val)

    # state_dict
    rec_model_ckpt_fp = opt2val["rec_model_ckpt_fp"]
    device = opt2val["device"]
    state_dict = torch.load(rec_model_ckpt_fp, map_location=device)

    if device == "cuda":
        recognizer = torch.nn.DataParallel(recognizer).to(device)
    else:
        # TODO temporary: multigpu 학습한 뒤 ckpt loading 문제
        from collections import OrderedDict

        def _sync_tensor_name(state_dict):
            state_dict_ = OrderedDict()
            for name, val in state_dict.items():
                name = name.replace("module.", "")
                state_dict_[name] = val
            return state_dict_

        state_dict = _sync_tensor_name(state_dict)

    recognizer.load_state_dict(state_dict)   

    return recognizer, converter

'''DEBUG'''
def rotate_image(image, angle):
    """Rotate image by angle degrees with padding to prevent cropping."""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the size of the new bounding box
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated


# def get_text(image_list, recognizer, converter, opt2val: dict, original_img: np.ndarray):
#     imgW = opt2val["imgW"]
#     imgH = opt2val["imgH"]
#     adjust_contrast = opt2val["adjust_contrast"]
#     batch_size = opt2val["batch_size"]
#     n_workers = opt2val["n_workers"]
#     contrast_ths = opt2val["contrast_ths"]

#     # TODO: figure out what is this for
#     # batch_max_length = int(imgW / 10)

#     coord = [item[0] for item in image_list]
#     img_list = [item[1] for item in image_list]
#     AlignCollate_normal = AlignCollate(imgH, imgW, adjust_contrast)
#     test_data = ListDataset(img_list)
#     test_loader = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=n_workers,
#         collate_fn=AlignCollate_normal,
#         pin_memory=True,
#     )


#     '''DEBUG'''
#     contrast_ths = 0.2
    
#     # predict first round
#     result1 = second_recognizer_predict(recognizer, converter, test_loader, opt2val)

#     # predict second round
#     low_confident_idx = [
#         i for i, item in enumerate(result1) if (item[1] < contrast_ths)
#     ]

    
#     # if len(low_confident_idx) > 0:
#     #     img_list2 = [img_list[i] for i in low_confident_idx]
#     #     AlignCollate_contrast = AlignCollate(imgH, imgW, adjust_contrast)
#     #     test_data = ListDataset(img_list2)
#     #     test_loader = torch.utils.data.DataLoader(
#     #         test_data,
#     #         batch_size=batch_size,
#     #         shuffle=False,
#     #         num_workers=n_workers,
#     #         collate_fn=AlignCollate_contrast,
#     #         pin_memory=True,
#     #     )
#     #     result2 = recognizer_predict(recognizer, converter, test_loader,
#     #                                  opt2val)
    
#     if len(low_confident_idx) > 0:
#         img_list2 = [img_list[i] for i in low_confident_idx]
#         results2 = []

#         for img in img_list2:
#             best_result = None
#             best_confidence = 0

#             # Original image OCR
#             AlignCollate_contrast = AlignCollate(imgH, imgW, adjust_contrast)
#             test_data = ListDataset([img])
#             test_loader = torch.utils.data.DataLoader(
#                 test_data,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=n_workers,
#                 collate_fn=AlignCollate_contrast,
#                 pin_memory=True,
#             )
#             result = second_recognizer_predict(recognizer, converter, test_loader, opt2val)[0]
#             # print(f"Original Text: {result[0]}, Confidence: {result[1]}")

#             # Compare confidence with rotation
#             for angle in [90, 270]:
#                 rotated_img = rotate_image(img, angle)
#                 test_data = ListDataset([rotated_img])
#                 test_loader = torch.utils.data.DataLoader(
#                     test_data,
#                     batch_size=1,
#                     shuffle=False,
#                     num_workers=n_workers,
#                     collate_fn=AlignCollate_contrast,
#                     pin_memory=True,
#                 )
#                 rotated_result = second_recognizer_predict(recognizer, converter, test_loader, opt2val)[0]
#                 rotated_confidence = rotated_result[1]

#                 # Display the rotated image
#                 # cv2.imshow(f'Rotated {angle} degrees', rotated_img)
#                 # print(f"Rotated {angle} degrees Text: {rotated_result[0]}, Confidence: {rotated_confidence}")
#                 # cv2.waitKey(0)  # Wait for a key press to move to the next image
#                 # cv2.destroyAllWindows()

#                 if rotated_confidence > best_confidence:
#                     best_result = rotated_result
#                     best_confidence = rotated_confidence

#             # Append the best result to results2
#             results2.append(best_result if best_result else result)

#         # 이제 results2를 result2로 설정
#         result2 = results2

#     result = []
#     for i, zipped in enumerate(zip(coord, result1)):
#         box, pred1 = zipped
#         if i in low_confident_idx:
#             pred2 = result2[low_confident_idx.index(i)]
#             if pred1[1] > pred2[1]:
#                 result.append((box, pred1[0], pred1[1]))
#             else:
#                 result.append((box, pred2[0], pred2[1]))
#         else:
#             result.append((box, pred1[0], pred1[1]))

#     '''DEBUG'''
#     img_with_boxes = original_img.copy()  # 원본 이미지 복사
#     # for idx in low_confident_idx:
#     for idx in range(len(img_list)):
#         bbox = coord[idx]
#         bbox = np.array(bbox).astype(int)
#         cv2.polylines(img_with_boxes, [bbox], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨간색으로 표시

#     # 이미지 시각화
#     cv2.imshow("Low Confidence Areas", img_with_boxes)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     '''DEBUG'''
#     a=1

#     return result

def get_text(image_list, recognizer, converter, opt2val: dict, original_img: np.ndarray):
    imgW = opt2val["imgW"]
    imgH = opt2val["imgH"]
    adjust_contrast = opt2val["adjust_contrast"]
    batch_size = opt2val["batch_size"]
    n_workers = opt2val["n_workers"]
    contrast_ths = opt2val["contrast_ths"]

    # TODO: figure out what is this for
    # batch_max_length = int(imgW / 10)

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    AlignCollate_normal = AlignCollate(imgH, imgW, adjust_contrast)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        collate_fn=AlignCollate_normal,
        pin_memory=True,
    )


    '''DEBUG'''
    contrast_ths = 0.2
    
    # predict first round
    result1 = second_recognizer_predict(recognizer, converter, test_loader, opt2val)

    # predict second round
    low_confident_idx = [
        i for i, item in enumerate(result1) if (item[1] < contrast_ths)
    ]

    
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        AlignCollate_contrast = AlignCollate(imgH, imgW, adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=AlignCollate_contrast,
            pin_memory=True,
        )
        result2 = recognizer_predict(recognizer, converter, test_loader,
                                     opt2val)
   
    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1] > pred2[1]:
                result.append((box, pred1[0], pred1[1]))
            else:
                result.append((box, pred2[0], pred2[1]))
        else:
            result.append((box, pred1[0], pred1[1]))

    '''DEBUG'''
    img_with_boxes = original_img.copy()  # 원본 이미지 복사
    # for idx in low_confident_idx:
    for idx in range(len(img_list)):
        bbox = coord[idx]
        bbox = np.array(bbox).astype(int)
        cv2.polylines(img_with_boxes, [bbox], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨간색으로 표시

    # 이미지 시각화
    # cv2.imshow("Low Confidence Areas", img_with_boxes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    '''DEBUG'''
    a=1

    return result