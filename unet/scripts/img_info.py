import requests
import os
import json
import glob
import matplotlib.pyplot as plt
from shutil import copyfile


class ISICApi(object):
    def __init__(self, hostname='https://isic-archive.com',
                 username=None, password=None):
        self.baseUrl = hostname + '/api/v1'
        self.authToken = None

        if username is not None:
            if password is None:
                password = input('Password for user: ' + username)
            self.authToken = self._login(username, password)

    def _makeUrl(self, endpoint):
        return self.baseUrl + '/' + endpoint

    def _login(self, username, password):
        authResponse = requests.get(
            self._makeUrl('user/authentication'),
            auth=(username, password)
        )
        if not authResponse.ok:
            raise Exception('Login error: ' + authResponse.json()["message"])

        authToken = authResponse.json()['authToken']['token']
        return authToken

    def get(self, endpoint):
        url = self._makeUrl(endpoint)
        headers = {'Girder-Token': self.authToken} if self.authToken else None
        return requests.get(url, headers=headers)

    def getJson(self, endpoint):
        return self.get(endpoint).json()

    def getJsonList(self, endpoint):
        endpoint += '&' if '?' in endpoint else '?'
        LIMIT = 50
        offset = 0
        while True:
            resp = self.get(
                endpoint + 'limit=' + str(LIMIT) + '&offset=' + str(offset)
            ).json()
            if not resp:
                break
            for elem in resp:
                yield elem
            offset += LIMIT


def dump_data():
    files = os.listdir('/home/qasima/isic_2018/ISIC2018_Task1_Training_Input/')

    username = "ahmad.qasim@tum.de"
    password = "Teemeres@12"

    api = ISICApi(username=username, password=password)

    imageDetails = []

    for i, f in enumerate(files):
        try:
            f = f.split('.')[0]
            imageList = api.getJson('image?name=' + f)
            imageDetail = api.getJson('image/%s' % imageList[0]['_id'])
            imageDetails.append(imageDetail)
        except:
            break

        print("Collected Meta-Data for Image: ", i)

    with open('./meta_data.json', 'w') as fout:
        json.dump(imageDetails, fout, indent=4, sort_keys=True)


def manipulate_data():
    unique_type = []
    ids = {"melanoma": [], "seborrheic keratosis": [], "nevus": []}
    nums = {"melanoma": 0, "seborrheic keratosis": 0, "nevus": 0}

    file = "./meta_data.json"

    source_img = "../isic2018/train/images/"
    source_mask = "../isic2018/train/segmentation_masks/"

    dest_img = "../isic2018/test/images/"
    dest_mask = "../isic2018/test/segmentation_masks/"

    with open(file, 'r') as f:
        objects = json.load(f)

    for obj in objects:
        diag = obj["meta"]["clinical"]["diagnosis"]

        nums[diag] += 1
        ids[diag].append(obj["name"])

        if diag not in unique_type:
            unique_type.append(diag)

    for cls in ids.keys():
        names = ids[cls]
        cls_num = int(len(names) * 0.1)
        names = names[0:cls_num]
        for name in names:
            os.rename(source_img + name + ".jpg", dest_img + name + ".jpg")
            os.rename(source_mask + name + "_segmentation.png", dest_mask + name + "_segmentation.png")


def get_classes(cls="seborrheic keratosis"):
    file = "./meta_data.json"
    source_img = "/home/qasima/segmentation_models.pytorch/isic2018/data/train/images/"
    dest_img = "/home/qasima/isic2018/seb/"

    source_list = os.listdir(source_img)

    with open(file, 'r') as f:
        objects = json.load(f)

    for obj in objects:
        diag = obj["meta"]["clinical"]["diagnosis"]

        if diag == cls:
            if obj["name"] + '.jpg' in source_list:
                copyfile(source_img + obj["name"] + '.jpg', dest_img + obj["name"] + '.jpg')


if __name__ == "__main__":
    get_classes()
