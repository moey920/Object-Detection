import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
 
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./tran.json'):
        '''
        :param labelme_json: 모든 labelme의 json 파일 경로로 이루어진 리스트
        :param save_json_path: json저장위치
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
 
        self.save_json()
 
    def data_transfer(self):
 
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # json 파일 추가
                self.images.append(self.image(data, num))
                for shapes in data['shapes']:
                    label = shapes['label']
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    points = shapes['points'] #여기 있는 포인트는 rectangle로 표시하여 얻은 점 두 개로, 네 개 점으로 옮겨야 한다.
                    points.append([points[0][0],points[1][1]])
                    points.append([points[1][0],points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1
 
    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data['imageData'])  #원격 이미지 데이터 해석
        # img=io.imread(data['imagePath']) # 이미지 경로를 통해 이미지 열기
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]
 
        self.height = height
        self.width = width
 
        return image
 
    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'Receipt'
        categorie['id'] = len(self.label) + 1  # 0 묵시적 배경
        categorie['name'] = label
        return categorie
 
    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) #list를 사용하여 json 파일을 저장하면 잘못 보고됨
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 이 방식을 사용하여 list로 전환
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.getcatid(label) #주의, 소스코드 묵인1
        annotation['id'] = self.annID
        return annotation
 
    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1
 
    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 경계선을 그리다
        # cv2.fillPoly(img, [np.asarray(points)], 1)  #다각형 그리기 내부 픽셀 값 1
        polygons = points
 
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)
 
    def mask2box(self, mask):
        '''mask에서 그 테두리를 역산하다
        mask：[h,w]  0、1로 구성된 이미지
        1대응 대상, 1대응 행렬 번호(좌측 상단 대열 번호, 우측 하단 대열 번호만 계산하면 그 테두리를 계산할 수 있다)
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 해석좌상각행렬호
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x
 
        # 해석우하각행렬호
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
 
        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] COCO에 대응하는 bbox 양식
 
    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask
 
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco
 
    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # json 파일 저장
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 더 보기 좋게 표시
 
 
labelme_json = glob.glob('C:/Users/Cashcow005/Desktop/Object-Detection/images/annotation/train/001.json')
# labelme_json=['./Annotations/*.json']
 
labelme2coco(labelme_json, 'C:/Users/Cashcow005/Desktop/Object-Detection/images/annotation/train/coco/test.json')
#labelme2coco(labelme_json, './json/test.json')
