# 안쓰는 변수, 메모리 최적화 완료

import random
import os
import cv2
import numpy as np

print('cv2 version: ', cv2.__version__)
print('numpy version: ', np.__version__)

# edge colors
white = np.array([255, 255, 255])
yellow = np.array([255, 255, 0])
green = np.array([0, 255, 0])

# drawing line colors
BLUE, RED = (255, 0, 0), (0, 0, 255)

# 제거영역 마커
# cv2.GC_FGD(보존영역): 1 이고 cv2.GC_BGD(제거영역): 0 인데 
# 0을 쓰면 grabCut mask와 값이 충돌나서 따로 2를 사용
GC_BGD = 2

def resize_rc(rc, scale_factor):
    # scale factor에 맞게 rc를 resize
    rc_resized = np.array(rc) * scale_factor
    rc_resized = rc_resized.astype(np.uint8)  # 소수점 절삭
    return tuple(rc_resized)  # tuple 형태로 반환

def find_scale_factor(rc):

    # WATCH 이 방식이 잘 되는지 계속 확인 필요
    # threshold 기준값 적당한지도 계속 확인 필요
    _, _, w, h = rc
    max_size = max(w, h)  # 가로, 세로 중 긴 쪽
    threshold = 200  # 기준 픽셀값

    if max_size > threshold:  # 기준 픽셀값 보다 사이즈가 크면 줄이기
        scale_factor = (threshold / max_size)  # 긴 쪽 크기를 threshold 에 맞추도록 설정

    else:
        scale_factor = 1  # threshold 보다 작으면 그대로 둠

    # scale_factor 확인용
    # print("Scale factor: ", scale_factor)
    return scale_factor

def change_edge_color(edge, color):
    # 검정색이 아닌 모든 픽셀을 지정된 color로 변경
    edge[np.any(edge != [0, 0, 0], axis=-1)] = color
    return edge

def get_bounding_rc(rc, margin_ratio=0.2):

    # rc(사각형 영역)를 감싸는 조금 더 큰 사각형 반환
    x, y, w, h = rc

    # margin 비율 만큼 더 크게 잡기
    margin_w = int(w * margin_ratio)
    margin_h = int(h * margin_ratio)

    # 새로운 바운딩 박스의 좌표와 크기 ( + 경계 처리)
    x_new = max(0, x - margin_w)
    y_new = max(0, y - margin_h)
    w_new = min(img_origin.shape[1] - x_new, w + 2 * margin_w)  # 가로 길이 조정
    h_new = min(img_origin.shape[0] - y_new, h + 2 * margin_h)  # 세로 길이 조정

    return x_new, y_new, w_new, h_new

def get_bounding_rect(mask, margin_ratio=0.2):
    """
    마스크에서 파랑색 색칠영역을 포함하는 최소 사각형을 반환하며,
    가로와 세로에 지정된 비율로 마진을 추가합니다.

    Parameters:
        mask (numpy.ndarray): 2D numpy array, 마스크 이미지
        margin_ratio (float): 가로, 세로에 추가할 마진의 비율

    Returns:
        tuple: (x, y, w, h)로 구성된 여유 있는 바운딩 사각형의 좌표와 크기
    """

    # FGD 영역만 남김
    # 파란색 드로윙만 감싸는 영역으로 설정, 빨강은 삐져 나가도 댐
    # 이렇게 해야 빨강색으로 시원시원하게 영역 제거가 가능함
    mask = np.where(mask == cv2.GC_FGD, 1, 0).astype(np.uint8)

    # 원래 바운딩 박스 구하기
    rc_init = cv2.boundingRect(mask)  # tight 하게 감싸는 영역
    x, y, w, h = get_bounding_rc(rc_init, margin_ratio)  # 영역에 마진 추가

    return x, y, w, h

def find_maximal_edge(mask):

    # set mask2
    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR) * 100 # 100배 해줘야 구분되는 경계가 명확해짐
    
    # WATCH ksize 7 고정 문제 없는지 지켜봐야 함
    ksize = 7
    median_blurred = cv2.medianBlur(mask2, ksize)
    
    # Canny edge detection
    edges = cv2.Canny(median_blurred, 100, 200)

    # find contours
    # cv2 4 이전 버젼
    # _,contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2 4.x 버젼
    #contours, _ = cv2.findContours(
    #    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    try:
        _,contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        edge = np.zeros_like(edges)

        # update edge with largest contour
        cv2.drawContours(edge, [largest_contour], -1, 255, thickness=2)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # 2D to 3D

    else:
        edge = np.zeros_like(mask)  # 3D

    # 흰색 픽셀을 노란색(임시 edge)으로 변경
    edge = change_edge_color(edge, yellow)
    
    return edge

def on_mouse_main(event, x, y, flags, param):  # main 창의 마우스 이벤트 처리 (드로잉)

    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:

        # 보존 영역 처리 (Marekr: BLUE)
        # cv2.GC_FGD: 1
        cv2.circle(img, (x, y), marker_size, BLUE, -1) # -1: 테두리 없이 원을 채우는 옵션
        cv2.circle(drawing_mask, (x, y), marker_size, cv2.GC_FGD, -1)
        cv2.imshow('image_origin', img)
        drawing = True

    elif event == cv2.EVENT_RBUTTONDOWN:

        # 제거 영역 처리 (RED)
        # cv2.GC_BGD: 0
        # drawing_mask 에는 GC_BGD (2) 로 따로 저장, 0으로 하면 grabCut mask와 값이 겹침
        cv2.circle(img, (x, y), marker_size, RED, -1)
        cv2.circle(drawing_mask, (x, y), marker_size,
                   GC_BGD, -1)  # GC_BGD = 2 로 따로 저장
        cv2.imshow('image_origin', img)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:

        # 드래그 처리
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(img, (x, y), marker_size, BLUE, -1)
            cv2.circle(drawing_mask, (x, y), marker_size, cv2.GC_FGD, -1)
            cv2.imshow('image_origin', img)

        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(img, (x, y), marker_size, RED, -1)
            cv2.circle(drawing_mask, (x, y), marker_size,
                       GC_BGD, -1)  # GC_BGD = 2 로 따로 저장
            cv2.imshow('image_origin', img)

# -------------------Select image------------------------
# sample images
# img = cv2.imread('sample_imgs/ct2.jpg')
img = cv2.imread('sample_imgs/scopy/cju0qkwl35piu0993l0dewei2.jpg')
# img = cv2.imread('sample_imgs/x_ray.png') # 418 x 921, 축소된 x-ray 이미지
# img = cv2.imread('sample_imgs/x_ray3.jpg') # 2.63 MB -> 이제 잘 됨 (1440 x 4500)
# img = cv2.imread('sample_imgs/x_ray2.jpg') # 8.75 MB -> 이제 잘 됨 (3000 x 7200)
# img = cv2.imread('sample_imgs/x_ray largest.jpg') # 8.81 MB (3082 x 7720) 받은 sample 중 가장 큰 이미지 -> 잘 됨

'''
# 위에 이미지 안 쓸거면 내시경 이미지 랜덤 선택
folder_path = 'sample_imgs/scopy'
image_files = [f for f in os.listdir(
    folder_path) if f.lower().endswith(('.png', '.jpg'))]
random_image = random.choice(image_files)
image_path = os.path.join(folder_path, random_image)
img = cv2.imread(image_path)
# ---------------------------------------------------------
'''

img_origin = img.copy()  # 원본 백업, 초기화 할 때만 이거 사용

# 메인 창 설정
cv2.namedWindow("image_origin", cv2.WINDOW_AUTOSIZE)  # 이미지 크기로 자동으로 고정
# cv2.namedWindow("image_origin", cv2.WINDOW_FREERATIO)  # 프리 사이즈 (비율 무시)

# TODO delete this
# 중간 결과 확인용 보조창
# cv2.namedWindow("w1", cv2.WINDOW_FREERATIO) # 프리 사이즈 (비율 무시)
# cv2.namedWindow("w2", cv2.WINDOW_FREERATIO) # 프리 사이즈 (비율 무시)

# 사용자 지정 영역 마커 사이즈 초기화
# +(=), - 키로 크기 조절
marker_size_range = [1, 2, 4, 8, 16, 20, 40, 80]
marker_size_idx = 2
marker_size = marker_size_range[marker_size_idx]

# flags 초기화
drawing = False  # 마커가 그려져 있는지 체크
edge_selected = False  # edge 선택 모드인지 체크

# masking 영역에 필요한 배열 초기화
drawing_mask = np.zeros(img_origin.shape[:2], np.uint8) # 전체 이미지 크기와 같은 사이즈

cv2.setMouseCallback('image_origin', on_mouse_main)
cv2.imshow('image_origin', img)
img_with_edge = img.copy()  # 선택된 edge + img_origin
edges = []  # 전체 edge 저장

while True:

    key = cv2.waitKey()

    if key == 13:  # ENTER

        if not drawing:  # 선택 영역이 없는 경우
            print('Plese Select(draw) Area!')
            continue

        # 1. 드로잉 영역을 기반으로 적당한 RC 영역 설정
        rc = get_bounding_rect(drawing_mask, margin_ratio=0.25)  # 마스킹을 감싸는 영역
        rc2 = get_bounding_rc(rc, margin_ratio=0.1)  # rc 보다 조금 더 큰 영역
        x0, y0, w0, h0 = rc2  # down sizing 전 rect 따로 저장

        # 2. 새로 설정된 영역만큼 이미지와 마스킹 영역 자르기
        img_rc = img_origin[y0:y0+h0, x0:x0+w0]  # img crop
        drawing_mask_rc = drawing_mask[y0:y0+h0, x0:x0+w0]  # drawing mask crop

        # 3. Down sizing을 위한 scale 비율 설정 (현재 200 픽셀 초과되면 downsizing 진행)
        scale_factor = find_scale_factor(rc2)

        # 4. 이미지 및 rc 다운 사이징
        # img_rc_r: img_rc resized -> img_rc는 나중에 사용해야 해서 살려둠
        img_rc_r = cv2.resize(img_rc, (0, 0), fx=scale_factor, fy=scale_factor)
        drawing_mask_rc = cv2.resize(
            drawing_mask_rc, (0, 0), fx=scale_factor, fy=scale_factor)

        # Down sizing rc
        rc = resize_rc(rc, scale_factor)
        rc2 = resize_rc(rc2, scale_factor)
        x, y, w, h = rc2

        # rc2에 대한 상대 좌표로 rc 변경 (rc2의 왼쪽 위 꼭짓점이 0,0 이 되므로)
        # Non negative 보장 되도록 설계 (rc[0] >= x, rc[1] >= y)
        rc = (rc[0] - x, rc[1] - y, rc[2], rc[3])

        # 5. initialize grabcut parameters
        mask = np.zeros(img_rc_r.shape[:2], np.uint8) # downsizing 크기 만큼만 mask 세팅
        bgdModel = np.zeros((1, 65), np.float64)  # initialize bgd
        fgdModel = np.zeros((1, 65), np.float64)  # initialize fgd

        # 6. apply grabcut
        # 먼저 RECT를 적용해서 init
        try:
            cv2.grabCut(img_rc_r, mask, rc, bgdModel,
                        fgdModel, 1, cv2.GC_INIT_WITH_RECT)

            # 위의 grabCut으로 새로 생성된 mask에 사용자 영역 덮어 씌우기
            # FGD -> 1, BGD -> 0
            mask = np.where(drawing_mask_rc == cv2.GC_FGD, 1,
                            np.where(drawing_mask_rc == GC_BGD, 0, mask))

            # 위의 mask를 사용하여 GrabCut with MASK
            cv2.grabCut(img_rc_r, mask, rc, bgdModel,
                        fgdModel, 1, cv2.GC_INIT_WITH_MASK)

        except: # grabCut 에러 난 경우
            
            img = img_origin.copy()
            cv2.imshow('image_origin', img)  # 이미지 원상복원

            drawing_mask = np.zeros(img.shape[:2], np.uint8)  # 사용자 표시영역 초기화

            print('Grabcut Error Try other regions!!!!')
            continue
    
        # 7. Edge Select
        
        try:
            edge = find_maximal_edge(mask) # 가장 큰 edge 경계 찾기
        except:
            print('No Edge found... Try other regions')
            continue

        # edge resize and apply Interpolation
        edge_resized = cv2.resize(
            edge, (w0, h0), interpolation=cv2.INTER_CUBIC)
        img = img_origin.copy()

        # 8. 선택된 엣지 이미지에 병합하여 표기
        img[y0:y0+h0, x0:x0+w0] = cv2.bitwise_or(img_rc, edge_resized)
        cv2.imshow('image_origin', img)
        edge_selected = True

        print('-----------------------------------------')
        print('Press S to save selected Edge ')
        print('-----------------------------------------')
        continue

    elif key == ord('s') and edge_selected:  # save selected edge

        edge_resized = change_edge_color(
            edge_resized, green)  # 저장된 엣지는 초록색으로 변경
        edge_resized = cv2.medianBlur(edge_resized, 5)  # 확대된 엣지 smoothing

        img_with_edge_rc = cv2.bitwise_or(
            img_with_edge[y0:y0+h0, x0:x0+w0], edge_resized)
        img_with_edge[y0:y0+h0, x0:x0+w0] = img_with_edge_rc
        cv2.imshow('image_origin', img_with_edge)  # 병합된 edge 출력
        img = img_origin.copy()

        # grabCut에 필요한 배열들 초기화
        drawing_mask = np.zeros(img.shape[:2], np.uint8)

        # cv2.imshow('edge_resized', edge_resized)
        # cv2.waitKey(0)
        
        # edge 저장 시작 위치 (x0,y0) 와 w0 x h0 사이즈의 edge_resized 같이 저장
        edges.append((x0, y0, edge_resized))
        print('Edge saved!!!!!!!')
        print('Now we have ', len(edges), 'edge(s)')
        edge_selected = False
        drawing = False

    elif key == ord('='):  # marker size up, + 키

        if marker_size_idx < len(marker_size_range) - 1:
            marker_size_idx += 1
            marker_size = marker_size_range[marker_size_idx]

        print("Marker_size: ", marker_size)

    elif key == ord('-'):  # marker size down, - 키

        if marker_size_idx > 0:
            marker_size_idx -= 1
            marker_size = marker_size_range[marker_size_idx]

        print("Marker_size: ", marker_size)

    elif key == ord('r'):  # 전체 reset

        # 전체 이미지 및 drawing 초기화
        img = img_origin.copy()
        img_with_edge = img_origin.copy()
        drawing_mask = np.zeros(img_origin.shape[:2], np.uint8)
        cv2.imshow('image_origin', img)

        # flag들 및 edges 초기화
        edge_selected = False
        drawing = False
        edges = []
        print('All Edges Deleted')

    elif key == 27:  # esc
        break