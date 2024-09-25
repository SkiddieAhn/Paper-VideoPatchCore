import torch

def get_yolo():
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    return yolo_model.eval()


def detect_nob(last_frame, yolo_model, device, confidence=.4, coi=[0]):
    '''
    No bactch allowed.
    Return:
        areas_filters: list of np.arrays of [xmin, ymin, xmax, ymax, conf, class]
    '''
    results = yolo_model(last_frame)
    areas = results.xyxy[0]
    if device == 'cpu':
        areas = areas.numpy()
    else:
        areas = areas.cpu().detach().numpy()
    areas_filtered = [area for area in areas if area[4] > confidence and area[-1] in coi]        
    return areas_filtered


def get_resized_area(areas, max_w, max_h, factor_x=1.5, factor_y=1.2):
    '''
    Should run it for each batch (for one area set from one frame_t)
    >>> batch_areas = detect(model, batch_frame_t)
    >>> for i in range(batch_size):
    >>>     new_areas = get_resized_area(batch_areas[i])
    '''
    if factor_x == 0 and factor_y == 0:
        return areas

    else:
        new_areas = []
        for area in areas:
            xmin_, ymin_, xmax_, ymax_, conf, cls_token = area

            xmin = xmin_ - (factor_x-1)*(xmax_-xmin_)
            ymin = ymin_ - (factor_y-1)*(ymax_-ymin_)
            xmax = xmax_ + (factor_x-1)*(xmax_-xmin_)
            ymax = ymax_ + (factor_y-1)*(ymax_-ymin_)

            x = xmax - xmin
            y = ymax - ymin

            if y > x:
                dif = (y-x)/2
                xmax += dif
                xmin -= dif
            elif y < x:
                dif = (x-y)/2
                ymax += dif
                ymin -= dif
                
            if xmin < 0:
                xmin = 0

            if ymin < 0:
                ymin = 0

            if(xmax > max_w):
                xmax = max_w

            if(ymax > max_h):
                ymax = max_h

            new_areas.append([xmin, ymin, xmax, ymax, conf, cls_token])

        return new_areas