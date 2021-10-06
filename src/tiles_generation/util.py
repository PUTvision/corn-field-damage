import cv2


def show_small_img(img, name='img', show=True):
    if not show:
        return

    small_height = 512
    shape = img.shape[:2]
    img_small = cv2.resize(img, (small_height, int(small_height / shape[1] * shape[0])))
    cv2.imshow(name, img_small)
    cv2.waitKey(1)
