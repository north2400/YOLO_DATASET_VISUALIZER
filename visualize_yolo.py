import numpy as np
import cv2
import os
import traceback

# Progress bar
def console_progressBar(pers):

    # プログレスバーの長さ
    max_length = 50

    # 進捗状況の位置の計算
    disp_num = int(max_length * pers)

    print("\r進捗[", end='')
    for i in range(disp_num):
        print("=", end='')
    print(">", end='')
    for i in range(max_length - int(disp_num)):
        print(" ", end='')
    print("] {:.1f}%完了".format(pers * 100), end='')

    if pers == 1.0:
        print()

# クラスごとの色定義
def define_color(class_id):
    match class_id:
        case 0:
            blue = 255
            green = 0
            red = 0
        case 1:
            blue = 0
            green = 255
            red = 0
        case 2:
            blue = 0
            green = 0
            red = 255
        case 3:
            blue = 255
            green = 255
            red = 0
        case 4:
            blue = 0
            green = 255
            red = 255
        case 5:
            blue = 255
            green = 0
            red = 255
        case 6:
            blue = 255
            green = 255
            red = 255
        case 7:
            blue = 63
            green = 0
            red = 0
        case 8:
            blue = 0
            green = 63
            red = 0
        case 9:
            blue = 0
            green = 0
            red = 63
        case 10:
            blue = 63
            green = 63
            red = 0
        case 11:
            blue = 0
            green = 63
            red = 63
    return blue, green, red

# 階層を無視して、ファイル絶対パスを取得します
def get_file_list(in_dir, mode=".txt"):
    file_list = []
    for fd_path, sb_folder, sb_file in os.walk(in_dir):
        for fil in sb_file:
            path = fd_path.replace(os.sep, '/') + "/" + fil
            root, ext = os.path.splitext(path)
            if ext == mode:
                file_list.append(path)
    return file_list

# ファイルパスから拡張子なしファイル名を取得する
def get_filename_without_ext_from_filepath(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


# 受け取った画像に受け取った座標の位置に矩形を描画（クラスごとに色を変更）
def rectangle_from_label(img_path = "./test_image.jpg", label_path="./test_label.txt"):

    def draw(img, cls, pt1_x, pt1_y, blue, green, red):
        cv2.putText(img,
                    text = "cls:" + str(cls),
                    org = (pt1_x, pt1_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color = (blue, green, red),
                    thickness = 1,
                    lineType=cv2.LINE_4)
        cv2.rectangle(img,
                    pt1=(pt1_x, pt1_y),
                    pt2=(pt2_x, pt2_y),
                    color = (blue, green, red),
                    thickness=1,
                    lineType=cv2.LINE_4,
                    shift=0)
        return img

    label = np.loadtxt(label_path, delimiter=' ', dtype="float32")
    img = cv2.imread(img_path)

    h = img.shape[0]
    w = img.shape[1]

    # labelが1行のときとそれ以外の時で分岐
    if label.ndim == 1:
        cls = int(label[0])
        blue, green, red = define_color(cls)

        # 受け取る矩形情報はYOLO形式（中心, 幅, 高さ, 正規化済み）のため左上端と右下端の点に変換
        pt1_x = int((label[1] - label[3]/2) * w)
        pt1_y = int((label[2] - label[4]/2) * h)
        pt2_x = int((label[1] + label[3]/2) * w)
        pt2_y = int((label[2] + label[4]/2) * h)

        img = draw(img, cls, pt1_x, pt1_y, blue, green, red)

    else:
        for i in range(label.shape[0]):
            cls = int(label[i, 0])
            blue, green, red = define_color(cls)

            # 正規化されているデータをピクセル値に戻す

            # 受け取る矩形情報はYOLO形式（中心, 幅, 高さ, 正規化済み）のため左上端と右下端の点に変換
            pt1_x = int((label[i,1] - label[i,3]/2) * w)
            pt1_y = int((label[i,2] - label[i,4]/2) * h)
            pt2_x = int((label[i,1] + label[i,3]/2) * w)
            pt2_y = int((label[i,2] + label[i,4]/2) * h)

            # YOLO形式 Center W H
            img = draw(img, cls, pt1_x, pt1_y, blue, green, red)
            
    name = get_filename_without_ext_from_filepath(img_path)
    cv2.imwrite("./labeled_images/" + name + ".jpg", img)

def main():
    f_list = get_file_list("D:\\kitamura\\Anaconda\\tool\\bbox_generater\\labels")
    i_list = get_file_list("D:\\kitamura\\Anaconda\\tool\\bbox_generater\\images", ".jpg")

    # 矩形描画処理
    for i, (img_path, label_path) in enumerate(zip(i_list, f_list)):
        console_progressBar(i/len(i_list))
        try:
            rectangle_from_label(img_path, label_path)
        except KeyboardInterrupt:
            exit()
        except:
            print("Err at ",label_path)
            traceback.print_exc()

main()



