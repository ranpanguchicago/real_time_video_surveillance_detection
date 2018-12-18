import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def movement_training_1(video_filepath, k=31, nb_frame=0, alpha=0.02,
                      motion_thresh=30, come_thresh=0.13, leave_thresh=0.13, leave_thresh_m=3,
                      lt_tlx=250, lt_tly=270, lt_brx=310, lt_bry=320,
                      rt_tlx=580, rt_tly=270, rt_brx=640, rt_bry=320):
    feed = cv2.VideoCapture(video_filepath)
    nb_pixels_in_motion = []
    nb_pixels_in_motion_lt = []
    nb_pixels_in_motion_rt = []
    running_avg = None
    running_avg_lt = None
    running_avg_rt = None
    count_coming = 0
    count_leaving = 0
    train_num = 0
    global text
    text = ''
    while feed.isOpened():
        nb_frame += 1
        current_frame = feed.read()[1]
        if current_frame is None:
            break
        cv2.rectangle(current_frame, (lt_tlx, lt_tly), (lt_brx, lt_bry), (0, 255, 0), 3)
        cv2.rectangle(current_frame, (rt_tlx, rt_tly), (rt_brx, rt_bry), (0, 255, 0), 3)
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        smooth_frame = cv2.GaussianBlur(gray_frame, (k, k), 0)
        smooth_frame_lt = smooth_frame[lt_tly:lt_bry, lt_tlx:lt_brx]
        smooth_frame_rt = smooth_frame[rt_tly:rt_bry, rt_tlx:rt_brx]
        if running_avg is None:
            running_avg = np.float32(smooth_frame)
        if running_avg_lt is None:
            running_avg_lt = np.float32(smooth_frame_lt)
        if running_avg_rt is None:
            running_avg_rt = np.float32(smooth_frame_rt)
        diff = cv2.absdiff(np.float32(smooth_frame), np.float32(running_avg))
        diff_lt_rec = cv2.absdiff(np.float32(smooth_frame_lt), np.float32(running_avg_lt))
        diff_rt_rec = cv2.absdiff(np.float32(smooth_frame_rt), np.float32(running_avg_rt))
        cv2.accumulateWeighted(np.float32(smooth_frame), running_avg, alpha)
        cv2.accumulateWeighted(np.float32(smooth_frame_lt), running_avg_lt, alpha)
        cv2.accumulateWeighted(np.float32(smooth_frame_rt), running_avg_rt, alpha)
        _, subtracted = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
        _, subtracted_lt = cv2.threshold(diff_lt_rec, motion_thresh, 255, cv2.THRESH_BINARY)
        _, subtracted_rt = cv2.threshold(diff_rt_rec, motion_thresh, 255, cv2.THRESH_BINARY)
        nb_pixels_in_motion.append({'Frame': nb_frame,
                                    'RatioPixel': np.count_nonzero(subtracted)/subtracted.size})
        nb_pixels_in_motion_lt.append({'Frame': nb_frame,
                                       'RatioPixel': np.count_nonzero(subtracted_lt)/subtracted_lt.size})
        frame_motion_lt = nb_pixels_in_motion_lt[-1]['RatioPixel']
        nb_pixels_in_motion_rt.append({'Frame': nb_frame,
                                       'RatioPixel': np.count_nonzero(subtracted_rt)/subtracted_rt.size})
        frame_motion_rt = nb_pixels_in_motion_rt[-1]['RatioPixel']
        frame_motion_diff = frame_motion_lt - frame_motion_rt
        if count_coming == 0:
            if (frame_motion_lt > come_thresh or frame_motion_rt > come_thresh) and \
                    frame_motion_diff > come_thresh:
                count_coming += 1
                print("COMING DIRECTION: left at frame " + str(nb_pixels_in_motion_lt[-1]['Frame']))
                if text == '':
                    text = "COMING DIRECTION: left at frame " + str(nb_pixels_in_motion_lt[-1]['Frame'])
                else:
                    text = text + '\n' + "COMING DIRECTION: left at frame " + str(nb_pixels_in_motion_lt[-1]['Frame'])
            if (frame_motion_lt > come_thresh or frame_motion_rt > come_thresh) and \
                    frame_motion_diff < -come_thresh:
                count_coming += 1
                print("COMING DIRECTION: right at frame " + str(nb_pixels_in_motion_lt[-1]['Frame']))
                if text == '':
                    text = "COMING DIRECTION: right at frame " + str(nb_pixels_in_motion_lt[-1]['Frame'])
                else:
                    text = text + '\n' + "COMING DIRECTION: right at frame " + str(nb_pixels_in_motion_lt[-1]['Frame'])
        else:
            if (frame_motion_lt < leave_thresh or frame_motion_rt < leave_thresh or
                frame_motion_lt/frame_motion_rt > leave_thresh_m) and \
                    frame_motion_diff > leave_thresh and count_leaving == 0:
                count_leaving += 1
                print("LEAVING DIRECTION: left at frame " + str(nb_pixels_in_motion_lt[-1]['Frame']))
                text = text + '\n' + "LEAVING DIRECTION: left at frame " + str(nb_pixels_in_motion_lt[-1]['Frame'])
            if (frame_motion_lt < leave_thresh or frame_motion_rt < leave_thresh or
                frame_motion_rt/frame_motion_lt > leave_thresh_m) and \
                    frame_motion_diff < -leave_thresh and count_leaving == 0:
                count_leaving += 1
                print("LEAVING DIRECTION: right at frame " + str(nb_pixels_in_motion_lt[-1]['Frame']))
                text = text + '\n' + "LEAVING DIRECTION: right at frame " + str(nb_pixels_in_motion_lt[-1]['Frame'])
        if count_coming == count_leaving and count_coming * count_leaving != 0:
            count_coming = 0
            count_leaving = 0
            train_num += 1
            print("NUMBER OF TRAINS PASSING BY: " + str(train_num))
            text = text + '\n' + "NUMBER OF TRAINS PASSING BY: " + str(train_num) + '\n' + ''
        smooth_frame = cv2.cvtColor(smooth_frame, cv2.COLOR_GRAY2BGR)
        subtracted = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2BGR)
        img_out = np.zeros((int(current_frame.shape[0]), int(current_frame.shape[1] + current_frame.shape[1] / 2), 3),
                           dtype=np.uint8)
        img_out[0:int(current_frame.shape[0]), 0:int(current_frame.shape[1]), :] = \
            cv2.resize(current_frame, (int(current_frame.shape[1]), int(current_frame.shape[0])))
        img_out[0:int(current_frame.shape[0] / 2), int(current_frame.shape[1]):int(current_frame.shape[1] + current_frame.shape[1] / 2), :] = \
            cv2.resize(smooth_frame, (int(current_frame.shape[1] / 2), int(current_frame.shape[0] / 2)))
        img_out[int(current_frame.shape[0] / 2):int(current_frame.shape[0]), int(current_frame.shape[1]):int(current_frame.shape[1] + current_frame.shape[1] / 2), :] = \
            cv2.resize(subtracted, (int(current_frame.shape[1] / 2), int(current_frame.shape[0] / 2)))
        y0, dy = 20, 20
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(img_out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.imshow('Result', img_out)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    df = pd.DataFrame(nb_pixels_in_motion)
    df = df.set_index('Frame')
    df_lt = pd.DataFrame(nb_pixels_in_motion_lt)
    df_lt = df_lt.set_index('Frame')
    df_rt = pd.DataFrame(nb_pixels_in_motion_rt)
    df_rt = df_rt.set_index('Frame')
    cv2.destroyAllWindows()
    feed.release()
    return df, df_lt, df_rt, train_num


def adjustment(video_filepath_1, video_filepath_2, video_filepath_3):
    counter = 0
    parameter = []
    for al in np.arange(0.019, 0.021, 0.001):
        for m_t in np.arange(29, 31, 1):
            for c_t in np.arange(0.12, 0.14, 0.01):
                for l_t in np.arange(0.12, 0.14, 0.01):
                    motion_df1, motion_df_lt1, motion_df_rt1, train_num1 = movement_training(video_filepath_1, alpha=al, motion_thresh=m_t, come_thresh=c_t, leave_thresh=l_t)
                    motion_df2, motion_df_lt2, motion_df_rt2, train_num2 = movement_training(video_filepath_2, alpha=al, motion_thresh=m_t, come_thresh=c_t, leave_thresh=l_t)
                    motion_df3, motion_df_lt3, motion_df_rt3, train_num3 = movement_training(video_filepath_3, alpha=al, motion_thresh=m_t, come_thresh=c_t, leave_thresh=l_t)
                    if train_num1 == 1 and train_num2 == 2 and train_num3 == 3:
                        parameter.append([al, m_t, c_t, l_t])
                    counter += 1
                    print(counter)
                    print(parameter)
    return parameter[int(len(parameter)/2)]


if __name__ == "__main__":
    print("'train_training_1.mp4' video test:")
    motion_df_1, motion_df_lt, motion_df_rt, _ = movement_training_1('test_video/train_training.mp4')
    plt.plot(motion_df_lt)
    plt.plot(motion_df_rt)
    print('')
    print("'train_training_1_reversed.mp4' reversed video test")
    motion_df_2, motion_df_lt, motion_df_rt, _ = movement_training_1('test_video/train_training_reversed.mp4')
    plt.plot(motion_df_lt)
    plt.plot(motion_df_rt)
    plt.show()
