import matplotlib.pyplot as plt
from train_training_1 import movement_training_1
from train_training_2 import movement_training_2


if __name__ == "__main__":
    print("'train_training_1.mp4' video test:")
    motion_df, motion_df_lt, motion_df_rt, _ = movement_training_1('test_video/train_training_1.mp4')
    motion_df.plot()
    motion_df_lt.plot()
    motion_df_rt.plot()
    plt.show()
    print('')
    print("'train_training_1_reversed.mp4' reversed video test")
    motion_df, motion_df_lt, motion_df_rt, _ = movement_training_1('test_video/train_training_1_reversed.mp4')
    motion_df.plot()
    motion_df_lt.plot()
    motion_df_rt.plot()
    plt.show()

    print('')
    print("'train_training_2.mp4' video test")
    motion_df, motion_df_lt, motion_df_rt, _ = movement_training_2('test_video/train_training_2.mp4')
    motion_df.plot()
    motion_df_lt.plot()
    motion_df_rt.plot()
    plt.show()
