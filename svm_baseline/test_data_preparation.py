'''
Test the data perparation for the SVM model

Author: Longshen Ou, 2024/07/25
'''

from data_preparation import parse_segment_info, float_dict


def main():
    rename_files()
    # test_bisec()
    # fp = '/home/longshen/work/SoloDetection/svm_baseline/segments.txt'
    # parse_segment_info(fp)

def test_bisec():
    from bisect import bisect

    keys = [0.2, 0.4, 0.6, 0.8, 1.0]
    # values = ['very low', 'low', 'medium', 'high', 'very high', 'overflow']
    values = ['very low', 'low', 'medium', 'high', 'very high']

    dic = float_dict(keys, values)
    print(dic[0.1])

    b = 2

def rename_files():
    dir = '/home/longshen/work/SoloDetection/svm_baseline/segmentation_outs'
    import os
    for file in os.listdir(dir):
        if file.endswith('.wav'):
            new_name = file.replace('.wav', '.txt')
            os.rename(os.path.join(dir, file), os.path.join(dir, new_name))


if __name__ == '__main__':
    main()