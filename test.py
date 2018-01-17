
import os
from Main import main_func,initialize_train_model


## replace this path with your folders path where the 'car_front_images' folder and 'result.txt' are present ##
BASE_DIR = 'F:\DATASCIENCE\Solivar_Work\MLGuard\\'

def test():
    initialize_train_model()

    files = os.listdir(BASE_DIR + 'car_front_images')
    files.sort(key=lambda name: int(name.split('.')[0].split('car')[1]))

    fp = open(BASE_DIR + 'results.txt')
    contents = fp.read().split('\n')

    i = 0
    count1 = count2 = count3 = total_count2 = 0 ; total_count1 = 0

    for name in files:
        result = main_func(BASE_DIR + 'car_front_images\\' + name)
        if result == 0:
            print "No chars were found in the plate"
        else:
            if len(result) == len(contents[i]):
                total_count1+=1
                if result == contents[i]:
                    count1 += 1
                total_count2 += len(contents[i])
                j = 0
                while j < len(contents[i]) and j < len(result):
                    if contents[i][j] == result[j]:
                        count2 += 1
                    j += 1
            else:
                count3 += 1
        i += 1
    print count1, " plates predicted correctly out of", total_count1
    print count2, " predicted correctly out of ", total_count2
    print "Number of plates whose length is more/less than the original ", count3
    return

if __name__ == '__main__':
    test()