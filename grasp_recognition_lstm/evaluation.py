# later need to test this code here
import numpy as np

def test_accuracy(test_data,test_label,model,class_num = 4):
    total_test_num = test_data.shape[0]
    count_num = [0]*class_num
    for i in range(class_num):
        exec("total_test_label"+str(i)+"_num = 0")
        exec("total_test_label"+str(i)+"_rightnum = 0")

    for i in range(total_test_num):
        prediction = model.predict(test_data[i,:,:][np.newaxis,:,:])
        arg_max = np.argmax(prediction)

        for j in range(class_num):
            if test_label[i][j] == 1:
                count_num[j] += 1
                exec("total_test_label"+str(j)+"_num += 1")
                if arg_max == j:
                    exec("total_test_label"+str(j)+"_rightnum += 1")        
    for i in range(class_num):
        print('total_test_label'+str(i)+'_rightnum = ', end='')
        exec("print(100 * total_test_label{}_rightnum/total_test_label{}_num, '%')".format(i, i))

    print("count number of test: ",count_num)