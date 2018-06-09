import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np

def main():
    with open("3a_layer_outputs.pkl", "rb") as f:
        outputs = pickle.load(f)

    in_bn1 = outputs['layer1'].cpu().data.numpy().reshape(-1, 1)
    in_bn3 = outputs['layer3'].cpu().data.numpy().reshape(-1, 1)
    in_bn5 = outputs['layer5'].cpu().data.numpy().reshape(-1, 1)
    in_bn7 = outputs['layer7'].cpu().data.numpy().reshape(-1, 1)
    GetHistogram(in_bn1, '1st', '')
    GetHistogram(in_bn3, '2nd', '')
    GetHistogram(in_bn5, '3rd', '')
    GetHistogram(in_bn7, '7th', '')
    
    mean = [np.mean(in_bn1), np.mean(in_bn3), np.mean(in_bn5), np.mean(in_bn7)]
    var = [np.var(in_bn1), np.var(in_bn3), np.var(in_bn5), np.var(in_bn7)]
    PlotCharacterics('Mean', mean)
    PlotCharacterics('Variance', var)

    GetHistogram((in_bn1-mean[0])/var[0], '1st', '_norm')
    GetHistogram((in_bn3-mean[1])/var[1], '2nd', '_norm')
    GetHistogram((in_bn5-mean[2])/var[2], '3rd', '_norm')
    GetHistogram((in_bn6-mean[3])/var[3], '7th', '_norm')  

def PlotCharacterics(type, data):
    fname = '3a'+type+'.png'
    ax1 = plt.figure()
    ax1.xlabel("Layer")
    ax1.ylabel(type)
    ax1.title("Running "+type)
    ax1.plot([1,3,5,7], data)
    ax1.savefig(fname)

def GetHistogram(output, label, norm):

    fname = '3a_hist_layer'+label+norm+'.png'
    plt.figure()
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title("Histogram of " + label + " Layer")
    plt.hist(output, 50) 
    plt.savefig(fname)


if __name__ == '__main__':
    main()