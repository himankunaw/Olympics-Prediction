import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,titl=None, save_path=None,  correction=1.0, line=False):
    plt.figure()
    plt.scatter(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    
    if line:
        xx=np.arange(0, max(y), 0.1)
        yy = xx
        plt.plot(xx,yy,'r')
        
    if titl:
        plt.title(titl)
    else:
        plt.title(x.name)
        
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_hist(df, col_name, save_path=None):
    plt.hist(df[col_name], bins=25)
    plt.xlabel('Training Examples')
    plt.ylabel('Number of medals won')
    plt.title('Histogram of medals won')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_clf(clf_perf, save_path=None):
    plt.bar(range(len(clf_perf)), list(clf_perf.values()), align='center')
    plt.xticks(range(len(clf_perf)), list(clf_perf.keys()))
    plt.xlabel('Classification Algorithm')
    plt.ylabel('Probability Classified Correctly')
    plt.title('Binary Classifier Performance')
    plt.ylim([0,1.0])
    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.4)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_reg(reg_perf, save_path=None):
    X_axis = np.arange(len(reg_perf))  
    total_std_dev = [array[0] for array in reg_perf.values()]
    tops_std_dev = [array[1] for array in reg_perf.values()]
    plt.subplots_adjust(bottom=0.35)
    plt.bar(X_axis - 0.2, total_std_dev, 0.4, label = 'All Countries')
    plt.bar(X_axis + 0.2, tops_std_dev, 0.4, label = 'Top 10 Scoring Countries')
    plt.xticks(range(len(reg_perf)), list(reg_perf.keys()))
    plt.ylim([0,25])
    plt.xlabel('Regressor')
    plt.ylabel('Average Std Dev [Medals]')
    plt.title('Regression Algorithm Performance')
    plt.xticks(rotation=70)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_before_after(reg_perf_before, reg_perf_after, save_path=None):
    X_axis = np.arange(len(reg_perf_before))  
    total_std_dev = [array[0] for array in reg_perf_before.values()]
    tops_std_dev = [array[0] for array in reg_perf_after.values()]
    plt.subplots_adjust(bottom=0.35)
    plt.bar(X_axis - 0.2, total_std_dev, 0.4, label = 'Before Tuning')
    plt.bar(X_axis + 0.2, tops_std_dev, 0.4, label = 'After Tuning')
    plt.xticks(range(len(reg_perf_before)), list(reg_perf_before.keys()))
    plt.ylim([0,10])
    plt.xlabel('Regressor')
    plt.ylabel('Average Std Dev [Medals]')
    plt.title('Regression Algorithm Performance')
    plt.xticks(rotation=70)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show() 