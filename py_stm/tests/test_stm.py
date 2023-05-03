# test the stm.py file 
import py_stm
from utils import common_texts


# create unit tests for Stm constructor
def  test_constructor():
    # import sklearn and get a dataset that can be used with topic analysis
    import sklearn
    from sklearn.datasets import fetch_20newsgroups
    # get the dataset
    dataset = fetch_20newsgroups(subset='all')
    # turn this dataset which is a list of strings into a list of lists of words
    dataset = [list(x.split()) for x in dataset.data]

    print(dataset[:10])

if __name__ == "__main__":
    test_constructor()
    print("Everything passed")