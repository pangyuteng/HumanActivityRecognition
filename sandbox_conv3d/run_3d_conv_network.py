import os, sys
from optparse import OptionParser
import numpy
from pylearn2.testing import skip
from pylearn2.config import yaml_parse
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.mlp import MLP
from pylearn2.space import Conv3DSpace
    
class ToyDataset(DenseDesignMatrix):
    def __init__(self):
        # simulated random dataset
        rng = numpy.random.RandomState(seed=42)
        data = rng.normal(size=(200, 28,28,28))
        self.y = numpy.random.binomial(1, 0.5, (200, 1))
        super(ToyDataset, self).__init__(X=data, y=self.y, y_labels=2)
        
def get_dataset_toy():
    """
    The toy dataset is only meant to used for testing pipelines.
    Do not try to visualize weights on it. It is not picture and
    has no color channel info to support visualization
    """
    trainset = ToyDataset()
    testset = ToyDataset()

    return trainset, testset

def get_mlp(structure):
    shape, channels = structure
    print('!',shape)
    print('!',channels)
    config = { 
        'layers':
        'input_space':Conv3DSpace(shape=shape,channels=channels)
        }
    return MLP(**config)

def main(args=None):
    # ref. run_deep_trainer.py
    
    """
    args is the list of arguments that will be passed to the option parser.
    The default (None) means use sys.argv[1:].
    """
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="dataset", default="toy",
                     help="specify the dataset, either cifar10, mnist or toy")
    (options,args) = parser.parse_args(args=args)
    
    if options.dataset == 'toy':
        trainset, testset = get_dataset_toy()
        n_output = 2
    else:
        NotImplementedError()

    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '.'))
    save_path = os.path.dirname(os.path.realpath(__file__))
    save_path = save_path.replace('\\', r'\\')

    layers = []
    layers.append(get_mlp([[28,28,28],1]))

if __name__ == "__main__":
    main()