from argparse import ArgumentParser
from pynm import pynm
import pandas as pd

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = ArgumentParser()
    parser.add_argument("--pheno_p",help="path to phenotype data",dest='pheno_p',required=True)
    parser.add_argument("--out_p",help="path to save restuls",dest='out_p',required=True)
    parser.add_argument("--confounds",help="list of confounds to use in gp model, formatted as a string with commas between confounds (column names from phenotype dataframe) and categorical confounds marked as C(my_confound).",default = 'age',dest='confounds')
    parser.add_argument("--conf",help="single confound to use in LOESS & centile models",default = 'age',dest='conf')
    parser.add_argument("--score",help="response variable, column title from phenotype dataframe",default = 'score',dest='score')
    parser.add_argument("--group",help="group, column title from phenotype dataframe",default = 'group',dest='group')
    return parser.parse_args()

def main():
    params = vars(_cli_parser())
    
    confounds = params['confounds'].split(',')            
    data = pd.read_csv(params['pheno_p'])
    
    m = pynm.PyNM(data,params['score'],params['group'],params['conf'],confounds)
    
    #Add a column to data w/ number controls used in this bin
    m.bins_num()
    
    #Run models
    m.loess_normative_model()
    m.centiles_normative_model()    
    m.gp_normative_model()
    
    m.data.to_csv(args.out_p,index=False)
    
if __name__ == "__main__":
    raise RuntimeError("`pynm/cli.py` should not be run directly. Please install `pynm`.")