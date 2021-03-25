from argparse import ArgumentParser
from pynm import pynm

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pheno_p",help="path to phenotype data",dest='pheno_p')
    parser.add_argument("--out_p",help="path to save restuls",dest='out_p')
    parser.add_argument("--confounds",help="list of confounds to use in gp model, formatted as a string with commas between confounds (column names from phenotype dataframe) and categorical confounds marked as C(my_confound).",default = 'age',dest='confounds')
    parser.add_argument("--conf",help="single confound to use in LOESS & centile models",default = 'age',dest='conf')
    parser.add_argument("--score",help="response variable, column title from phenotype dataframe",default = 'score',dest='score')
    parser.add_argument("--group",help="group, column title from phenotype dataframe",default = 'group',dest='group')
    return parser.parse_args()

def main():
    params = vars(_cli_parser())
    
    confounds = params['confounds'].split(',')            
    data = pd.read_csv(params['pheno_p'])
    
    py_nm = pynm.PyNM(data,params['score'],params['group'],params['conf'],confounds)
    
    #Add a column to data w/ number controls used in this bin
    py_nm.bins_num()
    
    #Run models
    py_nm.loess_normative_model()
    py_nm.centiles_normative_model()    
    py_nm.gp_normative_model()
    
    py_nm.data.to_csv(args.out_p,index=False)
    
if __name__ == "__main__":
    raise RuntimeError("`pynm/cli.py` should not be run directly. Please install `pynm`.")