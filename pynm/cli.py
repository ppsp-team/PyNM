from argparse import ArgumentParser
from pynm import pynm
import pandas as pd

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = ArgumentParser()
    parser.add_argument("--pheno_p",dest='pheno_p',required=True,
                        help="Path to phenotype data. Data must be in a .csv file.")
    parser.add_argument("--out_p",dest='out_p',required=True,
                        help="Path to output directory.")
    parser.add_argument("--confounds",default = 'age',dest='confounds',
                        help="List of confounds to use in the GP model."
                            "The list must formatted as a string with commas between confounds, "
                            "each confound must be a column name from the phenotype .csv file. "
                            "Categorical confounds must be denoted by with C(): "
                            "e.g. 'C(SEX)' for column name 'SEX'. "
                            "Default value is 'age'.")
    parser.add_argument("--conf",default = 'age',dest='conf',
                        help="Single numerical confound to use in LOESS & centile models. "
                            "Must be a column name from the phenotype .csv file. "
                            "Default value is 'age'.")
    parser.add_argument("--score",default = 'score',dest='score',
                        help="Response variable for all models. "
                        "Must be a column title from phenotype .csv file. "
                        "Default value is 'score'.")
    parser.add_argument("--group",default = 'group',dest='group',
                        help="Column name from the phenotype .csv file that "
                        "distinguishes probands from controls. The column must be "
                        "encoded with str labels using 'PROB' for probands and 'CTR' for controls "
                        "or with int labels using 1 for probands and 0 for controls. "
                        "Default value is 'group'.")
    parser.add_argument("--method",default = 'auto',dest='method',
                        help="Method to use for the GP model. Can be set to "
                            "'auto','approx' or 'exact'. In 'auto' mode, "
                            "the exact model will be used for datasets smaller "
                            "than 1000 data points. SVGP is used for the approximate model. "
                            "See documentation for details. Default value is 'auto'.")
    parser.add_argument("--num_epochs",default=20, dest='num_epochs',
                        help="Number of training epochs for SVGP model. "
                            "See documentation for details. Default value is 20.")
    parser.add_argument("--n_inducing",default=500,dest='n_inducing',
                        help="Number of inducing points for SVGP model. "
                            "See documentation for details. Default value is 500.")
    parser.add_argument("--batch_size",default=256,dest='batch_size',
                        help="Batch size for training and predicting from SVGP model. "
                            "See documentation for details. Default value is 256.")
    parser.add_argument("--length_scale",default=1,dest='length_scale',
                        help="Length scale of Matern kernel for exact model. "
                            "See documentation for details. Default value is 1.")
    parser.add_argument("--nu",default=2.5,dest='nu',
                        help="Nu of Matern kernel for exact and SVGP model. "
                            "See documentation for details. Default value is 2.5.")
    parser.add_argument("--train_sample",default='controls',dest='train_sample',
                        help="On what subset to train the model, can be 'controls', 'manual', "
                            "or a value in (0,1]. Default value is 'controls'.")
    return parser.parse_args()

def main():
    params = vars(_cli_parser())
    
    confounds = params['confounds'].split(',')            
    data = pd.read_csv(params['pheno_p'])
    
    m = pynm.PyNM(data,params['score'],params['group'],params['conf'],confounds,params['train_sample'])
    
    #Add a column to data w/ number controls used in this bin
    m.bins_num()
    
    #Run models
    m.loess_normative_model()
    m.centiles_normative_model()    
    m.gp_normative_model(length_scale=args.length_scale,nu=args.nu, method=args.method,batch_size=args.batch_size,n_inducing=args.n_inducing,num_epochs=args.num_epochs)
    
    m.data.to_csv(args.out_p,index=False)
    
if __name__ == "__main__":
    raise RuntimeError("`pynm/cli.py` should not be run directly. Please install `pynm`.")