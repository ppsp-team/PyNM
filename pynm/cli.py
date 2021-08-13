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
                            "For GP model all confounds will be used, for LOESS and Centiles models "
                            "only the first is used. For GAMLSS all confounds are used "
                            "unless formulas are specified. Categorical values must be denoted by c(var) "
                            "('c' must be lower case), e.g. 'c(SEX)' for column name 'SEX'. "
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
    parser.add_argument("--train_sample",default='controls',dest='train_sample',
                        help="On what subset to train the model, can be 'controls', 'manual', "
                            "or a value in (0,1]. Default value is 'controls'.")
    parser.add_argument("--LOESS",dest='LOESS',action='store_true',
                        help="Flag to run LOESS model.")
    parser.add_argument("--centiles",dest='centiles',action='store_true',
                        help="Flag to run Centiles model.")
    parser.add_argument("--bin_spacing",default = -1,dest='bin_spacing',
                        help="Distance between bins for LOESS & centiles models.")
    parser.add_argument("--bin_width",default = -1,dest='bin_width',
                        help="Width of bins for LOESS & centiles models.")
    parser.add_argument("--GP",dest='GP',action='store_true',
                        help="Flag to run Gaussian Process model.")
    parser.add_argument("--gp_method",default = 'auto',dest='gp_method',
                        help="Method to use for the GP model. Can be set to "
                            "'auto','approx' or 'exact'. In 'auto' mode, "
                            "the exact model will be used for datasets smaller "
                            "than 2000 data points. SVGP is used for the approximate model. "
                            "See documentation for details. Default value is 'auto'.")
    parser.add_argument("--gp_num_epochs",default=20, dest='gp_num_epochs',
                        help="Number of training epochs for SVGP model. "
                            "See documentation for details. Default value is 20.")
    parser.add_argument("--gp_n_inducing",default=500,dest='gp_n_inducing',
                        help="Number of inducing points for SVGP model. "
                            "See documentation for details. Default value is 500.")
    parser.add_argument("--gp_batch_size",default=256,dest='gp_batch_size',
                        help="Batch size for training and predicting from SVGP model. "
                            "See documentation for details. Default value is 256.")
    parser.add_argument("--gp_length_scale",default=1,dest='gp_length_scale',
                        help="Length scale of Matern kernel for exact model. "
                            "See documentation for details. Default value is 1.")
    parser.add_argument("--gp_nu",default=2.5,dest='nu',
                        help="Nu of Matern kernel for exact and SVGP model. "
                            "See documentation for details. Default value is 2.5.")
    parser.add_argument("--GAMLSS",dest='GAMLSS',action='store_true',
                        help="Flag to run GAMLSS.")
    parser.add_argument("--gamlss_mu",default=None,dest='gamlss_mu',
                        help="Formula for mu (location) parameter of GAMLSS. Default "
                        "formula for score is sum of confounds with non-categorical "
                        "columns as smooth functions, e.g. 'score ~ ps(age) + sex'.")
    parser.add_argument("--gamlss_sigma",default=None,dest='gamlss_sigma',
                        help="Formula for mu (location) parameter of GAMLSS. Default "
                        "formula is '~ 1'.")
    parser.add_argument("--gamlss_nu",default=None,dest='gamlss_nu',
                        help="Formula for mu (location) parameter of GAMLSS. Default "
                        "formula is '~ 1'.")
    parser.add_argument("--gamlss_tau",default=None,dest='gamlss_tau',
                        help="Formula for mu (location) parameter of GAMLSS. Default "
                        "formula is '~ 1'.")
    parser.add_argument("--gamlss_family",default='SHASHo2',dest='gamlss_family',
                        help="Family of distributions to use for fitting, default is 'SHASHo2'. "
                        "See R documentation for GAMLSS package for other available families of distributions.")
    parser.add_argument("--gamlss_what",default='mu',dest='gamlss_what',
                        help="What parameter for GAMLSS to predict, can be 'mu', 'sigma', 'nu' or 'tau'. "
                        "Default is 'mu'.")
    parser.add_argument("--gamlss_lib_loc",default=None,dest='gamlss_lib_loc',
                        help="Path to location of installed GAMLSS package. Default is None.")
    return parser.parse_args()

def main():
    params = vars(_cli_parser())
    
    confounds = params['confounds'].split(',')            
    data = pd.read_csv(params['pheno_p'])
    
    m = pynm.PyNM(data,params['score'],params['group'],params['conf'],confounds,params['train_sample'],
                bin_spacing=params['bin_spacing'], bin_width=params['bin_width'])
    
    #Run models
    if params['LOESS']:
        m.loess_normative_model()
        m.bins_num()
    if params['centiles']:
        m.centiles_normative_model()
        m.bins_num()
    if params['GP']:   
        m.gp_normative_model(length_scale=params['gp_length_scale'],nu=params['gp_nu'], 
                        method=params['gp_method'],batch_size=params['gp_batch_size'],
                        n_inducing=params['gp_n_inducing'],num_epochs=params['gp_num_epochs'])
    if args.GAMLSS:
        m.gamlss_normative_model(mu=params['gamlss_mu'],sigma=params['gamlss_sigma'],nu=params['gamlss_nu'],
                        tau=params['gamlss_tau'],family=params['gamlss_family'],what=params['gamlss_what'],lib_loc=params['gamlss_lib_loc'])
    
    m.data.to_csv(params['out_p'],index=False)
    
if __name__ == "__main__":
    raise RuntimeError("`pynm/cli.py` should not be run directly. Please install `pynm`.")