from pynm import pynm
import numpy as np
import pandas as pd
import scipy.stats as sp
import math
import pytest
from pynm.util import *
import matplotlib.pyplot as plt
from unittest.mock import patch 
from sklearn.model_selection import train_test_split

def model(age, sex, offset):
    noise = np.random.normal(0, 0.1)
    return 0.001*age-0.00001*(age-50)**2+0.5 + noise - np.random.uniform(0, 0.3) * sex + offset


def model_prob(age, sex, offset):
    noise = np.random.normal(0, 0.1)
    return 0.001*age-0.00001*(age-50)**2+0.5 + noise - np.random.uniform(0, 0.3) * sex - 0.2 * np.random.uniform() + offset

# randseed = 3, sample_size = 1, n_sites = 2 has ONE PROB n=6
# randseed = 1, sample_size = 1, n_sites = 2 has NO PROB n=12
def generate_data(group='PROB_CON', sample_size=1, n_sites=2, randseed=3):
    np.random.seed(randseed)
    n_sites = n_sites
    age_min = (np.random.rand(n_sites)*50).astype(int)
    sites = pd.DataFrame(data={'sex_ratio': np.random.rand(n_sites),
                               'prob_ratio': 0.5*np.random.rand(n_sites),
                               'age_min': age_min,
                               'age_max': (age_min+5+np.random.rand(n_sites)*50).astype(int),
                               'score_shift': np.random.randn(n_sites)/4,
                               'sample_size': (sample_size+np.random.rand(n_sites)*sample_size*10).astype(int)})

    participants = []
    for site in sites.iterrows():
        for participant in range(int(site[1]['sample_size'])):
            sex = np.random.binomial(1, site[1]['sex_ratio'])
            prob = np.random.binomial(1, site[1]['prob_ratio'])
            age = np.random.uniform(site[1]['age_min'], site[1]['age_max'])
            if prob:
                score = model_prob(age, sex, site[1]['score_shift'])
            else:
                score = model(age, sex, site[1]['score_shift'])
            participants.append([site[0], sex, prob, age, score])

    df = pd.DataFrame(participants, columns=['site', 'sex', 'group', 'age', 'score'])
    df.sex.replace({1: 'Female', 0: 'Male'}, inplace=True)
    if group == 'PROB_CON':
        df.group.replace({1: 'PROB', 0: 'CTR'}, inplace=True)
    return df

def sample_x(low=1,high=100,n_subs=1000,sampling='full'):
    if sampling =='full':
        x = np.random.uniform(low=low,high=high,size=n_subs)
    else:
        x = np.concatenate([np.random.normal(20,10,size=int(n_subs/2)),np.random.normal(80,10,size=int(n_subs/2))])
        x = x[(x<high) & (x > low)]
    return x

# Homoskedastic, gaussian noise
def dataset_homo(low=1,high=100,n_subs=1000,sampling='full'):
    x = sample_x(low=low,high=high,n_subs=n_subs,sampling=sampling)
    scores = np.array([np.log(i) + np.random.randn() for i in x])
    df = pd.DataFrame([x,scores],index=['x','score']).transpose()
    df['train_sample'] = 1
    return df

# Homoskedastic, skew noise
def dataset_skew(low=1,high=100,n_subs=1000,sampling='full'):
    x = sample_x(low=low,high=high,n_subs=n_subs,sampling=sampling)
    scores = np.array([np.log(i) + sp.skewnorm.rvs(a=2,size=1)[0] for i in x])
    df = pd.DataFrame([x,scores],index=['x','score']).transpose()
    df['train_sample'] = 1
    return df

# Heteroskedastic linear
def dataset_het(low=1,high=100,n_subs=1000,sampling='full'):
    x = sample_x(low=low,high=high,n_subs=n_subs,sampling=sampling)
    scores = np.array([np.log(i) + 0.15*np.log(i)*np.random.randn() for i in x])
    df = pd.DataFrame([x,scores],index=['x','score']).transpose()
    df['train_sample'] = 1
    return df

class TestBasic:
    def test_read_confounds_some_categorical(self):
        conf = ['a', 'b', 'c(c)']
        clean, cat = read_confounds(conf)
        assert clean == ['a', 'b', 'c']
        assert cat == ['c']

    def test_read_confounds_no_categorical(self):
        conf = ['a', 'b', 'c']
        clean, cat = read_confounds(conf)
        assert clean == conf
        assert cat == []

    def test_read_confounds_all_categorical(self):
        conf = ['c(a)', 'c(b)', 'c(c)']
        clean, cat = read_confounds(conf)
        assert clean == ['a', 'b', 'c']
        assert cat == ['a', 'b', 'c']
    
    def test_invalid_init(self):
        data1 = generate_data(randseed=1)
        data2 = generate_data(randseed=2)
        data = pd.concat([data1,data2])
        assert data.index.nunique() != data.shape[0]
        with pytest.raises(ValueError):
            m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])

    def test_set_group_names_PROB_CON_all_CON(self):
        data = generate_data(randseed=1)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        assert m.CTR == 'CTR'
        assert m.PROB == 'PROB'

    def test_set_group_names_PROB_CON(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        assert m.CTR == 'CTR'
        assert m.PROB == 'PROB'

    def test_set_group_names_01(self):
        data = generate_data(randseed=3, group='01')
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        assert m.CTR == 0
        assert m.PROB == 1

    def test_set_group_controls(self):
        data = generate_data(randseed=3, group='01')
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],train_sample=1)
        assert m.group == 'group'
    
    def test_set_group_33(self):
        data = generate_data(randseed=3, group='01')
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],train_sample='0.33')
        assert m.group == 'train_sample'
        assert m.data['train_sample'].sum() == 1
        assert m.data[(m.data['train_sample']==1) & (m.data['group']== 1)].shape[0] == 0
    
    def test_set_group_manual_no_col(self):
        data = generate_data(randseed=3, group='01')
        with pytest.raises(ValueError):
            m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],train_sample='manual')
    
    def test_set_group_manual_zero_col(self):
        data = generate_data(randseed=3, group='01')
        data['train_sample'] = 0
        with pytest.raises(ValueError):
            m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],train_sample='manual')
    
    def test_set_group_manual_good_col(self):
        data = generate_data(randseed=3, group='01')
        data['train_sample'] = [1,1,0,0,0,0]
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],train_sample='manual')
        assert m.PROB == 0
        assert m.group == 'train_sample'

    def test_create_bins(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.centiles_normative_model()
        assert m.bins is not None

    def test_bins_num(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=5, bin_width=10)
        m._create_bins()
        assert len(m.bins) == 6

    def test_loess_rank(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.loess_normative_model()
        assert np.sum(m.data.LOESS_rank) == 1

    def test_loess_normative_model(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.loess_normative_model()
        assert math.isclose(2.3482, np.sum(m.data.LOESS_z), abs_tol=0.00001)

    def test_centiles_rank(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.centiles_normative_model()
        assert np.sum(m.data.Centiles_rank) == -22

    def test_centiles_normative_model(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.centiles_normative_model()
        assert np.sum(m.data.Centiles) == 446

    def test_get_masks(self):
        a = np.array(list(range(6)))
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        ctr, prob = m._get_masks()
        assert a[ctr].shape[0] == 5
        assert a[prob][0] == 3

    def test_get_masks_all_CON(self):
        a = np.array(list(range(12)))
        data = generate_data(randseed=1)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        ctr, prob = m._get_masks()
        assert a[ctr].shape[0] == 12
        assert a[prob].shape[0] == 0

    def test_get_conf_mat(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        conf_mat = m._get_conf_mat()
        assert conf_mat.shape[0] == 6
        assert conf_mat.shape[1] == 3
        for i in range(3):
            assert not isinstance(conf_mat[0, i], str)

    def test_use_approx_auto_small(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        assert m._use_approx(method='auto') == False

    def test_use_approx_auto_big(self):
        data = generate_data(randseed=3,sample_size=1000)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        assert m._use_approx(method='auto') == True

    def test_use_approx_approx(self):
        data = generate_data(randseed=3,sample_size=1000)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        assert m._use_approx(method='approx') == True
    
    def test_use_approx_exact(self):
        data = generate_data(randseed=3,sample_size=2000)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        with pytest.warns(Warning) as record:
            use_approx = m._use_approx(method='exact')
        assert len(record) == 1
        assert record[0].message.args[0] == "Exact GP model with over 2000 data points requires large amounts of time and memory, continuing with exact model."
        assert use_approx == False

    def test_gp_normative_model(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gp_normative_model()
        assert 'GP_pred' in m.data.columns
        assert math.isclose(0,m.data['GP_residuals'].mean(),abs_tol=0.5)

    def test_homo_res(self):
        data = dataset_homo()
        m = pynm.PyNM(data,'score','train_sample',['x'])
        with pytest.warns(None) as record:
            m.gp_normative_model(method='exact')
        assert len(record) == 0

    def test_nongaussian_res(self):
        data = dataset_skew()
        m = pynm.PyNM(data,'score','train_sample',['x'])
        with pytest.warns(Warning) as record:
            m.gp_normative_model(method='exact')
        assert len(record) == 1
        assert record[0].message.args[0] == "The residuals are not Gaussian!"

    def test_het_res(self):
        data = dataset_het()
        m = pynm.PyNM(data,'score','train_sample',['x'])
        with pytest.warns(Warning) as record:
            m.gp_normative_model(method='exact')
        assert len(record) == 1
        assert record[0].message.args[0] == "The residuals are heteroskedastic!"
    
    def test_rmse(self):
        y_true = np.array([1,2,3,4,5])
        y_pred = np.array([0,1,3,5,2])
        assert RMSE(y_true,y_pred) == np.sqrt(2.4)

    def test_smse(self):
        y_true = np.array([1,2,3,4,5])
        y_pred = np.array([0,1,3,5,2])
        assert SMSE(y_true,y_pred) == np.sqrt(2.4)/np.sqrt(2)
    
    def test_msll(self):
        y_true = np.array([1,2,3,4,5])
        y_pred = np.array([0,1,3,5,2])
        sigma = np.array([1,2,1,2,1])
        y_train_mean = 2
        y_train_sigma = 1

        term1 = 0.5*np.log(2*np.pi*np.array([1,4,1,4,1])) + np.array([1/2,1/8,0,1/8,9/2])
        term2 = 0.5*np.log(2*np.pi) + np.array([1/2,0,1/2,2,4.5])

        assert MSLL(y_true,y_pred,sigma,y_train_mean,y_train_sigma) == np.mean(term1 - term2)
    
    def test_report(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.loess_normative_model()
        #m.gp_normative_model()
        #m.gamlss_normative_model()
        m.report()

@patch("matplotlib.pyplot.show")
class TestPlot:
    def test_plot_default(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.plot()
        assert True
    
    def test_plot_default_two_models(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.loess_normative_model()
        assert m.plot() is None
    
    def test_plot_default_no_models(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        with pytest.warns(Warning) as record:
            m.plot()
    
    def test_plot_valid_subset(self,mock_patch):
        subset = ['Centiles','LOESS']
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.loess_normative_model()
        assert m.plot(kind=subset) is None
    
    def test_plot_invalid_subset1(self,mock_patch):
        subset = ['Centiles',None]
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        with pytest.raises(ValueError):
            m.plot(kind=subset)
    
    def test_plot_invalid_subset2(self,mock_patch):
        subset = ['Centiles','GAMLSS']
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        with pytest.raises(KeyError):
            m.plot(kind=subset)
    
    def test_plot_res_default(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        assert m.plot_res() is None
    
    def test_plot_res_default_two_models(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.loess_normative_model()
        assert m.plot_res() is None
    
    def test_plot_res_default_no_models(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        with pytest.raises(ValueError):
            m.plot_res()
    
    def test_plot_res_valid_subset(self,mock_patch):
        subset = ['Centiles','LOESS']
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.loess_normative_model()
        assert m.plot_res(kind=subset) is None
    
    def test_plot_res_invalid_subset1(self,mock_patch):
        subset = ['Centiles',None]
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        with pytest.raises(ValueError):
            m.plot_res(kind=subset)
    
    def test_plot_res_invalid_subset2(self,mock_patch):
        subset = ['Centiles','GAMLSS']
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        with pytest.raises(ValueError):
            m.plot_res(kind=subset)
    
    def test_plot_z_default(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        assert m.plot_z() is None
    
    def test_plot_z_default_two_models(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.loess_normative_model()
        assert m.plot_z() is None
    
    def test_plot_z_default_no_models(self,mock_patch):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        with pytest.raises(ValueError):
            m.plot_z()
    
    def test_plot_z_valid_subset(self,mock_patch):
        subset = ['Centiles','LOESS']
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        m.loess_normative_model()
        assert m.plot_z(kind=subset) is None
    
    def test_plot_z_invalid_subset1(self,mock_patch):
        subset = ['Centiles',None]
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        with pytest.raises(ValueError):
            m.plot_z(kind=subset)
    
    def test_plot_z_invalid_subset2(self,mock_patch):
        subset = ['Centiles','GAMLSS']
        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.centiles_normative_model()
        with pytest.raises(ValueError):
            m.plot_z(kind=subset)

class TestApprox:
    def test_svgp_init(self):
        from pynm.models.approx import SVGP

        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        conf_mat = m._get_conf_mat()
        score = m._get_score()

        X_train,X_test,y_train,y_test = train_test_split(conf_mat, score)
        svgp = SVGP(X_train,X_test,y_train,y_test)
    
    def test_svgp_train(self):
        from pynm.models.approx import SVGP

        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        conf_mat = m._get_conf_mat()
        score = m._get_score()

        X_train,X_test,y_train,y_test = train_test_split(conf_mat, score)
        svgp = SVGP(X_train,X_test,y_train,y_test)
        svgp.train(num_epochs = 2)

        assert len(svgp.loss) == 2
    
    def test_svgp_predict(self):
        from pynm.models.approx import SVGP

        data = generate_data(randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        conf_mat = m._get_conf_mat()
        score = m._get_score()

        X_train,X_test,y_train,y_test = train_test_split(conf_mat, score)
        svgp = SVGP(X_train,X_test,y_train,y_test)
        svgp.train(num_epochs = 2)
        means,sigmas = svgp.predict()

        assert means.size(0) == y_test.shape[0]
        assert sigmas.size(0) == y_test.shape[0]

    def test_svgp_model(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gp_normative_model(method='approx')

        assert 'GP_pred' in m.data.columns
        assert math.isclose(0, m.data['GP_residuals'].mean(), abs_tol=0.5)

class TestGAMLSS:
    def test_get_r_formulas(self):
        from pynm.models import gamlss

        g = gamlss.GAMLSS(mu='score ~ 1')
        mu,sigma,_,_ = g._get_r_formulas('score ~ cs(age) + site',None,None,None)
        #assert not isinstance(mu,str)
        assert mu == 'score ~ cs(age) + site'
        assert sigma == '~ 1'

    def test_gamlss(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gamlss_normative_model(mu='score ~ cs(age)',sigma='~ age + site',tau='~ c(sex)')
        assert 'GAMLSS_pred' in m.data.columns
    
    def test_gamlss_smse(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gamlss_normative_model(mu='score ~ cs(age)',sigma='~ age + site',tau='~ c(sex)')
        assert m.SMSE_GAMLSS > 0
    
    def test_gamlss_default_formulas(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gamlss_normative_model()
        assert 'GAMLSS_pred' in m.data.columns
    
    def test_gamlss_invalid_init(self):
        from pynm.models import gamlss
        
        with pytest.raises(ValueError):
            gamlss.GAMLSS()
    
    def test_gamlss_nan_issue(self):
        df = generate_data(n_sites=4,sample_size=35,randseed=650)
        #Initialize pynm w/ data and confounds
        m = pynm.PyNM(df,'score','group',['age','c(sex)','c(site)'])
        m.loess_normative_model()
        m.centiles_normative_model()
        m.gamlss_normative_model(mu='score ~ ps(age) + c(sex) + c(site)',sigma = '~ age',family='SHASHo2')

    def test_gamlss_random_effect(self):
        df = generate_data(n_sites=4,sample_size=35,randseed=650)
        #Initialize pynm w/ data and confounds
        m = pynm.PyNM(df,'score','group',
                confounds = ['age','c(sex)','c(site)'])
        m.gamlss_normative_model(mu='score ~ ps(age) + c(sex) + random(as.factor(site))',sigma = '~ ps(age)',family='SHASHo2',method='mixed(10,50)')
    
    def test_gamlss_random_effect_not_converged(self):
        #TODO: Force example where algorithm not converged warning gets thrown
        df = generate_data(n_sites=4,sample_size=35,randseed=650)
        #Initialize pynm w/ data and confounds
        m = pynm.PyNM(df,'score','group',
                confounds = ['age','c(sex)','c(site)'])
        m.gamlss_normative_model(mu='score ~ ps(age) + c(sex) + random(as.factor(site))',sigma = '~ ps(age)',family='SHASHo2',method='RS')
    
    def test_gamlss_bad_formula(self):
        df = generate_data(n_sites=4,sample_size=35,randseed=650)
        #Initialize pynm w/ data and confounds
        m = pynm.PyNM(df,'score','group',
                confounds = ['age','c(sex)','c(site)'])
        with pytest.raises(ValueError):
            m.gamlss_normative_model(mu='score ~ xxx(age) + c(sex) + c(site)',family='SHASHo2')

class TestCV:
    def test_cv_1_loess(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.loess_normative_model()
        assert math.isclose(2.3482, np.sum(m.data.LOESS_z), abs_tol=0.00001)
    
    def test_cv_3_loess(self):
        data = generate_data(n_sites=1,sample_size=100,randseed=650)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.loess_normative_model(cv_folds=3)
        assert not np.isnan(m.RMSE_LOESS)
        assert not np.isnan(m.SMSE_LOESS)
    
    def test_cv_1_centiles(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.centiles_normative_model()
        assert np.sum(m.data.Centiles) == 446
    
    def test_cv_3_centiles(self):
        data = generate_data(n_sites=1,sample_size=100,randseed=650)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'],bin_spacing=8,bin_width=1.5)
        m.centiles_normative_model(cv_folds=3)
        assert not np.isnan(m.RMSE_Centiles)
        assert not np.isnan(m.SMSE_Centiles)
    
    def test_cv_1_gp(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gp_normative_model()
        assert 'GP_pred' in m.data.columns
        assert math.isclose(0,m.data['GP_residuals'].mean(),abs_tol=0.5)
    
    def test_cv_3_gp(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gp_normative_model(cv_folds=3)
        assert 'GP_pred' in m.data.columns
        assert math.isclose(0,m.data['GP_residuals'].mean(),abs_tol=0.5)
    
    def test_cv_1_svgp(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gp_normative_model(method='approx')

        assert 'GP_pred' in m.data.columns
        assert math.isclose(0, m.data['GP_residuals'].mean(), abs_tol=0.5)
    
    def test_cv_3_svgp(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gp_normative_model(method='approx',cv_folds=3,num_epochs=3)

        assert 'GP_pred' in m.data.columns
        assert math.isclose(0, m.data['GP_residuals'].mean(), abs_tol=0.5)
    
    def test_cv_1_gamlss(self):
        data = generate_data(sample_size=4, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gamlss_normative_model(mu='score ~ cs(age)',sigma='~ age + site',tau='~ c(sex)')
        assert 'GAMLSS_pred' in m.data.columns
    
    def test_cv_3_gamlss(self):
        data = generate_data(sample_size=5, n_sites=2, randseed=3)
        m = pynm.PyNM(data,'score','group',['age','c(sex)','c(site)'])
        m.gamlss_normative_model(mu='score ~ cs(age)',sigma='~ age + site',tau='~ c(sex)',cv_folds=3)
        assert 'GAMLSS_pred' in m.data.columns