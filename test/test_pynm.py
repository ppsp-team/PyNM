import pynm.pynm as pynm
import numpy as np
import pandas as pd
import math
import pytest


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


class TestBasic:
    def test_read_confounds_some_categorical(self):
        conf = ['a', 'b', 'C(c)']
        clean, cat = pynm.read_confounds(conf)
        assert clean == ['a', 'b', 'c']
        assert cat == ['c']

    def test_read_confounds_no_categorical(self):
        conf = ['a', 'b', 'c']
        clean, cat = pynm.read_confounds(conf)
        assert clean == conf
        assert cat == []

    def test_read_confounds_all_categorical(self):
        conf = ['C(a)', 'C(b)', 'C(c)']
        clean, cat = pynm.read_confounds(conf)
        assert clean == ['a', 'b', 'c']
        assert cat == ['a', 'b', 'c']

    def test_set_group_names_PROB_CON_all_CON(self):
        data = generate_data(randseed=1)
        m = pynm.PyNM(data)
        assert m.CTR == 'CTR'
        assert m.PROB == 'PROB'

    def test_set_group_names_PROB_CON(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data)
        assert m.CTR == 'CTR'
        assert m.PROB == 'PROB'

    def test_set_group_names_01(self):
        data = generate_data(randseed=3, group='01')
        m = pynm.PyNM(data)
        assert m.CTR == 0
        assert m.PROB == 1

    def test_create_bins(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data)
        m.create_bins()
        assert True

    def test_bins_num(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data)
        m.create_bins(bin_spacing=5, bin_width=10)
        assert len(m.bins) == 6

    def test_loess_rank(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data)
        m.loess_normative_model()
        m.loess_rank()
        assert np.sum(m.data.LOESS_rank) == 1

    def test_loess_normative_model(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data)
        m.loess_normative_model()
        assert math.isclose(2.3482, np.sum(m.data.LOESS_nmodel), abs_tol=0.00001)

    def test_centiles_rank(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data)
        m.centiles_normative_model()
        m.centiles_rank()
        assert np.sum(m.data.Centiles_rank) == -19

    def test_centiles_normative_model(self):
        data = generate_data(randseed=11)
        m = pynm.PyNM(data)
        m.centiles_normative_model()
        assert np.sum(m.data.Centiles_nmodel) == 446

    def test_get_masks(self):
        a = np.array(list(range(6)))
        data = generate_data(randseed=3)
        m = pynm.PyNM(data)
        ctr, prob = m.get_masks()
        assert a[ctr].shape[0] == 5
        assert a[prob][0] == 3

    def test_get_masks_all_CON(self):
        a = np.array(list(range(12)))
        data = generate_data(randseed=1)
        m = pynm.PyNM(data)
        ctr, prob = m.get_masks()
        assert a[ctr].shape[0] == 12
        assert a[prob].shape[0] == 0

    def test_get_conf_mat(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data)
        conf_mat = m.get_conf_mat()
        assert conf_mat.shape[0] == 6
        assert conf_mat.shape[1] == 3
        for i in range(3):
            assert not isinstance(conf_mat[0, i], str)

    def test_gp_normative_model(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data)
        m.gp_normative_model()
        assert 'GP_nmodel_pred' in m.data.columns
        assert math.isclose(0, m.data['GP_nmodel_residuals'].mean(), abs_tol=0.5)

    @pytest.fixture(scope='function')
    def test_plot(self):
        data = generate_data(randseed=3)
        m = pynm.PyNM(data)
        m.gp_normative_model()
        assert m.plot() is None
