# coding: utf-8

from __future__ import division

import time

import numpy as np
import pandas as pd
import xgboost as xgb

train = pd.read_csv('../data/input/train.csv')
entbase = pd.read_csv('../data/input/1entbase.csv')
alter = pd.read_csv('../data/input/2alter.csv')
branch = pd.read_csv('../data/input/3branch.csv')
invest = pd.read_csv('../data/input/4invest.csv')
right = pd.read_csv('../data/input/5right.csv')
project = pd.read_csv('../data/input/6project.csv')
lawsuit = pd.read_csv('../data/input/7lawsuit.csv')
breakfaith = pd.read_csv('../data/input/8breakfaith.csv')
recruit = pd.read_csv('../data/input/9recruit.csv')
qualification = pd.read_csv('../data/input/10qualification.csv', encoding='gbk')
test = pd.read_csv('../data/input/evaluation_public.csv')


def translate_date(date):
    year = int(date[:4])
    month = int(date[-2:])
    return (year - 2010) * 12 + month


def get_interaction_feature(df, feature_A, feature_B):
    feature_A_list = sorted(df[feature_A].unique())
    feature_B_list = sorted(df[feature_B].unique())
    count = 0
    mydict = {}
    for i in feature_A_list:
        mydict[int(i)] = {}
        for j in feature_B_list:
            mydict[int(i)][int(j)] = count
            count += 1
    return df.apply(lambda x: mydict[int(x[feature_A])][int(x[feature_B])], axis=1)


def get_entbase_feature(df):
    df = df.copy()

    mydf = df.fillna(value={'HY': 0, 'ZCZB': 0, 'MPNUM': 0, 'INUM': 0, 'ENUM': 0, 'FINZB': 0, 'FSTINUM': 0,
                            'TZINUM': 0})  # fill HY 为 0；ZCZB 为 0 表示缺失或错误

    mydf['allnum'] = mydf['MPNUM'] + mydf['INUM'] + mydf['MPNUM'] + mydf['TZINUM'] + mydf['ENUM']

    mydf['zczb/rgyear'] = mydf['ZCZB'] / mydf['RGYEAR']
    mydf["rgyear_zczb"] = get_interaction_feature(mydf, "RGYEAR", "ZCZB")
    mydf['rgyear_finzb'] = get_interaction_feature(mydf, 'RGYEAR', 'FINZB')
    mydf['rgyear_inum'] = get_interaction_feature(mydf, 'RGYEAR', 'INUM')
    mydf['rgyear_enum'] = get_interaction_feature(mydf, 'RGYEAR', 'ENUM')
    mydf['rgyear_fstinum'] = get_interaction_feature(mydf, 'RGYEAR', 'FSTINUM')
    mydf['rgyear_tzinum'] = get_interaction_feature(mydf, 'RGYEAR', 'TZINUM')
    mydf['rgyear_mpnum'] = get_interaction_feature(mydf, 'RGYEAR', 'MPNUM')
    mydf['zczb_rgyear'] = get_interaction_feature(mydf, 'ZCZB', 'RGYEAR')
    mydf['zczb_finzb'] = get_interaction_feature(mydf, 'ZCZB', 'FINZB')
    mydf['zczb_inum'] = get_interaction_feature(mydf, 'ZCZB', 'INUM')
    mydf['zczb_enum'] = get_interaction_feature(mydf, 'ZCZB', 'ENUM')
    mydf['zczb_fstinum'] = get_interaction_feature(mydf, 'ZCZB', 'FSTINUM')
    mydf['zczb_tzinum'] = get_interaction_feature(mydf, 'ZCZB', 'TZINUM')
    mydf['zczb_mpnum'] = get_interaction_feature(mydf, 'ZCZB', 'MPNUM')

    return mydf


def get_alter_feature(df):
    df = df.copy()

    alt_no = df.groupby(['EID', 'ALTERNO']).size().reset_index()
    alt_no = alt_no.groupby('EID')[0].agg([sum, len]).reset_index()
    alt_no.columns = ['EID', 'alt_count', 'alt_types_count']

    alt_no_oh = df.groupby(['EID', 'ALTERNO']).size().unstack().reset_index()
    alt_no_oh.columns = [i if i == 'EID' else 'alt_' + i for i in alt_no_oh.columns]

    df['date'] = df['ALTDATE'].apply(translate_date)
    date = df.groupby('EID')['date'].agg([min, max, np.ptp, np.std]).reset_index()
    date.columns = ['EID', 'alt_date_min', 'alt_date_max', 'alt_date_ptp', 'alt_date_std']

    df['altbe'] = df['ALTBE'].str.extract('(\d+\.?\d*)').astype(float)
    df['altaf'] = df['ALTAF'].str.extract('(\d+\.?\d*)').astype(float)
    alt_be_af = df.groupby('EID')['altbe', 'altaf'].agg([min, max, np.mean]).reset_index()
    alt_be_af.columns = ['EID', 'alt_be_min', 'alt_be_max', 'alt_be_mean', 'alt_af_min', 'alt_af_max', 'alt_af_mean']

    df['alt_be_af_gap'] = df['altaf'] - df['altbe']
    alt_be_af_gap = df.groupby('EID')['alt_be_af_gap'].agg([min, max, np.mean, np.ptp, np.std]).reset_index()
    alt_be_af_gap.columns = [i if i == 'EID' else 'alt_be_af_gap_' + i for i in alt_be_af_gap.columns]

    alt_1year = df[df['ALTDATE'] >= '2015-01'].groupby('EID').size().reset_index()
    alt_1year.columns = ['EID', 'alt_num(1year)']
    alt_2year = df[df['ALTDATE'] >= '2014-01'].groupby('EID').size().reset_index()
    alt_2year.columns = ['EID', 'alt_num(2year)']
    alt_3year = df[df['ALTDATE'] >= '2013-01'].groupby('EID').size().reset_index()
    alt_3year.columns = ['EID', 'alt_num(3year)']

    mydf = pd.merge(alt_no, alt_no_oh, how='left', on='EID')
    mydf = pd.merge(mydf, date, how='left', on='EID')
    mydf = pd.merge(mydf, alt_be_af, how='left', on='EID')
    mydf = pd.merge(mydf, alt_be_af_gap, how='left', on='EID')
    mydf = pd.merge(mydf, alt_1year, how='left', on='EID')
    mydf = pd.merge(mydf, alt_2year, how='left', on='EID')
    mydf = pd.merge(mydf, alt_3year, how='left', on='EID')

    return mydf


def get_right_feature(df):
    df = df.copy()

    rig_type = df.groupby(['EID', 'RIGHTTYPE']).size().reset_index()
    rig_type = rig_type.groupby('EID')[0].agg([sum, len]).reset_index()
    rig_type.columns = ['EID', 'rig_count', 'rig_types_count']

    rig_type_oh_rate = df.groupby(['EID', 'RIGHTTYPE']).size().unstack().reset_index()
    rig_type_oh_rate.iloc[:, 1:] = rig_type_oh_rate.iloc[:, 1:].div(rig_type['rig_count'], axis='index')
    rig_type_oh_rate.columns = [i if i == 'EID' else 'rig_rate_' + str(i) for i in rig_type_oh_rate.columns]

    df['ask_month'] = (
        pd.to_datetime(df['ASKDATE']).dt.to_period("M") - (pd.to_datetime('2010-01-01').to_period("M"))).fillna(
        -999).astype(int).replace(-999, np.NaN)
    ask_date = df.groupby('EID')['ask_month'].agg([max, min, np.ptp, np.std]).reset_index()
    ask_date.columns = ['EID', 'rig_askdate_max', 'rig_askdate_min', 'rig_askdate_ptp', 'rig_askdate_std']

    df['get_month'] = (
        pd.to_datetime(df['FBDATE']).dt.to_period("M") - (pd.to_datetime('2010-01-01').to_period("M"))).fillna(
        -999).astype(
        int).replace(-999, np.NaN)
    get_date = df.groupby('EID')['get_month'].agg([max, min, np.ptp, np.std]).reset_index()
    get_date.columns = ['EID', 'rig_getdate_max', 'rig_getdate_min', 'rig_getdate_ptp', 'rig_getdate_std']

    unget = df[df.FBDATE.isnull()]
    unget = unget.groupby('EID').size().reset_index()
    unget.columns = ['EID', 'rig_unget_num']

    right_1year = df[df['ASKDATE'] >= '2015-01'].groupby('EID')['ASKDATE'].count().reset_index()
    right_1year.columns = ['EID', 'ask_num(1year)']
    right_2year = df[df['ASKDATE'] >= '2014-01'].groupby('EID')['ASKDATE'].count().reset_index()
    right_2year.columns = ['EID', 'ask_num(2year)']
    right_5year = df[df['ASKDATE'] >= '2010-01'].groupby('EID')['ASKDATE'].count().reset_index()
    right_5year.columns = ['EID', 'ask_num(5year)']
    right_end_1year = df[df['FBDATE'] >= '2015-01'].groupby('EID')['FBDATE'].count().reset_index()
    right_end_1year.columns = ['EID', 'get_num(1year)']
    right_end_2year = df[df['FBDATE'] >= '2014-01'].groupby('EID')['FBDATE'].count().reset_index()
    right_end_2year.columns = ['EID', 'get_num(2year)']
    right_end_5year = df[df['FBDATE'] >= '2010-01'].groupby('EID')['FBDATE'].count().reset_index()
    right_end_5year.columns = ['EID', 'get_num(5year)']

    df['ask_get_month_gap'] = df['get_month'] - df['ask_month']
    ask_get_month_gap = df.groupby('EID')['ask_get_month_gap'].agg([max, min, np.ptp, np.mean, np.std]).reset_index()
    ask_get_month_gap.columns = [i if i == 'EID' else 'ask_get_month_gap_' + i for i in ask_get_month_gap.columns]

    mydf = pd.merge(rig_type, rig_type_oh_rate, how='left', on='EID')
    mydf = pd.merge(mydf, ask_date, how='left', on='EID')
    mydf = pd.merge(mydf, get_date, how='left', on='EID')
    mydf = pd.merge(mydf, unget, how='left', on='EID')
    mydf = pd.merge(mydf, right_1year, how='left', on='EID')
    mydf = pd.merge(mydf, right_2year, how='left', on='EID')
    mydf = pd.merge(mydf, right_5year, how='left', on='EID')
    mydf = pd.merge(mydf, right_end_1year, how='left', on='EID')
    mydf = pd.merge(mydf, right_end_2year, how='left', on='EID')
    mydf = pd.merge(mydf, right_end_5year, how='left', on='EID')
    mydf = pd.merge(mydf, ask_get_month_gap, how='left', on='EID')

    mydf['ask_rate(1year)'] = mydf['ask_num(1year)'] / mydf['rig_count']
    mydf['ask_rate(2year)'] = mydf['ask_num(2year)'] / mydf['rig_count']
    mydf['ask_rate(5year)'] = mydf['ask_num(5year)'] / mydf['rig_count']
    mydf['get_rate(1year)'] = mydf['get_num(1year)'] / mydf['rig_count']
    mydf['get_rate(2year)'] = mydf['get_num(2year)'] / mydf['rig_count']
    mydf['get_rate(5year)'] = mydf['get_num(5year)'] / mydf['rig_count']

    return mydf


def get_recruit_feature(df):
    df = df.copy()

    rec_wz = df.groupby(['EID', 'WZCODE']).size().reset_index()
    rec_wz = rec_wz.groupby('EID')[0].agg([sum, len]).reset_index()
    rec_wz.columns = ['EID', 'rec_wz_count', 'rec_wz_types_count']

    rec_wz_oh = df.groupby(['EID', 'WZCODE']).size().unstack().reset_index()
    rec_wz_oh.columns = [i if i == 'EID' else 'rec_wz_' + i for i in rec_wz_oh.columns]

    rec_pos = df.groupby(['EID', 'POSCODE']).size().reset_index()
    rec_pos = rec_pos.groupby('EID')[0].agg([sum, len]).reset_index()
    rec_pos.columns = ['EID', 'rec_pos_count', 'rec_pos_types_count']

    df['recdate'] = (
        pd.to_datetime(df['RECDATE']).dt.to_period("M") - (pd.to_datetime('2010-01-01').to_period("M"))).fillna(
        -999).astype(int).replace(-999, np.NaN)
    rec_date = df.groupby('EID')['recdate'].agg([max, min, np.ptp, np.std]).reset_index()
    rec_date.columns = ['EID', 'rec_date_max', 'rec_date_min', 'rec_date_ptp', 'rec_date_std']

    df['pnum'] = df['PNUM'].str.extract('(\d+)').fillna(1).astype(int)  # 若干=1
    rec_num = df.groupby('EID')['pnum'].agg([sum, max, min, np.ptp, np.std]).reset_index()
    rec_num.columns = ['EID' if i == 'EID' else 'rec_num_' + i for i in rec_num.columns]

    rec_count = df.groupby('EID').size().reset_index()
    rec_count.columns = ['EID', 'rec_count']

    mydf = pd.merge(rec_wz, rec_wz_oh, how='left', on='EID')
    mydf = pd.merge(mydf, rec_pos, how='left', on='EID')
    mydf = pd.merge(mydf, rec_date, how='left', on='EID')
    mydf = pd.merge(mydf, rec_num, how='left', on='EID')
    mydf = pd.merge(mydf, rec_count, how='left', on='EID')

    return mydf


def get_branch_feature(df):
    df = df.copy()

    bra_num = df.groupby('EID')['TYPECODE'].size().reset_index()
    bra_num.columns = ['EID', 'bra_count']

    bra_home = df.groupby(['EID', 'IFHOME']).size().unstack().reset_index()
    bra_home.columns = ['EID', 'bra_nothome', 'bra_home']

    bra_year = df.groupby('EID')['B_REYEAR'].agg([min, max, np.ptp, np.std]).reset_index()
    bra_year.columns = [i if i == 'EID' else 'bra_year_' + i for i in bra_year.columns]

    bra_endyear = df.groupby('EID')['B_ENDYEAR'].agg([min, max, np.ptp, np.std]).reset_index()
    bra_endyear.columns = [i if i == 'EID' else 'bra_endyear_' + i for i in bra_endyear.columns]

    bra_end_num = df[~df['B_ENDYEAR'].isnull()].groupby('EID').size().reset_index()
    bra_end_num.columns = ['EID', 'bra_end_num']
    bra_notend_num = df[df['B_ENDYEAR'].isnull()].groupby('EID').size().reset_index()
    bra_notend_num.columns = ['EID', 'bra_notend_num']

    df['bra_begin_end_gap'] = df['B_ENDYEAR'] - df['B_REYEAR']
    bra_begin_end_gap = df.groupby('EID')['bra_begin_end_gap'].agg([min, max, np.ptp, np.mean, np.std]).reset_index()
    bra_begin_end_gap.columns = [i if i == 'EID' else 'bra_begin_end_gap_' + i for i in bra_begin_end_gap.columns]

    mydf = pd.merge(bra_num, bra_home, how='left', on='EID')
    mydf = pd.merge(mydf, bra_year, how='left', on='EID')
    mydf = pd.merge(mydf, bra_endyear, how='left', on='EID')
    mydf = pd.merge(mydf, bra_end_num, how='left', on='EID')
    mydf = pd.merge(mydf, bra_notend_num, how='left', on='EID')
    mydf = pd.merge(mydf, bra_begin_end_gap, how='left', on='EID')

    return mydf


def get_invest_feature(df):
    df = df.copy()

    inv_num = df.groupby('EID').size().reset_index()
    inv_num.columns = ['EID', 'inv_count']

    inv_home = df.groupby(['EID', 'IFHOME']).size().unstack().reset_index()
    inv_home.columns = ['EID', 'inv_nothome_num', 'inv_home_num']

    inv_bl = df.groupby('EID')['BTBL'].agg([sum, min, max, np.ptp, np.std]).reset_index()
    inv_bl.columns = [i if i == 'EID' else 'inv_bl_' + i for i in inv_bl.columns]

    inv_year = df.groupby('EID')['BTYEAR'].agg([min, max, np.ptp, np.std]).reset_index()
    inv_year.columns = [i if i == 'EID' else 'inv_year_' + i for i in inv_year.columns]

    inv_endyear = df.groupby('EID')['BTENDYEAR'].agg([min, max, np.ptp, np.std]).reset_index()
    inv_endyear.columns = [i if i == 'EID' else 'inv_endyear_' + i for i in inv_endyear.columns]

    inved_num = df.groupby('BTEID').size().reset_index()
    inved_num.columns = ['EID', 'inved_num']

    inved_home = df.groupby(['BTEID', 'IFHOME']).size().unstack().reset_index()
    inved_home.columns = ['EID', 'inved_nothome_num', 'inved_home_num']

    inved_bl = df.groupby('BTEID')['BTBL'].agg([sum, min, max, np.ptp, np.std]).reset_index()
    inved_bl.columns = ['EID' if i == 'BTEID' else 'inved_bl_' + i for i in inved_bl.columns]

    inved_year = df.groupby('BTEID')['BTYEAR'].agg([min, max, np.ptp, np.std]).reset_index()
    inved_year.columns = ['EID' if i == 'BTEID' else 'inved_year_' + i for i in inved_year.columns]

    inved_endyear = df.groupby('BTEID')['BTENDYEAR'].agg([min, max, np.ptp, np.std]).reset_index()
    inved_endyear.columns = ['EID' if i == 'BTEID' else 'inved_endyear_' + i for i in inved_endyear.columns]

    df['inv_begin_end_gap'] = df['BTENDYEAR'] - df['BTYEAR']
    inv_begin_end_gap = df.groupby('EID')['inv_begin_end_gap'].agg([min, max, np.ptp, np.mean, np.std]).reset_index()
    inv_begin_end_gap.columns = [i if i == 'EID' else 'inv_begin_end_gap_' + i for i in inv_begin_end_gap.columns]

    mydf = pd.merge(inv_num, inv_home, how='left', on='EID')
    mydf = pd.merge(mydf, inv_bl, how='left', on='EID')
    mydf = pd.merge(mydf, inv_year, how='left', on='EID')
    mydf = pd.merge(mydf, inv_endyear, how='left', on='EID')
    mydf = pd.merge(mydf, inved_num, how='left', on='EID')
    mydf = pd.merge(mydf, inved_home, how='left', on='EID')
    mydf = pd.merge(mydf, inved_bl, how='left', on='EID')
    mydf = pd.merge(mydf, inved_year, how='left', on='EID')
    mydf = pd.merge(mydf, inved_endyear, how='left', on='EID')
    mydf = pd.merge(mydf, inv_begin_end_gap, how='left', on='EID')

    return mydf


def get_lawsuit_feature(df):
    df = df.copy()

    law_num = df.groupby('EID').size().reset_index()
    law_num.columns = ['EID', 'law_count']

    df['lawdate'] = df['LAWDATE'].apply(lambda x: x.replace('年', '-').replace('月', '')).apply(translate_date)
    law_date = df.groupby('EID')['lawdate'].agg([min, max, np.ptp, np.std]).reset_index()
    law_date.columns = [i if i == 'EID' else 'law_date_' + i for i in law_date.columns]

    law_amout = df.groupby('EID')['LAWAMOUNT'].agg([sum, min, max, np.mean, np.ptp, np.std]).reset_index()
    law_amout.columns = [i if i == 'EID' else 'law_amout_' + i for i in law_amout.columns]

    mydf = pd.merge(law_num, law_date, how='left', on='EID')
    mydf = pd.merge(mydf, law_amout, how='left', on='EID')

    return mydf


def get_project_feature(df):
    df = df.copy()

    pro_num = df.groupby('EID').size().reset_index()
    pro_num.columns = ['EID', 'pro_count']

    df['djdate'] = df['DJDATE'].apply(translate_date)
    pro_date = df.groupby('EID')['djdate'].agg([min, max, np.ptp, np.std]).reset_index()
    pro_date.columns = [i if i == 'EID' else 'pro_date_' + i for i in pro_date.columns]

    pro_home = df.groupby(['EID', 'IFHOME']).size().unstack().reset_index()
    pro_home.columns = ['EID', 'pro_nothome_num', 'pro_home_num']

    mydf = pd.merge(pro_num, pro_date, how='left', on='EID')
    mydf = pd.merge(mydf, pro_home, how='left', on='EID')

    return mydf


def get_qualification_feature(df):
    df = df.copy()

    qua_num = df.groupby('EID').size().reset_index()
    qua_num.columns = ['EID', 'qua_count']

    qua_type = df.groupby(['EID', 'ADDTYPE']).size().unstack().reset_index()
    qua_type.columns = [i if i == 'EID' else 'qua_type_' + str(i) for i in qua_type.columns]

    df['begindate'] = df['BEGINDATE'].apply(lambda x: x.replace(u'年', '-').replace(u'月', '')).apply(translate_date)
    qua_begindate = df.groupby('EID')['begindate'].agg([min, max, np.ptp, np.std]).reset_index()
    qua_begindate.columns = [i if i == 'EID' else 'qua_begindate_' + i for i in qua_begindate.columns]

    df['expirydate'] = df['EXPIRYDATE'].apply(
        lambda x: x.replace(u'年', '-').replace(u'月', '') if not pd.isnull(x) else np.nan)
    df['expirydate'] = (
        pd.to_datetime(df['expirydate']).dt.to_period("M") - (pd.to_datetime('2010-01-01').to_period("M"))).fillna(
        -999).astype(int).replace(-999, np.NaN)
    qua_expirydate = df.groupby('EID')['expirydate'].agg([min, max, np.ptp, np.std]).reset_index()
    qua_expirydate.columns = [i if i == 'EID' else 'qua_expirydate_' + i for i in qua_expirydate.columns]

    df['qua_begin_expiry_gap'] = df['expirydate'] - df['begindate']
    qua_begin_expiry_gap = df.groupby('EID')['qua_begin_expiry_gap'].agg(
        [min, max, np.ptp, np.mean, np.std]).reset_index()
    qua_begin_expiry_gap.columns = [i if i == 'EID' else 'qua_begin_expiry_gap_' + i for i in
                                    qua_begin_expiry_gap.columns]

    mydf = pd.merge(qua_num, qua_type, how='left', on='EID')
    mydf = pd.merge(mydf, qua_begindate, how='left', on='EID')
    mydf = pd.merge(mydf, qua_expirydate, how='left', on='EID')
    mydf = pd.merge(mydf, qua_begin_expiry_gap, how='left', on='EID')

    return mydf


def get_breakfaith_feature(df):
    df = df.copy()

    bre_num = df.groupby('EID').size().reset_index()
    bre_num.columns = ['EID', 'bre_count']

    df['fbdate'] = df['FBDATE'].apply(lambda x: x.replace('年', '-').replace('月', '')).apply(translate_date)
    bre_date = df.groupby('EID')['fbdate'].agg([min, max, np.ptp, np.std]).reset_index()
    bre_date.columns = [i if i == 'EID' else 'bre_date_' + i for i in bre_date.columns]

    df['sxenddate'] = (
        pd.to_datetime(df['SXENDDATE']).dt.to_period("M") - (pd.to_datetime('2010-01-01').to_period("M"))).fillna(
        -999).astype(int).replace(-999, np.NaN)
    bre_enddate = df.groupby('EID')['sxenddate'].agg([min, max, np.ptp, np.std]).reset_index()
    bre_enddate.columns = [i if i == 'EID' else 'bre_enddate_' + i for i in bre_enddate.columns]

    df['bre_begin_end_gap'] = df['sxenddate'] - df['fbdate']
    bre_begin_end_gap = df.groupby('EID')['bre_begin_end_gap'].agg([min, max, np.ptp, np.mean, np.std]).reset_index()
    bre_begin_end_gap.columns = [i if i == 'EID' else 'bre_begin_end_gap_' + i for i in bre_begin_end_gap.columns]

    mydf = pd.merge(bre_num, bre_date, how='left', on='EID')
    mydf = pd.merge(mydf, bre_enddate, how='left', on='EID')
    mydf = pd.merge(mydf, bre_begin_end_gap, how='left', on='EID')

    return mydf


entbase_feat = get_entbase_feature(entbase)
alter_feat = get_alter_feature(alter)
right_feature = get_right_feature(right)
recruit_feat = get_recruit_feature(recruit)
branch_feat = get_branch_feature(branch)
invest_feat = get_invest_feature(invest)
lawsuit_feat = get_lawsuit_feature(lawsuit)
project_feat = get_project_feature(project)
qualification_feat = get_qualification_feature(qualification)
breakfaith_feat = get_breakfaith_feature(breakfaith)

dataset = pd.merge(entbase_feat, alter_feat, on='EID', how='left')
dataset = pd.merge(dataset, right_feature, on='EID', how='left')
dataset = pd.merge(dataset, recruit_feat, on='EID', how='left')
dataset = pd.merge(dataset, branch_feat, on='EID', how='left')
dataset = pd.merge(dataset, invest_feat, on='EID', how='left')
dataset = pd.merge(dataset, lawsuit_feat, on='EID', how='left')
dataset = pd.merge(dataset, project_feat, on='EID', how='left')
dataset = pd.merge(dataset, qualification_feat, on='EID', how='left')
dataset = pd.merge(dataset, breakfaith_feat, on='EID', how='left')

dataset['alt_count/rgyear'] = dataset['alt_count'] / dataset['RGYEAR']
dataset['rig_count/rgyear'] = dataset['rig_count'] / dataset['RGYEAR']
dataset['rec_count/rgyear'] = dataset['rec_count'] / dataset['RGYEAR']
dataset['bra_count/rgyear'] = dataset['bra_count'] / dataset['RGYEAR']
dataset['inv_count/rgyear'] = dataset['inv_count'] / dataset['RGYEAR']
dataset['inved_num/rgyear'] = dataset['inved_num'] / dataset['RGYEAR']
dataset['law_count/rgyear'] = dataset['law_count'] / dataset['RGYEAR']
dataset['pro_count/rgyear'] = dataset['pro_count'] / dataset['RGYEAR']
dataset['qua_count/rgyear'] = dataset['qua_count'] / dataset['RGYEAR']
dataset['bre_count/rgyear'] = dataset['bre_count'] / dataset['RGYEAR']

dataset['alt_num(1year)/rgyear'] = dataset['alt_num(1year)'] / dataset['RGYEAR']
dataset['alt_num(2year)/rgyear'] = dataset['alt_num(2year)'] / dataset['RGYEAR']
dataset['alt_num(3year)/rgyear'] = dataset['alt_num(3year)'] / dataset['RGYEAR']

dataset['MPNUM_CLASS'] = dataset['INUM'].apply(lambda x: x if x <= 4 else 5)
dataset['FSTINUM_CLASS'] = dataset['FSTINUM'].apply(lambda x: x if x <= 6 else 7)
dataset.fillna(value={'alt_count': 0, 'rig_count': 0}, inplace=True)
for column in ['MPNUM', 'INUM', 'FINZB', 'FSTINUM', 'TZINUM', 'ENUM', 'ZCZB', 'allnum', 'RGYEAR', 'alt_count', 'rig_count']:
    groupby_list = [['HY'], ['ETYPE'], ['HY', 'ETYPE'], ['HY', 'PROV'], ['ETYPE', 'PROV'], ['MPNUM_CLASS'], ['FSTINUM_CLASS']]
    for groupby in groupby_list:
        if 'MPNUM_CLASS' in groupby and column == 'MPNUM':
            continue
        if 'FSTINUM_CLASS' in groupby and column == 'FSTINUM':
            continue
        groupby_keylist = []
        for key in groupby:
            groupby_keylist.append(dataset[key])
        tmp = dataset[column].groupby(groupby_keylist).agg([sum, min, max, np.mean]).reset_index()
        tmp = pd.merge(dataset, tmp, on=groupby, how='left')
        dataset['ent_' + column.lower() + '-mean_gb_' + '_'.join(groupby).lower()] = dataset[column] - tmp['mean']
        dataset['ent_' + column.lower() + '-min_gb_' + '_'.join(groupby).lower()] = dataset[column] - tmp['min']
        dataset['ent_' + column.lower() + '-max_gb_' + '_'.join(groupby).lower()] = dataset[column] - tmp['max']
        dataset['ent_' + column.lower() + '/sum_gb_' + '_'.join(groupby).lower()] = dataset[column] / tmp['sum']
dataset.drop(['MPNUM_CLASS', 'FSTINUM_CLASS'], axis=1, inplace=True)

dataset = dataset.join(pd.get_dummies(dataset['PROV'], prefix='prov'))
dataset = dataset.join(pd.get_dummies(dataset['HY'], prefix='hy').mul(dataset['ZCZB'], axis=0))
dataset = dataset.join(pd.get_dummies(dataset['ETYPE'], prefix='etype').mul(dataset['RGYEAR'], axis=0))
dataset.drop(['PROV', 'HY', 'ETYPE'], axis=1, inplace=True)

trainset = pd.merge(train, dataset, on='EID', how='left')
testset = pd.merge(test, dataset, on='EID', how='left')

train_feature = trainset.drop(['TARGET', 'ENDDATE'], axis=1)
train_label = trainset.TARGET.values
test_feature = testset
test_index = testset.EID.values
print train_feature.shape, train_label.shape, test_feature.shape

# EID 前面的字母代表不同省份，已提供了 PROV 列，因此字母是冗余信息，直接舍弃
train_feature['EID'] = train_feature['EID'].str.extract('(\d+)').astype(int)
test_feature['EID'] = test_feature['EID'].str.extract('(\d+)').astype(int)

config = {
    'rounds': 10000,
    'folds': 5
}

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'stratified': True,
    'scale_pos_weights ': 0,
    'max_depth': 10,
    'min_child_weight': 15,
    #     'gamma': 0.1,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    #     'lambda': 1,

    'eta': 0.01,
    'seed': 42,
    'silent': 1,
    'eval_metric': 'auc'
}


def xgb_cv(train_feature, train_label, params, folds, rounds):
    start = time.clock()
    print train_feature.columns
    params['scale_pos_weights '] = float(len(train_label[train_label == 0])) / len(train_label[train_label == 1])
    dtrain = xgb.DMatrix(train_feature, label=train_label)
    num_round = rounds
    print 'run cv: ' + 'round: ' + str(rounds)
    res = xgb.cv(params, dtrain, num_round, nfold=folds, verbose_eval=10, early_stopping_rounds=100)
    elapsed = (time.clock() - start)
    print 'Time used:', elapsed, 's'
    return len(res), res.loc[len(res) - 1, 'test-auc-mean']


def xgb_predict(train_feature, train_label, test_feature, rounds, params):
    params['scale_pos_weights '] = float(len(train_label[train_label == 0])) / len(train_label[train_label == 1])
    dtrain = xgb.DMatrix(train_feature, label=train_label)
    dtest = xgb.DMatrix(test_feature, label=np.zeros(test_feature.shape[0]))
    watchlist = [(dtrain, 'train')]
    num_round = rounds
    model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=30)
    predict = model.predict(dtest)
    return model, predict


def store_result(test_index, pred, threshold, name):
    result = pd.DataFrame({'EID': test_index, 'FORTARGET': 0, 'PROB': pred})
    mask = result['PROB'] >= threshold
    result.at[mask, 'FORTARGET'] = 1
    result.to_csv('../data/output/sub/' + name + '.csv', index=0)
    return result


iterations, best_score = xgb_cv(train_feature, train_label, params, config['folds'], config['rounds'])

model, pred = xgb_predict(train_feature, train_label, test_feature, iterations, params)

importance = pd.DataFrame(model.get_fscore().items(), columns=['feature', 'importance']).sort_values('importance', ascending=False)
importance.to_csv('../data/output/feat_imp/importance-1207-%f(r%d).csv' % (best_score, iterations), index=False)

res = store_result(test_index, pred, 0.18, '1207-xgb-%f(r%d)' % (best_score, iterations))
