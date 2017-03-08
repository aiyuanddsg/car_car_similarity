
# coding=utf-8
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics

def gen_rule(expression):
    feature, condition = expression.split()[:2]
    condition = condition.replace('==NA','.isnull()').replace('!=NA','.notnull()')
    return lambda x: x[0:0] if feature not in x else x[eval("x[u'%s']%s" % (feature, condition))]

def desc_rule_hits(hits, good_label=1, bad_label=0):
    good = hits[hits.LABEL==good_label].shape[0]
    bad = hits[hits.LABEL==bad_label].shape[0]
    prec = np.round(float(bad) / (bad + good), 2) if (bad+good>0) else np.nan
    return {'good':good, 'bad':bad, 'prec':prec}

def fire_rules(data, rules, return_hits=False, label_col='LABEL', id_col='_apply_id'):
    hits, acc_hits = [], []
    for k,r in rules.iteritems():
        h = r(data)[[label_col, id_col]]
        acc_hits.append(h.copy() if not acc_hits else acc_hits[-1].append(h).drop_duplicates())
        h['r_' + k] = 1
        hits.append(h)
    res = pd.DataFrame(map(desc_rule_hits, hits))
    acc_res = pd.DataFrame(map(desc_rule_hits, acc_hits))
    ret = pd.merge(res, acc_res, left_index=True, right_index=True, suffixes=['', '_a'])
    ret['pbad'] = (ret['bad_a'] / (data[label_col]==0).sum()).round(2)
    ret['pgood'] = (ret['good_a'] / (data[label_col]==1).sum()).round(2)
    ret['rule'] = rules.keys()
    return (ret, pd.concat(hits).fillna(0).groupby(id_col).max()) if return_hits else ret

def find_rules(data, cols, label_col='LABEL', prop_smoother=0.01, min_abs_woe=0.1, filter=True):
    if not isinstance(cols, list): cols = [cols]
    rules = []
    for col in cols:
        if col==label_col: continue
        pt = gen_pivot_table(data[col], data[label_col], False)
        if pt is None: print col; continue
        if 0 not in pt: pt[0] = 0
        if 1 not in pt: pt[1] = 0
        # is_numeric = sum(1 for i in pt.index if isinstance(i, float) or isinstance(i, int)) > 0
        r = discover_rules_from_feature(pt, expand=True, is_numeric=data[col].dtype.name!='object',
            min_hit=10, prop_smoother=prop_smoother, min_abs_woe=min_abs_woe)
        if r.size > 0:
            r['key'] = col
            rules.append(r[r.columns.tolist()[:-2] + ['key', 'val']])
    rules = pd.concat(rules)
    if filter:
        rules.sort_values('woe', inplace=True)
        ret = []
        ret.append(rules[rules.woe < 0].groupby('key').head(1)) # min of negative woe
        ret.append(rules[rules.woe > 0].groupby('key').tail(1)) # max of positive woe
        ret.append(rules.sort_values('iv').groupby('key').tail(1)) # max of IV
        rules = pd.concat(ret).drop_duplicates()
    return rules.round(3)

# pt = gen_pivot_table(d.trans_first_date_daysago, d['LABEL'], False)

def discover_rules_from_feature(pt, expand=False, is_numeric=False, min_hit=1, prop_smoother=0.01, min_abs_woe=0, na_name='NA', verbose=True):
    pt = pt[[1,0]].rename(columns = {1:'pos', 0:'neg'}).sort_index()
    tpos, tneg = pt.sum()['pos'], pt.sum()['neg']
    if expand and pt.index.size > 0:
        rs = pt.rename(lambda x: "==" + (x.encode('utf-8') if isinstance(x, unicode) else str(x)))
        if na_name in pt.index:
            pt.drop(na_name, inplace=True)
        neq = pd.Series(pt.sum(), name=['!=' + na_name])  # neq = (totals - pt).rename(lambda x: "!=" + str(x))
        if is_numeric:
            top = 10 # TEMP
            lt = pt.cumsum().iloc[:-1].iloc[1:top].rename(lambda x: "<=" + (x.encode('utf-8') if isinstance(x, unicode) else str(x))) # ignore "<= min"
            gt = (pt.sum().iloc[:top] - pt.cumsum().iloc[:top]).rename(lambda x: ">" + (x.encode('utf-8') if isinstance(x, unicode) else str(x)))
            rs = rs.append(lt).append(gt)
        rs = rs.append(neq)
    else:
        rs = pt
    rs['ppos'] = rs.pos / tpos
    rs['pneg'] = rs.neg / tneg
    rs['prec'] = (rs.pos / (rs.pos + rs.neg))
    rs['woe'] = np.log((rs.ppos + prop_smoother) / (rs.pneg + prop_smoother))
    rs['iv'] = (rs.ppos - rs.pneg) * rs.woe
    rs = rs[abs(rs.woe) >= min_abs_woe]
    rs['val'] = rs.index.copy()
    rs.index = range(rs.index.size)
    P, N = rs.pos.sum(), rs.neg.sum()
    if (not expand) and verbose:
        print 'IV = %.3f' % rs['iv'].sum()
        print 'P:N = %d : %d (%.3f)' % (P, N, float(P)/max(1, P+N))
    rs.columns.name = None
    return rs[rs.pos + rs.neg >= min_hit]

def eval_features(data, label_col, numerical_cols, categorical_cols, fillna=-1, prop_smoother=0.01):
    labels = data[label_col].copy()
    evals = [eval_feature(labels, data[col], True, fillna) for col in numerical_cols if col != label_col and col in data]
    evals += [eval_feature(labels, data[col], False, fillna) for col in categorical_cols if col != label_col and col in data]
    ret = pd.DataFrame(evals) #, columns=['type','bins','hit','hpos','hneg','hit_p','hauc','auc','ks','ks_p','iv','name'])
    # ret['score'] = ret.iv + np.maximum(abs(ret['hauc'].fillna(0.5) - 0.5), abs(ret['auc'].fillna(0.5) - 0.5))
    ret['woe'] = np.log((ret.hpos+prop_smoother) / (ret.hneg+prop_smoother)).round(3)
    for col in ret.columns:
        if ret[col].dtype == np.float64:
            ret[col] = ret[col].round(3)
    dst_cols = [c for c in ['type','dcnt','bins','hit','hpos','hneg','hit_p','hauc','auc','ks','ks_p','iv','woe','name'] if c in ret.columns]
    return ret[dst_cols].sort_values('iv', ascending=False)

def va(data, col, bins=None, label_col='LABEL', dropna=False, min_hit=1, min_abs_woe=0, prop_smoother=0.01, verbose=True):
    assert isinstance(data, pd.DataFrame)
    values = col if isinstance(col, pd.Series) else data[col]
    bin_ranges = None
    if bins is not None:
        values, bin_ranges = bin_values(values, bins, show_bin_ranges=True)
    pt = gen_pivot_table(values, data[label_col], dropna)
    if 0 not in pt: pt[0] = 0
    if 1 not in pt: pt[1] = 0
    ret = discover_rules_from_feature(pt, min_hit=min_hit, min_abs_woe=min_abs_woe, prop_smoother=prop_smoother, verbose=verbose)
    ret['prop'] = ((ret.pos + ret.neg) / pt.sum().sum())
    if bin_ranges is not None:
        ret['val'] = ret.val.map(bin_ranges)
    return ret.sort_values('woe', ascending=False).round(3) if bins is None else ret.round(3)


def eval_clf_result(y_true, y_pred, tag=None, y_th=0, bins=10, denys=None, data=None, prop_smoother=0.01, verbose=0, weight=None, top_pcts=None):
    if denys is not None: y_pred *= 1-denys.astype(int)
    if isinstance(y_true, pd.Series): y_true = y_true.values
    if isinstance(y_pred, pd.Series): y_pred = y_pred.values
    num_pos = y_true.sum()
    num_neg = y_true.size - num_pos
    ks = eval_ks(y_pred[y_true > y_th], y_pred[y_true <= y_th])[0]
    auc = metrics.roc_auc_score(y_true, y_pred, sample_weight=weight)
    ap = metrics.average_precision_score(y_true, y_pred, sample_weight=weight)
    ret = pd.DataFrame([[num_pos, num_neg, ap, auc, ks]], columns=['pos','neg','ap','auc','ks'])
    # ret = {'ap':ap, 'auc':auc, 'ks':ks} #'ksp':ksp
    s = np.argsort(y_pred)[::-1]
    top_precs = None if top_pcts is None else pd.Series([y_true[s][:int(round(y_true.size*pct/100.0))].mean() for pct in top_pcts], index=top_pcts)
    tops = pd.DataFrame(y_true[s])
    tops['bin'] = tops.index.to_series() * bins // tops.index.size
    cnt = tops.groupby('bin').count().rename(columns={0:'cnt'})
    pos = tops.groupby('bin').sum().rename(columns={0:'pos'})
    zz = pd.concat([cnt, pos], axis=1)
    zz['neg'] = zz.cnt - zz.pos
    zz['ppos'] = (zz.pos / num_pos).round(3)
    zz['pneg'] = (zz.neg / num_neg).round(3)
    zz['woe'] = np.log((zz.pos/num_pos+prop_smoother) / (zz.neg/num_neg+prop_smoother)).round(3)
    zz['pre'] = (zz.pos / zz.cnt).round(3)
    zz['cum'] = (zz.pos.cumsum() /zz.cnt.cumsum()).rename(columns={0:'cum'}).round(3)
    zz.index.name = None
    ret['iv'] = float(((zz.ppos - zz.pneg) * zz.woe).sum())
    ret['ll'] = metrics.log_loss(y_true, y_pred, sample_weight=weight)
    ret['rmse'] = np.sqrt(metrics.mean_squared_error(y_true, y_pred, sample_weight=weight))
    ret['rmse+'] = np.sqrt(metrics.mean_squared_error(y_true, y_pred, y_true if weight is None else y_true * weight)) # pos only
    ret['rmse-'] = np.sqrt(metrics.mean_squared_error(y_true, y_pred, 1-y_true if weight is None else (1-y_true) * weight)) # neg only
    ret['pred+'] = y_pred[y_true > 0].mean()
    ret['pred-'] = y_pred[y_true <= 0].mean()
    # for k in range(1):
    #     ret['p' + str(k+1)] = y_true[s[k*num_pos : (k+1)*num_pos]].mean() # true pos rate in top bins
    if tag: ret['tag'] = tag
    # top = data[y_pred >= np.percentile(y_pred, 100 - 100/bins)] if data is not None else None
    if verbose>0: print '%d vs %d' % (num_pos, num_neg); print 'ap\tauc\tks\tiv\n%.3f\t%.3f\t%.3f\t%.3f' % (ret['ap'],ret['auc'],ret['ks'],ret['iv'])
    return ret.round(3), zz, top_precs

def get_feature_importance(clf, feature_names, group=False):
    fi = pd.DataFrame(zip(feature_names, clf.feature_importances_), columns=['feature','importance'])
    fi['rank'] = fi.importance.rank('min', ascending=False).astype(int)
    if not group:
        return fi.sort_index(by='importance', ascending=False)
    ret = fi.groupby('key').agg({'importance':sum, 'feature':np.size}).sort_values('feature').rename(columns={'feature':'cnt'})
    ret['prefix'] = ret.index.to_series().apply(lambda x: x.split('_')[0])
    return ret


def gen_pivot_table(index, columns, dropna=False, na_name='NA'):
    assert isinstance(index, pd.Series)
    if index.count() == 0:
        return None
    pt = pd.crosstab(index, columns, dropna=False)
    if not dropna: # consider missing value explicitly
        na_indices = index.isnull()
        if na_indices.any():
            if pt.index.dtype.name == 'category':
                pt.index = pt.index.astype(str)
            pt.loc[na_name] = columns[na_indices].value_counts()
    return pt.fillna(0)

def bin_values(values, bins, show_bin_ranges=False):
    assert isinstance(values, pd.Series)
    if values.nunique() <= bins: # exclude NA
        return values, None
    try_bins = bins * 2
    pcts = values.rank(method='average', na_option='keep', pct=True)
    while True:
        bvalues = (pcts * try_bins).round()
        if bvalues.nunique() <= bins:
            bvalues = bvalues.rank(method='dense')
            if not show_bin_ranges:
                return bvalues, None
            t = pd.concat([values.to_frame(name='val'), bvalues.to_frame('bin')], axis=1).groupby('bin').val
            branges = '[' + t.min().astype(str) + ', ' + t.max().astype(str) + ']'
            return bvalues, branges
        try_bins -= 0.1

def eval_feature(labels, values, is_numerical_type, fillna, dropna=False, prop_smoother=0.01, iv_bins=5):
    assert isinstance(labels, pd.Series)
    assert isinstance(values, pd.Series)
    assert labels.shape == values.shape
    assert np.issubdtype(labels.dtype, np.int)
    assert not labels.isnull().any() # no missing label
    assert len(labels.unique()) == 2 # two-class

    ret = {'name': values.name}
    dcnt = values.nunique() # exclude NA
    ret['dcnt'] = dcnt
    ret['hit'] = float(values.count()) / values.size; # proportion of non-NaN values
    ret['hpos'] = values[labels>0].notnull().mean()
    ret['hneg'] = values[labels<=0].notnull().mean()

    ctab = None
    if is_numerical_type:
        ret['type'] = 'N'
        ret['hauc'] =  eval_auc_ks(labels.values, values.values, True)[0] # ignore NA
        ret['auc'], ret['ks'], ret['ks_p'] = eval_auc_ks(labels.values, values.fillna(fillna).values, True)
        binned_values = bin_values(values, iv_bins, show_bin_ranges=False)[0]
        ret['bins'] = binned_values.nunique()
        ctab = gen_pivot_table(binned_values, labels, dropna)
    else:
        ret['type'] = 'C'
        ret['bins'] = dcnt
        ctab = gen_pivot_table(values, labels, dropna)

    if ctab is not None and len(ctab.columns) == 2:
        ret['iv'] = eval_iv(ctab.iloc[:,0].values, ctab.iloc[:,1].values, prop_smoother)
        # ret['chi_p'] = eval_chi2(ctab.values)[1]

    na = values.isnull()
    if na.any() and na.sum() < values.size:
        ret['hit_p'] = eval_chi2(pd.crosstab(na, labels).values)[1]
        # ret['na_auc'] = eval_auc(labels.values, na.astype(int).values)

    return ret

def eval_auc_ks(labels, values, normalize_by_ks):
    assert isinstance(labels, np.ndarray)
    assert isinstance(values, np.ndarray)
    uniq_labels = np.unique(labels)
    assert len(uniq_labels) == 2

    auc = eval_auc(labels, values)
    indices = (labels == uniq_labels[0])
    ks, ks_p = eval_ks(values[indices], values[~indices])
    if normalize_by_ks and (auc is not None) and (ks_p is not None):
        auc = (auc - 0.5) * (1.0 - ks_p) + 0.5
    return (auc, ks, ks_p)

def eval_auc(labels, values):
    """Evaluate AUC score from ground-truth labels and predicted scores.

    Parameters
    ----------
    labels : np.ndarray
        ground-truth, with 2 unique values
    values : np.ndarray
        continuous predictions

    Returns
    -------
    auc : float
    """
    assert isinstance(labels, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert len(np.unique(labels)) == 2

    # NOTE: must exclude NaNs
    indices = ~np.isnan(values)
    li = labels[indices]
    return np.nan if len(np.unique(li)) != 2 else metrics.roc_auc_score(li, values[indices])

def eval_auc_top(labels, values, top_rate=1.0):
    assert 0 < top_rate <= 1.0
    assert isinstance(labels, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert len(np.unique(labels)) == 2
    indices = values >= np.percentile(values, (1.0-top_rate)*100.0)
    li = labels[indices]
    return np.nan if len(np.unique(li)) != 2 else metrics.roc_auc_score(li, values[indices])

def eval_ks(arr1, arr2):
    """Computes the Kolmogorov-Smirnov statistic on 2 samples.

    Parameters
    ----------
    arr1 : np.ndarray
    arr2 : np.ndarray

    Returns
    -------
    D : float
        KS statistic
    p-value : float
        two-tailed p-value
    """
    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)
    a1 = arr1[~np.isnan(arr1)]
    a2 = arr2[~np.isnan(arr2)]
    if a1.size > 0 and a2.size > 0:
        return stats.ks_2samp(a1, a2)
    return (np.nan, np.nan)

def eval_chi2(contingency_table):
    """Evaluate chi-square statistics.
    """
    try:
        ret = stats.chi2_contingency(contingency_table)
        return ret[:2]
    except:
        return (np.nan, np.nan)

def eval_iv(p0, q0, prop_smoother=0.01):
    assert isinstance(p0, np.ndarray)
    assert isinstance(q0, np.ndarray)
    p = p0 * (1.0 / p0.sum()) + prop_smoother
    q = q0 * (1.0 / q0.sum()) + prop_smoother
    return np.inner(p - q, np.log(p / q))

def eval_gini(values):
    assert isinstance(values, np.ndarray)
    assert values.size > 1
    return 1 - (np.cumsum(np.sort(values)).sum() / float(values.sum()) * 2 - 1) / values.size

def eval_dcg(rels):
    """Discounted cumulative gain (DCG)
    """
    gains = 2 ** rels.astype(float) - 1
    discounts = np.log2(np.arange(2, rels.size+2))
    return np.sum(gains / discounts)

def eval_ndcg(rels, k):
    """normalized discounted cumulative gain @ top K
    """
    if 0 < k <= 1: k = int(round(float(rels.size) * k))
    if k < 1: return np.nan
    ideal_dcg = eval_dcg(np.sort(rels)[::-1][:k])
    return eval_dcg(rels[:k]) / ideal_dcg

# def eval_iv(distrs=None, labels=None, values=None, distr_smoother=10.0, prop_smoother=0.01):
#     """Evaluate Information-Value from ground-truth labels and feature values.
#     Note that NAs are ignored. Fill NAs beforehand if necessary.

#     Parameters
#     ----------
#     labels : np.ndarray
#         ground-truth, with 2 unique values
#     values : np.ndarray
#         continuous predictions
#     smoother : float
#         pseudo count added to distribution

#     Returns
#     -------
#     iv : float
#     """
#     if distrs is None:
#         distrs = pd.crosstab(values, labels).values  # values as rows, labels as cols
#     assert isinstance(distrs, np.ndarray)

#     d0, d1 = distrs[0], distrs[1]
#     s = d0 + d1
#     s *= float(distr_smoother) / float(s.sum())
#     p = d0 + s
#     q = d1 + s
#     p /= p.sum()
#     q /= q.sum()
#     woe = np.log((p+prop_smoother) / (q+prop_smoother))
#     biv = (p - q) * woe
#     ret = dict()
#     ret['woe'] = woe.values
#     ret['biv'] = biv.values
#     ret['iv'] = biv.sum() # np.inner(p - q, np.log(p / q))
#     m = (p+q) * 0.5
#     ret['js'] = (stats.entropy(p, m) + stats.entropy(q, m)) * 0.5 / np.log(2)
#     return ret




