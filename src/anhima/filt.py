"""
Utility functions for working with genotype data.

See also the examples at:

- http://nbviewer.ipython.org/github/alimanfoo/anhima/blob/master/examples/filt.ipynb

Richard Pearson, 2014
"""


# third party dependencies
import numpy as np
import scipy
import matplotlib.pyplot as plt
import collections
import matplotlib as mpl
import numexpr as ne


def plot_diagnostic_track(variable_values, diagnostic_values, track_type=None,
                          filters=None, variable_name=None,
                          diagnostic_name='diagnostic variable',
                          should_add_jitter=True, jitter_scale=1e-10,
                          thresholds=None, bins=100,
                          color='red', x_mark_values=None,
                          x_mark_color='black', y_mark_values=None,
                          y_mark_color='black', xlabel=None,
                          median_info_on_x_labels=True,
                          percentiles_on_x_labels=True,
                          should_label_only_x_marks=False,
                          ylabel=None, ax=None,
                          plot_kwargs=None):

    # check input arrays
    variable_values = np.asarray(variable_values)
    # diagnostic_values = np.asarray(diagnostic_values)
    diagnostic_values = np.ma.masked_array(
        diagnostic_values,
        np.isnan(diagnostic_values)
    )
    assert variable_values.ndim == 1
    assert diagnostic_values.ndim == 1
    assert len(variable_values) == len(diagnostic_values)
    if filters is not None:
        assert filters.ndim == 1
        assert len(variable_values) == len(filters)
        variable_values = variable_values[filters]
        diagnostic_values = diagnostic_values[filters]
    if x_mark_values is not None:
        if not isinstance(x_mark_values, collections.Iterable):
            x_mark_values = [x_mark_values]

    # In order to make values distinct, it can be useful to add a small amount
    # of random jitter. Particularly useful, e.g. for boolean variables, or
    # those with a small number of possible integer values
    if should_add_jitter:
        variable_values = 1.0*variable_values + np.random.normal(
            loc=0.0, scale=jitter_scale, size=len(variable_values))

    # Determine thresholds for value bins. If not set, will assign values to
    # equal (in terms of number of variants) size bins
    if thresholds is None:
        percentiles = ((np.arange(bins + 1) * 100.0) / bins).tolist()
        thresholds = np.array(
            scipy.stats.scoreatpercentile(variable_values, percentiles)
        )
    else:
        if percentiles_on_x_labels:
            percentiles = np.array(
                [
                    scipy.stats.percentileofscore(variable_values, x)
                    for x in thresholds
                ]
            )
        else:
            percentiles = None

    number_of_bins = len(thresholds) - 1

    # Determine where to place vertical line at mark_value
    x_mark_pos_list = list()
    for i, mark_value in enumerate(x_mark_values):
        if issubclass(variable_values.dtype.type, np.integer):
            x_mark_pos_list.append(scipy.stats.percentileofscore(
                variable_values, mark_value, kind='strict'))
        else:
            x_mark_pos_list.append(scipy.stats.percentileofscore(
                variable_values, mark_value))

    # determine track type if not supplied
    if track_type is None:
        if diagnostic_values.dtype == 'bool':
            track_type = 'count'
        elif isinstance(diagnostic_values[0], np.number):
            track_type = 'mean'
        else:
            raise RuntimeError('''Cannot determine track type for diagnostic
            variable with dtype %s''' % diagnostic_values.dtype)

    if track_type == 'count':
        bin_values = np.zeros(number_of_bins, dtype=np.integer)
    elif track_type in ['mean', 'ratio']:
        bin_values = np.zeros(number_of_bins, dtype=np.float)
    else:
        raise RuntimeError('''Unknown track_type %s''' % track_type)

    for bin_number in np.arange(number_of_bins):
        flt = (
            (variable_values >= thresholds[bin_number]) &
            (variable_values < thresholds[bin_number+1])
        )
        if track_type == 'count':
            bin_values[bin_number] = np.count_nonzero(diagnostic_values[flt])
        elif track_type == 'bin_proportion':
            bin_values[bin_number] = (
                np.count_nonzero(diagnostic_values[flt]) * 1.0 /
                len(diagnostic_values[flt])
            )
        elif track_type == 'all_proportion':
            bin_values[bin_number] = (
                np.count_nonzero(diagnostic_values[flt]) * 1.0 /
                np.count_nonzero(diagnostic_values)
            )
        elif track_type == 'mean':
            bin_values[bin_number] = np.mean(diagnostic_values[flt])
        elif track_type == 'ratio':
            count_true = np.count_nonzero(diagnostic_values[flt])
            count_false = np.count_nonzero(
                np.logical_not(diagnostic_values[flt])
            )
            ratio = (count_true * 1. / count_false) if count_false > 0 else 0.0
            bin_values[bin_number] = ratio

    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x*2, 1))
        ax = fig.add_subplot(111)

    # Determine y-label
    if ylabel is None:
        if track_type == 'count':
            ylabel = "Number of %s" % diagnostic_name
        elif track_type == 'bin_proportion':
            ylabel = "Proportion of bin which are %s" % diagnostic_name
        elif track_type == 'all_proportion':
            ylabel = "Proportion of %s which are in bin" % diagnostic_name
        elif track_type == 'mean':
            ylabel = "Mean of %s" % diagnostic_name
        elif track_type == 'ratio':
            ylabel = "True/False ratio of %s" % diagnostic_name

    # plot values
    ax.bar(np.arange(number_of_bins), bin_values, 1, color=color)
    ax.set_ylabel(ylabel, rotation='horizontal', horizontalalignment='right')
    if variable_name is None:
        ax.set_xticks([])
    else:
        if issubclass(variable_values.dtype.type, np.integer):
            x_labels = np.array(
                map(str, [int(round(threshold)) for threshold in thresholds])
            )[np.arange(number_of_bins)]
        else:
            x_labels = np.array(
                map(str, [round(threshold, 6) for threshold in thresholds])
            )[np.arange(number_of_bins)]
        if median_info_on_x_labels:
            thresholds_as_proportion_of_median = (
                thresholds[:-1] /
                np.median(variable_values)
            )
            thresholds_as_SDs_from_median = (
                (thresholds[:-1] - np.median(variable_values)) /
                np.std(variable_values)
            )
            label_extensions = np.core.defchararray.add(
                np.array(
                    [", %.2f" % x for x in thresholds_as_proportion_of_median]
                ),
                np.array(
                    [", %.2f" % x for x in thresholds_as_SDs_from_median]
                )
            )
            x_labels = np.core.defchararray.add(x_labels, label_extensions)
        if percentiles_on_x_labels:
            # percentiles = np.array(
            #     [
            #         scipy.stats.percentileofscore(variable_values, x)
            #         for x in thresholds[:-1]
            #     ]
            # )
            x_labels = np.core.defchararray.add(
                x_labels,
                np.array(
                    [", %.1f" % x for x in percentiles[:-1]]
                )
            )
        ax.set_xticks(np.arange(number_of_bins))
        ax.set_xticklabels(x_labels, rotation='vertical')
        ax.set_xlabel(xlabel)

    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_xlim(0, number_of_bins)
    # if mark_values is not None:
    for x_mark_pos in x_mark_pos_list:
        ax.axvline(x_mark_pos, color=x_mark_color)

    for y_mark_value in y_mark_values:
        ax.axhline(y_mark_value, color=y_mark_color)

    return ax


def plot_diagnostics(variable_values, diagnostic_values_dict, track_types,
                     filters=None, variable_name='unknown',
                     should_add_jitter=True, jitter_scale=1e-10,
                     thresholds=None, bins=100,
                     colors='wbgrcmyk', x_mark_values=None,
                     x_mark_color='black',
                     y_mark_values=None, y_mark_color='black',
                     median_info_on_x_labels=True,
                     percentiles_on_x_labels=True,
                     ylabel=None, plot_filename=None, ax=None,
                     plot_kwargs=None):

    # check input arrays
    variable_values = np.asarray(variable_values)
    assert variable_values.ndim == 1
    for diagnostic_names in diagnostic_values_dict:
        diagnostic_values_dict[diagnostic_names] = np.asarray(
            diagnostic_values_dict[diagnostic_names]
        )
        assert diagnostic_values_dict[diagnostic_names].ndim == 1
        assert len(variable_values) == len(
            diagnostic_values_dict[diagnostic_names]
        )

    # repeat colors if not enough
    number_of_tracks = len(diagnostic_values_dict)
    colors = colors * (np.ceil(number_of_tracks / float(len(colors))))

    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x * 2, number_of_tracks))

    for (i, diagnostic_name) in enumerate(diagnostic_values_dict):
        ax = fig.add_subplot(number_of_tracks, 1, i+1)
        if i == number_of_tracks - 1:
            variable_name_this_track = variable_name
        else:
            variable_name_this_track = None

        plot_diagnostic_track(variable_values,
            diagnostic_values_dict[diagnostic_name], track_type=track_types[i],
            filters=filters, variable_name=variable_name_this_track,
            diagnostic_name=diagnostic_name,
            should_add_jitter=should_add_jitter, jitter_scale=jitter_scale,
            thresholds=thresholds, bins=bins, color=colors[i],
            x_mark_values=x_mark_values, x_mark_color=x_mark_color,
            y_mark_values=y_mark_values, y_mark_color=y_mark_color,
            xlabel=variable_name_this_track,
            median_info_on_x_labels=median_info_on_x_labels,
            percentiles_on_x_labels=percentiles_on_x_labels,
            ylabel=ylabel, ax=ax, plot_kwargs=plot_kwargs)

    # save
    if plot_filename is not None:
        fig.savefig(plot_filename, bbox_inches='tight')

    return fig


def plot_filtering_curve(variable_values, bad_snps, high_values_are_bad=False,
                         variable_name=None, mark_value=None,
                         curve_color='blue', mark_color='black',
                         xlabel='Number of variants',
                         ylabel='Number of false positives',
                         max_num_values=None, max_num_bad_snps=None,
                         max_proportion_values=None,
                         max_proportion_bad_snps=None,
                         ax=None,
                         plot_kwargs=None):
    """Plots an ROC curve for a single variable.

    Parameters
    ----------

    variable_values : array_like, numeric
        A 1-dimensional array of values of some variant-level variable.
    bad_snps : array_like, bool
        A 1-dimensional boolean where True indicates the variant is to be
        considered some kind of false positive. This must be the same length
        as `variable_values`.
    high_values_are_bad : bool, optional
        If True, variable is considered to be ranked with lower values
        indicating "better" variants and higher values indicating "worse"
        variants. If False, variable is considered to be ranked with lower
        values indicating "worse" variants and higher values indicating "better"
        variants.
    variable_name : string, optional
        The name of the variable for which `variable_values` are given
    mark_value : numeric, optional
        The value of the variable at which to draw a vertical line
    curve_color : string, optional
        Color to be used for plotting the curve
    mark_color : string, optional
        Color to be used for vertical line
    xlabel : string, optional
        Text to be displayed on x-axis
    ylabel : string, optional
        Text to be displayed on y-axis
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    plot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.plot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check input arrays
    # Note that it was necessary to set dtype to np.float in the following as
    # uint were getting incorrectly ranked by np.argsort
    variable_values = np.asarray(variable_values, dtype=np.float)
    bad_snps = np.asarray(bad_snps)
    assert variable_values.ndim == 1
    assert bad_snps.ndim == 1
    assert len(variable_values) == len(bad_snps)
    
    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x, (x//3) * 2))
        ax = fig.add_subplot(111)

    # determine maximum numbers if only proportions given
    if max_num_values is None and max_proportion_values is not None:
        max_num_values = int(
            len(variable_values) * max_proportion_values
        )
    if max_num_bad_snps is None and max_proportion_bad_snps is not None:
        max_num_bad_snps = int(
            np.count_nonzero(bad_snps) * max_proportion_bad_snps
        )

    # determine values
    if high_values_are_bad:
        sort_indices = np.argsort(-(variable_values))
        if mark_value is not None:
            line_position_x = sum(variable_values >= mark_value)
    else:
        sort_indices = np.argsort(variable_values)
        if mark_value is not None:
            line_position_x = sum(variable_values <= mark_value)
    cumulative_bad_snps = np.cumsum(bad_snps[sort_indices])
    
    # plot values
    if plot_kwargs is None:
        plot_kwargs = dict()
    ax.plot(np.arange(len(cumulative_bad_snps)), cumulative_bad_snps,
            color=curve_color, label=variable_name, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xlim = len(variable_values)
    ylim = cumulative_bad_snps[-1]
    if max_num_values is not None:
        xlim = max_num_values
        ylim = cumulative_bad_snps[max_num_values]
    if max_num_bad_snps is not None:
        xlim = np.min(np.where(cumulative_bad_snps >= max_num_bad_snps)[0][0])
        ylim = max_num_bad_snps
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)

    # add vertical line at given value (e.g. filtering threshold)
    if mark_value is not None:
        ax.axvline(line_position_x, color=mark_color)
        if line_position_x < len(cumulative_bad_snps):
            ax.axhline(cumulative_bad_snps[line_position_x], color=mark_color)

    return ax, xlim, ylim


def random_colors(n=1, seed_value=1):
    import random
    random.seed(seed_value)
    colors = list()
    r = lambda: random.randint(0,255)
    for i in range(n):
        colors.append('#%02X%02X%02X' % (r(), r(), r()))
    return colors


def plot_filtering_curves(variants, bad_snps, filters=None, plot_variables=None,
                          high_values_are_bad_dict=dict(),
                          mark_variable=None, mark_value=None, colors=None,
                          mark_color='black', xlabel='Number of variants',
                          ylabel='Number of false positives',
                          max_num_values=None, max_num_bad_snps=None,
                          max_proportion_values=None,
                          max_proportion_bad_snps=None,
                          ax=None, plot_kwargs=None):
    """Plots a set of ROC curves for a number of variables.

    Parameters
    ----------

    variants : dict-like, numeric
        A set of 1-dimensional arrays of values of variant-level variables. E.g.
        this might take the form of a numpy structure array or a h5py Dataset.
        All arrays must have the same length.
    bad_snps : array_like, bool
        A 1-dimensional boolean array where True indicates the variant is to be
        considered some kind of false positive. This must be the same length
        as each array in `variants`.
    filters : array_like, bool
        A 1-dimensional boolean array where True indicates the variant is to be
        included in the analysis. Note that if supplied this will filter both
        variants and bad_snps.
    plot_variables : array_like, optional
        A 1-dimensional array of names of variables to be plotted.
    high_values_are_bad_dict : dict, optional
        If True, variable is considered to be ranked with lower values
        indicating "better" variants and higher values indicating "worse"
        variants. If False, variable is considered to be ranked with lower
        values indicating "worse" variants and higher values indicating "better"
        variants.
    mark_variable : string, optional
        The name of the variable for which to draw a vertical line
    mark_value : numeric, optional
        The value of the variable at which to draw a vertical line
    colors : sequence, optional
        Colors to use for plotting the different curves.
    mark_color : string, optional
        Color to be used for vertical line
    xlabel : string, optional
        Text to be displayed on x-axis
    ylabel : string, optional
        Text to be displayed on y-axis
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    plot_kwargs : dict-like
        Additional keyword arguments passed through to `plt.plot`.

    Returns
    -------

    ax : axes
        The axes on which the plot was drawn.

    """

    # check plot variables are OK
    if plot_variables is None:
        plot_variables = list()
        for variable_name in variants.keys():
            if isinstance(variants[variable_name][0], np.number):
                plot_variables.append(variable_name)

    # generate random colours if not supplied
    if colors is None:
        colors = random_colors(len(plot_variables))

    # repeat colors if not enough
    colors = colors * (np.ceil(len(plot_variables) / float(len(colors))))

    # set up axes
    if ax is None:
        x = plt.rcParams['figure.figsize'][0]
        fig = plt.figure(figsize=(x*2, (x//3) * 4))
        ax = fig.add_subplot(111)

    min_xlim = len(bad_snps)
    max_ylim = 0

    for i, variable_to_test in enumerate(plot_variables):

        # filter variants if necessary
        if filters is None:
            # the ellipses here will correctly deal with h5py Dataset objects
            variable_values = variants[variable_to_test][...]
            bad_snps_to_use = bad_snps
        else:
            variable_values = variants[variable_to_test][filters]
            bad_snps_to_use = bad_snps[filters]

        # determine whether variable ranked from bad to good or good to bad
        if variable_to_test in high_values_are_bad_dict.keys():
            high_values_are_bad = high_values_are_bad_dict[variable_to_test]
        else:
            num_bad_high_values = np.count_nonzero(
                bad_snps_to_use[variable_values >= np.median(variable_values)]
            )
            num_bad_low_values = np.count_nonzero(
                bad_snps_to_use[variable_values < np.median(variable_values)]
            )
            high_values_are_bad = (num_bad_high_values > num_bad_low_values)

        # is this the variable we want to mark with a vertical line?
        if variable_to_test == mark_variable:
            mark_value_this_var = mark_value
        else:
            mark_value_this_var = None

        # create plot for this variable
        ax, xlim, ylim = plot_filtering_curve(
            variable_values, bad_snps_to_use,
            high_values_are_bad=high_values_are_bad,
            variable_name=variable_to_test, mark_value=mark_value_this_var,
            curve_color=colors[i], mark_color=mark_color, xlabel=xlabel,
            ylabel=ylabel, max_num_values=max_num_values,
            max_num_bad_snps=max_num_bad_snps,
            max_proportion_values=max_proportion_values,
            max_proportion_bad_snps=max_proportion_bad_snps,
            ax=ax, plot_kwargs=plot_kwargs)

        if xlim < min_xlim:
            min_xlim = xlim
        if ylim > max_ylim:
            max_ylim = ylim

    ax.set_xlim(0, min_xlim)
    ax.set_ylim(0, max_ylim)

    ax.legend(loc='lower right')

    return ax


def get_fp_percentile(variable_values, bad_snps, variants_percentile=0.1,
                      high_values_are_bad=False):
    """Determines the proportion of false positives that are included in a given
    proportion of all variants when ranked by a set of variant values.

    Parameters
    ----------

    variable_values : array_like, numeric
        A 1-dimensional array of values of some variant-level variable.
    bad_snps : array_like, bool
        A 1-dimensional boolean where True indicates the variant is to be
        considered some kind of false positive. This must be the same length
        as `variable_values`.
    variants_percentile : float, default 0.1
        Calculate the proportion of false positive variants for
        `variants_percentile` of all variants
    high_values_are_bad : bool, optional
        If True, variable is considered to be ranked with lower values
        indicating "better" variants and higher values indicating "worse"
        variants. If False, variable is considered to be ranked with lower
        values indicating "worse" variants and higher values indicating "better"
        variants.

    Returns
    -------

    fp_percentile : float
        Proportion of false positive variants for `variants_percentile`
        of all variants, when ranking based on `variable_values`

    """

    # check input arrays
    variable_values = np.asarray(variable_values)
    bad_snps = np.asarray(bad_snps)
    assert variable_values.ndim == 1
    assert bad_snps.ndim == 1
    assert len(variable_values) == len(bad_snps)

    # determine values
    if high_values_are_bad:
        sort_indices = np.argsort(-(variable_values))
    else:
        sort_indices = np.argsort(variable_values)
    cumulative_bad_snps = np.cumsum(bad_snps[sort_indices])

    number_of_variants = int(len(variable_values) * variants_percentile)
    number_of_fp = cumulative_bad_snps[number_of_variants]
    proportion_of_fp = float(number_of_fp) / np.sum(bad_snps)

    return proportion_of_fp


def get_variants_percentile(variable_values, bad_snps, fp_percentile=0.9,
                            high_values_are_bad=False):
    """Determines the proportion of variants that would need to be filtered out
    to remove a given proportion of false positives when ranking by a set of
    variant values.

    Parameters
    ----------

    variable_values : array_like, numeric
        A 1-dimensional array of values of some variant-level variable.
    bad_snps : array_like, bool
        A 1-dimensional boolean where True indicates the variant is to be
        considered some kind of false positive. This must be the same length
        as `variable_values`.
    fp_percentile : float, default 0.1
        Calculate the proportion of all variants for `fp_percentile`
        of false positive variants
    high_values_are_bad : bool, optional
        If True, variable is considered to be ranked with lower values
        indicating "better" variants and higher values indicating "worse"
        variants. If False, variable is considered to be ranked with lower
        values indicating "worse" variants and higher values indicating "better"
        variants.

    Returns
    -------

    variants_percentile : float
        Proportion of all variants for `fp_percentile` of false positive
        variants, when ranking based on `variable_values`

    """

    # check input arrays
    variable_values = np.asarray(variable_values)
    bad_snps = np.asarray(bad_snps)
    assert variable_values.ndim == 1
    assert bad_snps.ndim == 1
    assert len(variable_values) == len(bad_snps)

    # determine values
    if high_values_are_bad:
        sort_indices = np.argsort(-(variable_values))
    else:
        sort_indices = np.argsort(variable_values)
    cumulative_bad_snps = np.cumsum(bad_snps[sort_indices])

    number_of_fp = int(np.sum(bad_snps) * fp_percentile)
    number_of_variants = np.argmax(cumulative_bad_snps >= number_of_fp)
    proportion_of_variants = float(number_of_variants) / len(variable_values)

    return proportion_of_variants


def get_variants_percentiles(variants, bad_snps, fp_percentile=0.9,
                             filters=None, evaluation_variables=None,
                             high_values_are_bad_dict=dict()):

    # set up output dict
    out = dict()

    # check evaluation variables are OK
    if evaluation_variables is None:
        evaluation_variables = list()
        for variable_name in variants.keys():
            if isinstance(variants[variable_name][0], np.number):
                evaluation_variables.append(variable_name)

    for i, variable_to_test in enumerate(evaluation_variables):

        # filter variants if necessary
        if filters is None:
            # the ellipses here will correctly deal with h5py Dataset objects
            variable_values = variants[variable_to_test][...]
        else:
            variable_values = variants[variable_to_test][filters]

        # determine whether variable ranked from bad to good or good to bad
        if variable_to_test in high_values_are_bad_dict.keys():
            high_values_are_bad = high_values_are_bad_dict[variable_to_test]
        else:
            num_bad_high_values = np.count_nonzero(
                bad_snps[variable_values >= np.median(variable_values)]
            )
            num_bad_low_values = np.count_nonzero(
                bad_snps[variable_values < np.median(variable_values)]
            )
            high_values_are_bad = (num_bad_high_values > num_bad_low_values)

        # calculate proportion of variants for this variable
        out[variable_to_test] = get_variants_percentile(
            variable_values, bad_snps, high_values_are_bad=high_values_are_bad,
            fp_percentile= fp_percentile)

    return out
