# Copyright Contributors to the Pyro-Cov project.
# SPDX-License-Identifier: Apache-2.0

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist

from pyrocov import mutrans


def plusminus(mean, std):
    """Helper function for generating 95% CIs as the first dim of a tenson"""
    p95 = 1.96 * std
    return torch.stack([mean - p95, mean, mean + p95])


def generate_forecast(fit, queries=None, future_fit=None):
    """Generate forecasts for specified fit

    :param dict fit: the model fit
    :param weekly_cases: aggregate

    """

    # Get locations from dataset to perform query
    location_id = fit["location_id"]

    weekly_cases = fit["weekly_cases"]

    # queries is the list of regions we want to plot
    if queries is None:
        queries = list(location_id)
    elif isinstance(queries, str):
        queries = [queries]

    # These are the T x P x S shaped probabilities (e.g. [35,869,1281])
    probs_orig = fit["mean"]["probs"]

    # Weekly strains is of size TxPxS (e.g. [29, 869, 1281])
    # It does not include forecast periods
    weekly_clades = fit["weekly_clades"]

    # Trim weekly cases to size of weekly_clades
    #   weekly_cases_trimmed = weekly_cases[: len(weekly_clades)]

    # forecast is anything we don't have data for
    # both probs and weekly_clades are of dim TxPxS
    forecast_steps = probs_orig.shape[0] - fit["weekly_clades"].shape[0]
    assert forecast_steps >= 0

    # augment data with +/-1 SD
    probs = plusminus(probs_orig, fit["std"]["probs"])  # [3, T, P, S]

    # Pad weekly_cases [42 x 1070] with entries for the forecasting steps
    # using the last weekly_cases values
    padding = 1 + weekly_cases[-1:].expand(forecast_steps, -1)
    weekly_cases_padded = torch.cat([weekly_cases, padding], 0)

    # Weight weekly cases by strain probabilities
    predicted = probs * weekly_cases_padded[..., None]

    ids = torch.tensor(
        [i for name, i in location_id.items() if any(q in name for q in queries)]
    )

    # get the strain identifiers
    strain_ids = weekly_clades[:, ids].sum([0, 1]).sort(-1, descending=True).indices

    # get date_range
    date_range = mutrans.date_range(len(fit["mean"]["probs"]))

    forecast = {
        "queries": queries,
        "date_range": date_range,
        "strain_ids": strain_ids,
        "predicted": predicted,
        "location_id": location_id,
        "weekly_cases": weekly_cases,
        "weekly_clades": weekly_clades,
        "lineage_id_inv": fit["lineage_id_inv"],
        "weekly_cases_future": future_fit["weekly_cases"]
        if future_fit is not None
        else None,
        "weekly_clades_future": future_fit["weekly_clades"]
        if future_fit is not None
        else None,
    }

    return forecast


def get_forecast_values(forecast):
    """Calculate forecast values for strains

    :param forecast: forecast return values from generate_forecast()

    :return: Dictionary of queries and predicted tensor and observed tensor. Tensors have shapes
        num_queries x 3 (stats) x T x S
    """

    # get the data from the input forecast
    queries = forecast["queries"]
    date_range = forecast["date_range"]
    strain_ids = forecast["strain_ids"]
    predicted = forecast["predicted"]
    location_id = forecast["location_id"]
    weekly_cases = forecast["weekly_cases"]  # T x P
    weekly_clades = forecast["weekly_clades"]  # T x P x S
    lineage_id_inv = forecast["lineage_id_inv"]

    weekly_cases_future = forecast["weekly_cases_future"]
    weekly_clades_future = forecast["weekly_clades_future"]

    # log input shapes
    logging.debug(f"date_range shape {date_range.shape}")
    logging.debug(f"predicted shape {predicted.shape}")
    logging.debug(f"queries length {len(queries)}")
    logging.debug(f"weekly_cases shape {tuple(weekly_cases.shape)}")
    logging.debug(f"weekly_clades shape {tuple(weekly_clades.shape)}")

    # Determine output tensor shapes
    logging.info("Generating output tensor")

    output_predicted_tensor_shape = list(predicted.shape)
    # remove the place dimension as we will be summing over that
    del output_predicted_tensor_shape[-2]
    # append a leftmost dim for each query
    output_predicted_tensor_shape.insert(0, len(queries))

    output_observed_tensor_shape = list(weekly_clades.shape)
    del output_observed_tensor_shape[-2]
    output_observed_tensor_shape.insert(0, len(queries))

    logging.debug(f"Output predicted tensor shape: {output_predicted_tensor_shape}")
    logging.debug(f"Output observed tensor shape: {output_observed_tensor_shape}")

    # Generate output tensors
    output_predicted = torch.zeros(output_predicted_tensor_shape)
    output_observed = torch.zeros(output_observed_tensor_shape)

    # If we are processing future data
    if weekly_cases_future is not None:
        output_observed_future_tensor_shape = list(weekly_clades_future.shape)
        del output_observed_future_tensor_shape[-2]
        output_observed_future_tensor_shape.insert(0, len(queries))
        output_observed_future = torch.zeros(output_observed_future_tensor_shape)

    # Process each query
    for k, query in enumerate(queries):
        logging.info(f"--- Processing query {k}")
        ids = torch.tensor([i for name, i in location_id.items() if query in name])
        logging.debug(f"ids in query: {ids}")
        logging.debug(f"ids length:{len(ids)}")

        # Calculate predicted
        pred = predicted.index_select(-2, ids).sum(-2)
        # Sum over places
        pred /= pred[1].sum(-1, True).clamp_(min=1e-8)
        logging.debug(f"pred shape {tuple(pred.shape)}")
        # of shape 3 x T_total x S, first dim is stats, T_total includes data abd predicted

        # Calculate observed
        obs = weekly_clades[:, ids].sum(1)
        obs /= obs.sum(-1, True).clamp_(min=1e-9)
        logging.debug(f"obs shape {tuple(obs.shape)}")
        # same shape as pred above

        if weekly_cases_future is not None:
            obs_future = weekly_clades_future[:, ids].sum(1)
            obs_future /= obs_future.sum(-1, True).clamp_(min=1e-9)
            output_observed_future[k, :] = obs_future

        logging.debug(f"output_predicted shape: {tuple(output_predicted.shape)}")
        logging.debug(f"output_observed shape: {tuple(output_observed.shape)}")
        logging.debug(f"k: {k}")

        output_predicted[k, :] = pred
        output_observed[k, :] = obs

    return {
        "queries": queries,
        "predicted": output_predicted,  # Query x 3 (stats) x T_total x S
        "observed": output_observed,  # Query x 3 (stats) x T_total x S
        "date_range": date_range,
        "strain_ids": strain_ids,
        "lineage_id_inv": lineage_id_inv,
        "observed_future": output_observed_future
        if weekly_cases_future is not None
        else None,
    }


def get_fit_by_index(fits, i):
    k = list(fits.keys())
    logging.debug(f"key list length {len(k)}")
    key = k[i]
    logging.debug(f"key is {key}")
    fit = fits[key]
    return (key, fit)


def get_available_strains(fit, num_strains=100):
    """Get the strains available for plotting  in the specified fit

    :param fits: fits
    :param fit_i: index of fit key to look at
    :param num_strains: number of strains to pass to generate_forecast

    """

    # Select the fit
    # k = list(fits.keys())  # probably should sort here
    # fit = fits[k[fit_i]]

    # Generate a forecast and get values
    # here
    fc1 = generate_forecast(fit=fit, num_strains=num_strains)
    forecast_values = get_forecast_values(forecast=fc1)

    # Extract the names of the lineages
    strain_ids = forecast_values["strain_ids"]
    lineage_id_inv = forecast_values["lineage_id_inv"]

    return [lineage_id_inv[i] for i in strain_ids]


@torch.no_grad()
def evaluate_fit_forecast(
    fit,
    future_fit,
):
    """
    Evaluate the forecast produced by a fit w.r.t. future data

    :param fit: fit to evaluate
    :param future_fit: fit to get future data values form

    """
    # Ensure the strains of the fit and future_fit match
    assert fit["lineage_id_inv"] == future_fit["lineage_id_inv"]

    # Get the truth from the future dataset
    true = future_fit["weekly_clades"]
    logging.debug(f"initial true shape: {true.shape}")

    # Get the predicted fit
    pred = fit["median"]["probs"]
    logging.debug(f"initial pred shape: {pred.shape}")

    # Restrict to the forecast interval.
    t0 = len(fit["weekly_clades"])
    t1 = min(len(true), len(pred))
    pred = pred[t0:t1]
    true = true[t0:t1]

    # Get indices of the common regions in the two fits
    common_regions = list(future_fit["location_id"].keys() & fit["location_id"].keys())
    future_fit_loc_idx = torch.tensor(
        [future_fit["location_id"][ct] for ct in common_regions]
    )
    fit_loc_idx = torch.tensor([fit["location_id"][ct] for ct in common_regions])

    # Put tensors in same order for place (P)
    true = true.index_select(1, future_fit_loc_idx)
    pred = pred.index_select(1, fit_loc_idx)
    logging.debug(f"true shape: {true.shape}")
    logging.debug(f"pred shape: {pred.shape}")

    # Calculate log likelihood per observation, over time
    log_likelihood = (
        dist.Multinomial(probs=pred, validate_args=False).log_prob(true).sum(-1)
    )
    num_obs = true.sum([1, -1])
    log_likelihood = log_likelihood / num_obs

    # Compute obs-weighted perplexity as baseline for log_likelihood.
    probs = true + 1e-8  # [T, P, S]
    probs /= probs.sum(-1, True)
    entropy = -(probs * probs.log()).sum(-1)  # [T, P]
    perplexity = entropy.exp()
    kl = (probs * (probs.log() - pred.log())).sum(-1)
    # compute a weighted average over regions.
    weight = true.sum(-1)  # [T, P]
    weight /= weight.sum(-1, True)
    entropy = (entropy * weight).sum(1)  # [T]
    perplexity = (perplexity * weight).sum(1)  # [T]
    kl = (kl * weight).sum(1)  # [T]

    pred_sorted, indices_1 = pred.clone().sort(-1)
    true_sorted, indices_2 = probs.clone().sort(-1)

    # Wasterstein summed over place
    p = 2
    wasserstein = (pred_sorted - true_sorted).abs().pow(p).sum(2).div(p).sum(1)

    # Calculate error over time
    error = true - pred * true.sum(-1, True)
    # mae = error.abs().sum(-1).mean(1)
    mae = error.abs().mean(-1).sum(1)
    rmse = error.square().sum(-1).mean(0).sqrt()

    # return the calculated statistics, each batched over time
    return {
        "log_likelihood": log_likelihood,
        "entropy": entropy,
        "perplexity": perplexity,
        "kl": kl,
        "mae": mae,
        "rmse": rmse,
        "wasserstein": wasserstein,
    }


def generate_strain_color_map_default(n_colors, n_colors_distinct=20):
    n_color_black = n_colors - n_colors_distinct
    colors = [f"C{i}" for i in range(n_colors)] + ["black"] * n_color_black
    return colors


def generate_strain_color_map_dict(strains, n_colors_distinct=20):
    n_colors = len(strains)
    n_color_black = n_colors - n_colors_distinct
    c_tmp = [f"C{i}" for i in range(n_colors)] + ["black"] * n_color_black
    color_map = {strains[i]: c_tmp[i] for i in range(len(strains))}
    return color_map


def plot_fit_forecasts(
    fit,
    queries=["England", "USA / California", "Brazil"],
    strains_to_show=None,
    show_forecast=True,
    show_observed=True,
    future_fit=None,
    filename=None,
    forecast_periods_plot=None,
    colors=None,
):
    """
    Function to plot forecasts of specific strains in specific regions

    :param fit: the fit to plot
    :param queries: region queries to plot
    :param strains_to_show: strains to plot
    :param show_forecast: plot model strain fit
    :param show_observed: show the observed points
    :param num_strains: num_strains param to pass downstream
    :param future_fit: optional fit with future data, used to print datapoint in predicted interval
    :param filename: filename to save plot
    """

    logging.debug("Entering plot_fit_forecast()")

    assert strains_to_show is not None

    if isinstance(queries, str):
        logging.debug("queries was string; converting to array")
        queries = [queries]

    logging.debug("Generating forecast...")
    fc1 = generate_forecast(
        fit=fit,
        queries=queries,
        future_fit=future_fit,
    )

    # Check that all the strains to show are in the lineage_id_inv
    assert all([item in fc1["lineage_id_inv"] for item in strains_to_show])

    logging.debug("Getting forecast values...")
    forecast_values = get_forecast_values(forecast=fc1)

    dates = matplotlib.dates.date2num(mutrans.date_range(len(fit["mean"]["probs"])))
    logging.debug(f"dates length: {len(dates)}")

    # Strain ids
    strain_ids = forecast_values["strain_ids"]
    lineage_id_inv = forecast_values["lineage_id_inv"]

    n_queries = len(queries)

    fig, ax = plt.subplots(nrows=n_queries, sharex=True)

    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]

    if colors is None:
        colors = generate_strain_color_map_dict(strain_ids.numpy())

    for i, (k, ax_c, query) in enumerate(zip(range(n_queries), ax, queries)):
        logging.debug(f"** Plotting {queries}")

        # 2nd dim is 1 because we want means only
        sel_forecast = forecast_values["predicted"][k, 1, :]
        sel_forecast_lb = forecast_values["predicted"][k, 0, :]
        sel_forecast_ub = forecast_values["predicted"][k, 2, :]
        sel_observed = forecast_values["observed"][k, :]

        if future_fit is not None:
            sel_observed_future = (
                forecast_values["observed_future"][k, :]
                if future_fit is not None
                else None
            )
            forecast_periods = len(sel_forecast) - len(sel_observed) - 1

        ax_c.set_ylim(0, 1)
        ax_c.set_yticks(())
        ax_c.set_ylabel(query.replace(" / ", "\n"))
        ax_c.set_xlim(dates.min(), dates.max())

        print(f"sel_forecast shape {sel_forecast.shape}")
        print(f"sel_forecast_lb shape {sel_forecast_lb.shape}")

        # Plot one strain at a time
        # for s, color in zip(strain_ids, colors):
        for s in strain_ids:
            color = colors[s.item()]

            strain = lineage_id_inv[s]
            if strain in strains_to_show:
                logging.debug(f"Drawing strain {s}, {strain}")
                if show_forecast:
                    ax_c.fill_between(
                        dates[: len(sel_forecast)],
                        sel_forecast_lb[: len(sel_forecast), s],
                        sel_forecast_ub[: len(sel_forecast), s],
                        color=color,
                        alpha=0.2,
                        zorder=-10,
                    )
                    ax_c.plot(
                        dates[: len(sel_forecast)],
                        sel_forecast[:, s],
                        label=strain,
                        color=color,
                    )
                if show_observed:
                    ax_c.plot(
                        dates[: len(sel_observed)],
                        sel_observed[:, s],
                        lw=0,
                        marker="o",
                        color=color,
                    )
                # Plot actual points from future fit
                if future_fit is not None:
                    last_point = len(sel_observed) + forecast_periods
                    if forecast_periods_plot:
                        last_point = len(sel_observed) + forecast_periods_plot
                    ax_c.plot(
                        dates[len(sel_observed) : last_point],
                        sel_observed_future[len(sel_observed) : last_point, s],
                        lw=0,
                        marker="x",
                        color=color,
                    )
            if i == 0:
                ax_c.legend(loc="upper left", fontsize=8)

    ax_c.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax_c.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    plt.subplots_adjust(hspace=0)

    plt.xticks(rotation=90)
    fig.show()

    if filename:
        plt.savefig(filename)

    return {"colors": colors}
