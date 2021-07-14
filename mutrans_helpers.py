import datetime
import math
import os
import pickle
import re
import logging
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pyro.distributions as dist
from pyrocov import mutrans, pangolin, stats
from pyrocov.stats import normal_log10bf
from pyrocov.util import pretty_print, pearson_correlation


def plusminus(mean, std):
    """Helper function for generating 95% CIs as the first dim of a tenson"""
    p95 = 1.96 * std
    return torch.stack([mean - p95, mean, mean + p95])

def generate_forecast(fit, queries=None, num_strains=10):
    """Generate forecasts for specified fit
    
    #sphinx format 
    :param dict fit: the model fit
    :param weekly_cases: aggregate
    -- 
    
    """
    
    # Get locations from dataset to perform query
    location_id = fit['location_id']
    
    weekly_cases = fit['weekly_cases']

    # queries is the list of regions we want to plot
    if queries is None:
        queries = list(location_id)
    elif isinstance(queries, str):
        queries = [queries]
    
    # These are the T x P x S shaped probabilities (e.g. [35,869,1281])
    probs_orig = fit["mean"]["probs"]
    
    # Weekly strains is of size TxPxS (e.g. [29, 869, 1281])
    # It does not include forecast periods
    weekly_strains = fit['weekly_strains']
        
    # Trim weekly cases to size of weekly_strains
    weekly_cases_trimmed = weekly_cases[:len(weekly_strains)]
        
    # forecast is anything we don't have data for
    # both probs and weekly_strains are of dim TxPxS
    forecast_steps = probs_orig.shape[0] - fit['weekly_strains'].shape[0]
    assert forecast_steps >= 0
    
    # augment data with +/-1 SD
    probs = plusminus(probs_orig, fit["std"]["probs"])  # [3, T, P, S]
    
    # Pad weekly_cases [42 x 1070] with entries for the forecasting steps
    # using the last weekly_cases values
    padding = 1 + weekly_cases[-1:].expand(forecast_steps, -1)
    weekly_cases_padded = torch.cat([weekly_cases, padding], 0)
    
    # Weight weekly cases by strain probabilities
    predicted = probs * weekly_cases_padded[..., None]
    
    ids = torch.tensor([i for name, i in location_id.items() if any(q in name for q in queries)])
    
    # get the strain identifiers
    strain_ids = weekly_strains[:, ids].sum([0, 1]).sort(-1, descending=True).indices
    strain_ids = strain_ids[:num_strains]
    
    # get date_range
    date_range = mutrans.date_range(len(fit['mean']['probs']))
    
    forecast = {
        'queries': queries,
        'date_range': date_range,
        'strain_ids': strain_ids,
        'predicted': predicted,
        'location_id': location_id,
        'weekly_cases': weekly_cases,
        'weekly_strains': weekly_strains,
        'lineage_id_inv': fit['lineage_id_inv'],
    }
    
    return forecast

def plot_forecast(forecast, filename=None, plot_relative_cases=True, plot_relative_samples=True, plot_observed=True, plot_fit=True, plot_fit_ci=True):
    """Plot a forecast generated from the model"""
    
    queries = forecast['queries']
    date_range = forecast['date_range']
    strain_ids = forecast['strain_ids']
    predicted = forecast['predicted']
    location_id = forecast['location_id']
    weekly_cases = forecast['weekly_cases'] # T x P
    weekly_strains = forecast['weekly_strains'] # T x P x S
    lineage_id_inv = forecast['lineage_id_inv']
    
    num_strains = forecast1['strain_ids'].shape[0]
    
    # generate figure and axes -- one axes per query
    fig, axes = plt.subplots(len(queries), figsize=(8, 1.5 + 2 * len(queries)), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
        
    # construct date range for ploting
    dates = matplotlib.dates.date2num(forecast['date_range'])
   
    # generate colors
    colors = [f"C{i}" for i in range(10)] + ["black"] * 90
    assert len(colors) >= num_strains
    light = "#bbbbbb"
    
    for row, (query, ax) in enumerate(zip(queries, axes)):
        
        ids = torch.tensor([i for name, i in location_id.items() if query in name])
        
        print(f"{query} matched {len(ids)} regions")
        
        # Plot relative numbers of cases
        if plot_relative_cases:
            # Sum over places and max normalize
            counts = weekly_cases[:, ids].sum(1)
            counts /= counts.max()
            ax.plot(dates[:len(counts)], counts, "k-", color=light, lw=0.8, zorder=-20)
        
        if plot_relative_samples:
            # Plot relative numbers of samples
            # Sum over places 
            counts = weekly_strains[:, ids].sum([1, 2])
            counts /= counts.max()
            ax.plot(dates[:len(counts)], counts, "k--", color=light, lw=1, zorder=-20)

        # .sum(-2) sums over regions
        pred = predicted.index_select(-2, ids).sum(-2)
        pred /= pred[1].sum(-1, True).clamp_(min=1e-8)
        
        obs = weekly_strains[:, ids].sum(1)
        obs /= obs.sum(-1, True).clamp_(min=1e-9)
        
        # Plot individual strains
        for s, color in zip(strain_ids, colors):
            lb, mean, ub = pred[..., s]
            
            if plot_fit:
                if plot_fit_ci:
                    ax.fill_between(dates, lb, ub, color=color, alpha=0.2, zorder=-10)
                ax.plot(dates, mean, color=color, lw=1, zorder=-9)
                
            strain = lineage_id_inv[s]
            
            if plot_observed:
                ax.plot(
                    dates[:len(obs)], 
                    obs[:, s], 
                    color=color, 
                    lw=0, 
                    marker='o', 
                    markersize=3,
                    label=strain if row == 0 else None)
            
        # Set axis
        ax.set_ylim(0, 1)
        ax.set_yticks(())
        ax.set_ylabel(query.replace(" / ", "\n"))
        ax.set_xlim(dates.min(), dates.max())
        
        # Add legends
        if row == 0:
            ax.legend(loc="upper left", fontsize=8)
        elif row == 1:
            if plot_relative_samples:
                ax.plot([], "k--", color=light, lw=1, label="relative #samples")
            if plot_relative_cases:
                ax.plot([], "k-", color=light, lw=0.8, label="relative #cases")
            ax.plot([], lw=0, marker='o', markersize=3, color='gray', label="observed portion")
            ax.fill_between([], [], [], color='gray', label="predicted portion")
            ax.legend(loc="upper left", fontsize=8)
            
    # x axis dates
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=90)
    
    plt.subplots_adjust(hspace=0)
    
    if filename:
        plt.savefig(filename)