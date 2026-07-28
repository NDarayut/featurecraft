# Benchmark Report

## Datasets

| abbrev | dataset | task | sector | metric |
|---|---|---|---|---|
| QB | qsar-biodegradation | classification | chemistry | auc |
| E | electricity | classification | energy | auc |
| BM | bank-marketing | classification | finance | auc |
| BPC | bnp-paribas-claims | classification | finance | auc |
| GC | german-credit | classification | finance | auc |
| HCD | home-credit-default | classification | finance | auc |
| ICF | ieee-cis-fraud | classification | finance | auc |
| TC | telecom-churn | classification | finance/telco | auc |
| J | jannis | classification | general | auc_ovr |
| N | nomao | classification | general | auc |
| BCW | breast-cancer-wisconsin | classification | healthcare | auc |
| D1 | diabetes-130us | classification | healthcare | auc_ovr |
| HD | heart-disease | classification | healthcare | auc_ovr |
| BM2 | broken-machine | classification | industrial | auc |
| VS | vehicle-sensit | classification | physical-sensor | auc_ovr |
| C | covertype | multiclass | physical-science | auc_ovr |
| MM | microsoft-mslr | regression | general | r2 |
| CH | california-housing | regression | general/real-estate | r2 |
| HP | house-prices | regression | general/real-estate | r2 |
| M | medical | regression | healthcare | r2 |
| CS | concrete-strength | regression | physical/materials | r2 |
| S | superconductivity | regression | physical/materials | r2 |

## Overview

Mean held-out score across the three model families (per-family scores are fold-means; metrics differ per dataset -- see the legend).

| method | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | **0.924** | 0.896 | 0.882 | 0.704 | 0.808 | **0.694** | **0.854** | 0.821 | **0.812** | 0.989 | 0.997 | **0.655** | 0.788 | **0.574** | **0.927** | **0.851** | **0.145** | 0.724 | 0.193 | 0.972 | 0.766 | 0.849 |
| autofeat | 0.920 | — | — | — | 0.798 | — | — | 0.824 | — | — | 0.996 | — | 0.788 | — | — | — | — | 0.763 | — | 0.972 | **0.867** | **0.855** |
| featurecraft | 0.918 | **0.910** | **0.894** | **0.710** | 0.808 | **0.694** | — | **0.830** | 0.812 | **0.989** | 0.997 | **0.655** | 0.788 | **0.574** | **0.927** | **0.851** | — | 0.724 | **0.290** | 0.969 | 0.851 | 0.849 |
| openfe | 0.920 | 0.907 | — | — | **0.810** | — | — | 0.806 | — | — | **0.997** | — | **0.789** | — | — | — | — | **0.795** | — | **0.975** | 0.848 | — |

## Per-method scores

| method | model | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | knn | 0.905 | 0.905 | 0.850 | 0.632 | 0.780 | 0.591 | 0.774 | 0.790 | 0.741 | 0.983 | 0.999 | 0.595 | 0.764 | 0.499 | 0.909 | 0.852 | 0.084 | 0.705 | 0.655 | 0.970 | 0.716 | 0.887 |
|  | linear | 0.931 | 0.816 | 0.863 | 0.724 | 0.815 | 0.739 | 0.850 | 0.848 | 0.816 | 0.988 | 0.994 | 0.654 | 0.791 | 0.498 | 0.922 | 0.827 | 0.131 | 0.614 | -0.659 | 0.976 | 0.647 | 0.740 |
|  | tree | 0.937 | 0.968 | 0.934 | 0.755 | 0.829 | 0.753 | 0.938 | 0.826 | 0.878 | 0.995 | 0.998 | 0.715 | 0.809 | 0.725 | 0.949 | 0.872 | 0.219 | 0.855 | 0.582 | 0.970 | 0.935 | 0.921 |
| autofeat | knn | 0.901 | — | — | — | 0.766 | — | — | 0.797 | — | — | 0.999 | — | 0.764 | — | — | — | — | 0.739 | — | 0.971 | 0.805 | 0.889 |
|  | linear | 0.922 | — | — | — | 0.809 | — | — | 0.849 | — | — | 0.993 | — | 0.791 | — | — | — | — | 0.695 | — | 0.976 | 0.861 | 0.756 |
|  | tree | 0.937 | — | — | — | 0.820 | — | — | 0.826 | — | — | 0.997 | — | 0.809 | — | — | — | — | 0.854 | — | 0.969 | 0.935 | 0.920 |
| featurecraft | knn | 0.898 | 0.925 | 0.860 | 0.647 | 0.780 | 0.591 | — | 0.808 | 0.741 | 0.983 | 0.999 | 0.595 | 0.764 | 0.499 | 0.909 | 0.852 | — | 0.705 | 0.684 | 0.961 | 0.794 | 0.887 |
|  | linear | 0.930 | 0.835 | 0.891 | 0.727 | 0.815 | 0.739 | — | 0.850 | 0.816 | 0.989 | 0.994 | 0.654 | 0.791 | 0.498 | 0.922 | 0.827 | — | 0.614 | -0.392 | 0.976 | 0.828 | 0.740 |
|  | tree | 0.926 | 0.969 | 0.933 | 0.757 | 0.829 | 0.753 | — | 0.831 | 0.877 | 0.995 | 0.998 | 0.715 | 0.809 | 0.725 | 0.949 | 0.872 | — | 0.855 | 0.579 | 0.970 | 0.931 | 0.921 |
| openfe | knn | 0.894 | 0.918 | — | — | 0.776 | — | — | 0.812 | — | — | 0.999 | — | 0.768 | — | — | — | — | 0.832 | — | 0.973 | 0.818 | — |
|  | linear | 0.930 | 0.832 | — | — | 0.822 | — | — | 0.822 | — | — | 0.995 | — | 0.790 | — | — | — | — | 0.685 | — | 0.976 | 0.789 | — |
|  | tree | 0.937 | 0.969 | — | — | 0.831 | — | — | 0.785 | — | — | 0.997 | — | 0.809 | — | — | — | — | 0.869 | — | 0.977 | 0.936 | — |

## Speed (feature-generation wall-time)

| method | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s |
| autofeat | 111.5 s | — | — | — | 27.9 s | — | — | 2.2 min | — | — | 37.9 s | — | 15.2 s | — | — | — | — | 10.0 s | — | 14.3 s | 4.8 s | 80.1 s | 27.9 s |
| featurecraft | 31.6 s | 14.0 s | 24.1 s | 4.3 min | 12.9 s | 3.6 min | — | 12.0 s | 2.1 min | 113.8 s | 9.6 s | 63.0 s | 9.5 s | 2.5 min | 2.1 min | 2.3 min | — | 9.7 s | 101.2 s | 8.2 s | 7.0 s | 87.0 s | 47.3 s |
| openfe | 2.4 min | 92.2 s | — | — | 54.6 s | — | — | 103.6 s | — | — | 35.8 s | — | 17.5 s | — | — | — | — | 38.2 s | — | 34.7 s | 15.3 s | — | 38.2 s |

## Feature counts (before -> after)

Number of columns fed into the downstream models before (original, post max-cols-cap) and after a method's generated features are added.

| method | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 41 -> 41 | 8 -> 8 | 16 -> 16 | 132 -> 132 | 20 -> 20 | 121 -> 121 | 393 -> 393 | 19 -> 19 | 54 -> 54 | 118 -> 118 | 30 -> 30 | 49 -> 49 | 13 -> 13 | 58 -> 58 | 100 -> 100 | 54 -> 54 | 136 -> 136 | 8 -> 8 | 80 -> 80 | 5 -> 5 | 8 -> 8 | 81 -> 81 |
| autofeat | 41 -> 47 | — | — | — | 20 -> 24 | — | — | 19 -> 20 | — | — | 30 -> 32 | — | 13 -> 13 | — | — | — | — | 8 -> 16 | — | 5 -> 11 | 8 -> 12 | 81 -> 120 |
| featurecraft | 41 -> 84 | 8 -> 24 | 16 -> 48 | 132 -> 139 | 20 -> 20 | 121 -> 121 | — | 19 -> 20 | 54 -> 55 | 118 -> 144 | 30 -> 30 | 49 -> 49 | 13 -> 13 | 58 -> 58 | 100 -> 100 | 54 -> 54 | — | 8 -> 8 | 80 -> 130 | 5 -> 14 | 8 -> 23 | 81 -> 81 |
| openfe | 41 -> 51 | 8 -> 18 | — | — | 20 -> 30 | — | — | 19 -> 29 | — | — | 30 -> 40 | — | 13 -> 14 | — | — | — | — | 8 -> 18 | — | 5 -> 15 | 8 -> 18 | — |

## By task

Mean of the per-dataset overview score (mean across model families) over each dataset's group; a method missing on every dataset in a group shows —.

| method | classification | multiclass | regression |
|---|---|---|---|
| baseline | 0.822 | **0.851** | 0.608 |
| autofeat | 0.865 | — | 0.864 |
| featurecraft | 0.822 | **0.851** | 0.737 |
| openfe | **0.872** | — | **0.873** |

## By sector

Mean of the per-dataset overview score (mean across model families) over each dataset's group; a method missing on every dataset in a group shows —.

| method | chemistry | energy | finance | finance/telco | general | general/real-estate | healthcare | industrial | physical-science | physical-sensor | physical/materials |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | **0.924** | 0.896 | 0.788 | 0.821 | 0.648 | 0.459 | 0.853 | **0.574** | **0.851** | **0.927** | 0.808 |
| autofeat | 0.920 | — | 0.798 | 0.824 | — | 0.763 | 0.919 | — | — | — | **0.861** |
| featurecraft | 0.918 | **0.910** | 0.777 | **0.830** | **0.900** | 0.507 | 0.852 | **0.574** | **0.851** | **0.927** | 0.850 |
| openfe | 0.920 | 0.907 | **0.810** | 0.806 | — | **0.795** | **0.921** | — | — | — | 0.848 |

## Failures / timeouts / crashes

| dataset | method | status | count |
|---|---|---|---|
| bank-marketing | autofeat | timeout | 1 |
| bank-marketing | openfe | timeout | 1 |
| bnp-paribas-claims | autofeat | timeout | 1 |
| bnp-paribas-claims | openfe | timeout | 1 |
| broken-machine | autofeat | timeout | 1 |
| broken-machine | openfe | timeout | 1 |
| covertype | autofeat | timeout | 1 |
| covertype | openfe | timeout | 1 |
| diabetes-130us | autofeat | error | 1 |
| diabetes-130us | openfe | timeout | 1 |
| electricity | autofeat | timeout | 1 |
| home-credit-default | autofeat | error | 1 |
| home-credit-default | openfe | timeout | 1 |
| house-prices | autofeat | error | 1 |
| house-prices | openfe | timeout | 1 |
| ieee-cis-fraud | autofeat | timeout | 1 |
| ieee-cis-fraud | featurecraft | timeout | 1 |
| ieee-cis-fraud | openfe | timeout | 1 |
| jannis | autofeat | timeout | 1 |
| jannis | openfe | timeout | 1 |
| microsoft-mslr | autofeat | timeout | 1 |
| microsoft-mslr | featurecraft | timeout | 1 |
| microsoft-mslr | openfe | timeout | 2 |
| nomao | autofeat | timeout | 1 |
| nomao | openfe | timeout | 1 |
| superconductivity | openfe | timeout | 1 |
| vehicle-sensit | autofeat | timeout | 1 |
| vehicle-sensit | openfe | timeout | 1 |