import os
import argparse
import pandas as pd
import gzip
from datetime import datetime, date, timedelta

def subset(voc_date_first_seen):
    filename = "metadata_2022-09-19.tsv.gz"
    with gzip.open(filename) as g:
        print("Unzipping and reading CSV...")
        gisaid = pd.read_csv(g, delimiter = "\t")
        print("Processing...")
        gisaid = gisaid.fillna("")
        gisaid = gisaid[~gisaid["AA Substitutions"].str.contains("stop|B|X|Z")] # remove sequences with stop codons or ambigous amino acids

        time_index = {-2, 0, 2}

        print("Writing...")
        for voc in voc_date_first_seen:
            for wk in time_index:
                print(voc)
                date = str(datetime.strptime(voc_date_first_seen[voc], '%Y-%m-%d') + timedelta(weeks= wk))[:10]
                print(date)
                gisaid_i = gisaid[gisaid["Submission date"] >= date]
                filepath = "epochs/"+voc+"/results."+date+"/gisaid"
                os.makedirs(filepath)
                gisaid_i.to_csv(filepath+"/metadata_"+date+".tsv.gz", 
                                sep = '\t',
                                index = False, 
                                compression="gzip")

def main(args):                
    argsdict = {"alpha":{"alpha": "2020-12-02"},
                "beta":{"beta": "2021-03-26"},
                "delta":{"delta": "2021-03-31"},
                "BA.1":{"BA.1": "2021-11-22"},
                "BA.2":{"BA.2": "2021-11-27"},
                "BA.2.12.1":{"BA.2.12.1": "2022-02-10"},
                "BA.4":{"BA.4": "2022-04-12"}}
    subset(argsdict[args.voc])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subset metadata file")
    parser.add_argument("--voc", default=None)
    args = parser.parse_args()
    
    main(args)
