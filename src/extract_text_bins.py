import glob, os
import pandas as pd
import numpy as np
import json


# set input and output paths here
t_dir = "/home/ibilgin/brain_encoding/json_aa/"
output_dir = "/home/ibilgin/brain_encoding/"

transcript_list = sorted(glob.glob(f'{t_dir}/friends_s*_aa.json'))
bin = 0.5
TR = 1.49


for t_path in transcript_list[:1]:

    epi_num = os.path.basename(t_path).split("_")[1]
    print(epi_num)
    
    with open(t_path) as json_file:
        t = json.load(json_file)
    
    words = t['results']['channels'][0]['alternatives'][0]['words']
    # print(words)
    tsv_dict = {
        "text_per_bin": [],
        "words_per_bin": [],
        "onsets_per_bin": [],
        "durations_per_bin": [],
    }

    print(words[-1]["end"])

    # from the 0 till the last word's offset + 3 second, it creates TR timesamps
    start_times = [x for x in np.arange(0, words[-1]["end"] + 3.0, bin)]
    # print(start_times)
    j = 0
    for k, start in enumerate(start_times[:40]):
        word_string = ""
        bin_words = []
        bin_onsets = []
        bin_durations = []
        tr_values = []
            
        while j < len(words) and words[j]["end"] < start + bin:
            print(f"start={start+bin}")
            time = words[j]["end"]
            print(f"word end: {time}")
            word_string += words[j]["word"] + " "
            bin_words.append(words[j]["word"])
            bin_onsets.append(words[j]["start"])
            bin_durations.append(np.float32(words[j]["end"]-words[j]["start"]))
            j += 1
            
        tsv_dict["text_per_bin"].append(word_string)
        tsv_dict["words_per_bin"].append(bin_words)
        tsv_dict["onsets_per_bin"].append(bin_onsets)
        tsv_dict["durations_per_bin"].append(bin_durations)

    bin_dict = {
        "text_per_bin": [],
        "onsets_per_bin": [],
        "durations_per_bin": [],
    }


    df_results = pd.DataFrame.from_dict(tsv_dict)
    df_results.insert(loc=0, column="episode", value=epi_num)
    df_results.to_csv(f"{output_dir}/friends_{epi_num}_aa.tsv", sep='\t', header=True, index=False)
    
    for i in range(321, len(tsv_dict["words_per_bin"])):
        if tsv_dict["words_per_bin"][i]:  # Check if the list is not empty
            bin_dict["onsets_per_bin"].append(tsv_dict["onsets_per_bin"][i][-1])
            bin_dict["durations_per_bin"].append(tsv_dict["durations_per_bin"][i][-1])
            bin_dict["text_per_bin"].append(tsv_dict["text_per_bin"][i][-1])
                    
    