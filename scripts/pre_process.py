import csv
import pandas as pd
from collections import defaultdict
from IPython.core.display import HTML,display
import requests
from bs4 import BeautifulSoup
import unicodedata
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
import collections

def fill_health():
    filename = 'raw_data/health_processed.csv'
    rows = []
    with open(filename, 'r') as f:
        read = csv.reader(f)
        for i in read:
            rows.append(i)
            #print(rows)

    for col in range(2, 8):
        medians = []
        count = 0
        for i in range(1, len(rows)):
            row = rows[i]
            if row[col] != 'null':
                medians.append(float(row[col]))
                count += 1
        if count != 0:
            median = sum(medians) / count
        for i in range(1, len(rows)):
            row = rows[i]
            if row[col] == 'null':
                row[col] = median



    new = [['lga', 'total_dentist+nurse', 'total_medical_person_per_100000']]

    for i in range(1, len(rows)):
        row = rows[i]
        t_d_n = float(row[2]) + float(row[4])
        t_m_p_p = float(row[3]) + float(row[5]) + float(row[7])
        k = [row[1], t_d_n, t_m_p_p]
        new.append(k)

    with open('health_filled.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new)

# =============================================================        
        
def fill_house_price():
    filename = 'raw_data/Suburb_House_final.csv'
    rows = []
    with open(filename, 'r') as f:
        read = csv.reader(f)
        for i in read:
            rows.append(i)

    rows = rows[3:]
    for col in range(1, 6):
        medians = []
        count = 0
        for i in range(len(rows)):
            if rows[i][col] != 'NA':
                medians.append(float(rows[i][col]))
                count += 1

        if count != 0:
            median = sum(medians) / count
        for i in range(len(rows)):
            if rows[i][col] == 'NA':
                rows[i][col] = median

    for i in rows:
        i.append(0)

    rows.insert(0, ['lga', '2019', '2020', 'change 2018-2019', 'change 2009-2019', 'Growth PA', 'no. of sales 2020'])
   
    filename2 = 'raw_data/House_Medians3rdQtr2020-xls.csv'
    rows2 = []
    with open(filename2, 'r') as f:
        read = csv.reader(f)
        for i in read:
            rows2.append(i)

    rows2 = rows2[3:]
    for i in rows2:
        for j in rows[1:]:
            if i[0] == j[0]:
                j[6] = i[1]
                break
    
    for i in rows[1:]:
        i[0] = i[0].lower()
    
    with open('suburb_house_final_filled.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

        
# ================================================================


def finalize_hp():
    # CREATING CONVERSION TABLE FOR LGA AND SUBURBS
    # Create a dictionary to search each suburb's lga name 
    pattern = r'\s\(\w*\)'
    lga_dict = defaultdict(list)
    fname = "raw_data/school.csv"
    df = pd.read_csv(fname)
    df['LGA_Name'] = df['LGA_Name'].str.replace(pattern, '', regex = True)
    for i in range(len(df)):
        suburb = df.at[i,'Address_Town']
        lga = df.at[i,'LGA_Name']
        lga_dict[lga].append(suburb.lower())
    
    # Remove all duplicate and sort the dictionary in lga's alphabetic order 
    new_dict = {a:list(set(b)) for a, b in lga_dict.items()}
    new_dict = collections.OrderedDict(sorted(new_dict.items()))

    # Create a CSV file for the dictionary
    with open('lga-suburb.csv','w',newline='') as f: 
        w = csv.writer(f)
        w.writerow(['LGA','suburb'])
        for key,items in new_dict.items():
            for item in items:
                w.writerow([key,item])

    # ADD EACH SUBURB HOUSE PRICE TO THE CORRESPONDING LGA DICTIONARY 
    df1 = pd.read_csv('suburb_house_final_filled.csv')
    df2 = pd.read_csv('lga-suburb.csv')
    lst = list(df2['suburb'].copy())
    for i in range(len(df1['lga'])):
        suburb_name = df1.loc[i,'lga']
        index = df2[df2['suburb'] == suburb_name].index.values
        if suburb_name in lst:
            df1.loc[i, 'lga'] = ','.join(df2.loc[index,'LGA'])
        
    df1.to_csv("temp_hp.csv", index=False)
    
    # AVERAGE THE HOUSE PRICE IN EACH LGA 
    # Read CSV file
    # final_hp is the manually corrected version of temp_hp done using OpenRefine
    fname = 'data/final_hp.csv'
    df = pd.read_csv(fname)
    # Check if all LGA are separated properly
    df['lga'].str.contains(',')


    # Create dictionary for mean house price in 2019
    dict_2019 = defaultdict(list)
    for i in range(len(df)):
        dict_2019[df.at[i,'lga']].append(df.at[i,'2019'])

    d2019 = defaultdict(list)
    for key in dict_2019:
        d2019[key] = sum(dict_2019[key])/len(dict_2019[key])
    
    with open('lga-hp-2019.csv','w',newline='') as f: 
        w = csv.writer(f)
        w.writerow(['LGA','Mean House Price'])
        for key,items in d2019.items():
         w.writerow([key,items])


# ============================================================

def fill_open_space():
    filename = 'raw_data/open-space_processed.csv'
    rows = []
    with open(filename, 'r', encoding = 'utf-8') as f:
        read = csv.reader(f)
        for i in read:
            rows.append(i)
        
    district = defaultdict(list)
    for i in range(1, len(rows)):
        del rows[i][0]
        district[rows[i][0]].append(rows[i])

    ''' there are some repeating open space listed in the original 
    data set and are differed by some other factors that we are not considering
    so we need to get rid of the repeating open sapce when counting them '''
    non_rep_district = defaultdict(list)
    for lga in district.keys():
        same = set()
        space = district[lga]
        non_rep = []
        for i in space:
            if i not in non_rep:
                non_rep.append(i)
        non_rep_district[lga] = non_rep
    
    
    types = []
    for lga in non_rep_district.keys():
        space = non_rep_district[lga]
        for i in space:
            if i[1] not in types:
                types.append(i[1])
    types.remove('Tertiary institutions')
    types.remove('Non-government schools')
    types.remove('Government schools')
    types.remove('Cemeteries')

    #print(types)


    final = defaultdict(dict)
    for lga in non_rep_district.keys():
        total = 0
        dic = defaultdict(int)
        for type in types:
            dic[type] = 0
        space = non_rep_district[lga]
        for i in space:
            if i[1] in types:
                dic[i[1]] += 1
                total += 1
        final[lga] = dic
        final[lga]['total'] = total

    #print(final)

    result = [['lga', 'Total']]

    for lga in final.keys():
        row = [lga.lower()]
        row.append(final[lga]['total'])
        result.append(row)

    
    with open('open_space_filled.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result)
    
# =========================================================
    
def crime():
    file = 'raw_data/crime.csv'

    with open(file, 'r') as f:
        read = csv.reader(f)
        head = next(read)
        with open('crime_data.csv', 'w') as w:
            write = csv.writer(w)
            write.writerow([head[3], head[4], head[5]])
            for i in read:
                if i[0] == '2020' and i[3] != 'Total' and i[5] != '':
                    write.writerow([i[3].lower(), i[4], i[5]])


# ===========================================================



def highschool_rank19():

    urls = []
    records = [['ranking', 'School', 'Location', 'median VCE Score', 'VCE 40+ %']]
    for i in range(1, 26):
        u = 'https://www.topscores.co/Vic/vce-school-rank-median-vce/2019/?pageno=' + str(i)
        urls.append(u)
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')

        section = soup.find(id='reportCanvas')
        result = section.findNext('table')
        #display(HTML(str(result)))
        rows = result.find_all('tr')
        for j in range(1, 27):
            record = []
            if j != 6:
                row = rows[j]
                cells = row.find_all('td')
                #print("{0}::{1}::{2}::{3}::{4}".format(cells[0].text.strip(), cells[1].text.strip(), cells[2].text.strip(), cells[3].text.strip(), cells[4].text.strip()))
                ranking = int(unicodedata.normalize("NFKD", cells[0].text.strip()))
                record.append(int(ranking))
            
                school = unicodedata.normalize("NFKD", cells[1].text.strip())
                record.append(school)
            
                location = unicodedata.normalize("NFKD", cells[2].text.strip())
                record.append(location)
            
                median = unicodedata.normalize("NFKD", cells[3].text.strip())
                record.append(median)
            
                High = unicodedata.normalize("NFKD", cells[4].text.strip())
                record.append(High)
            
                records.append(record)

    with open('sec_rank.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(records)

# ==========================================================================

def prischool():
    records = [['ranking', 'School', 'Postcode', 'State Overall Score', 'Better Eduction Percentile']]
    url = 'https://bettereducation.com.au/school/primary/vic/melbourne_top_government_primary_schools.aspx'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    section = soup.find(id='ctl00_ContentPlaceHolder1_UpdatePanel1')
    result = section.findNext('table')
    #display(HTML(str(result)))
    rows = result.find_all('tr')
    for j in range(1, 220):
        record = []
        row = rows[j]
        cells = row.find_all('td')
        #print("{0}::{1}::{2}::{3}::{4}".format(cells[0].text.strip(), cells[1].text.strip(), cells[2].text.strip(), cells[3].text.strip(), cells[4].text.strip()))
        ranking = int(unicodedata.normalize("NFKD", cells[0].text.strip()))
        record.append(int(ranking))
            
        school = unicodedata.normalize("NFKD", cells[1].text.strip())
        record.append(school)
            
        postcode = unicodedata.normalize("NFKD", cells[2].text.strip())
        record.append(postcode)
            
        State_Overall_Score = unicodedata.normalize("NFKD", cells[3].text.strip())
        record.append(State_Overall_Score)
            
        BEP = unicodedata.normalize("NFKD", cells[4].text.strip())
        record.append(BEP)
            
        records.append(record)

    with open('pri_rank.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(records)
 

#====================================================================================


def school_preprocess():
    schoolpath = 'raw_data/school.csv'

    # create a csv file with only school name, school type and lga name
    all_school = pd.read_csv(schoolpath, encoding = 'ISO-8859-1')
    update = all_school.loc[:,['School_Name','School_Type', 'LGA_Name']]

    # get rid of any non-alphabetical symbols in school names
    update['School_Name'] = update['School_Name'].replace(regex=r'\'', value='')
    update.to_csv('revised_school.csv', index=False)


# ====================================================================

def highschool_weight():
    schoolpath = 'data/manual_revised_school.csv'
    sec_rankpath = 'data/sec_rank.csv'

    all_school = pd.read_csv(schoolpath, encoding = 'ISO-8859-1')
    all_sec = all_school.loc[all_school['School_Type'].isin(['Secondary', 'Pri/Sec'])]
  
    # change all school names to lower case
    all_school['School_Name'] = all_school['School_Name'].str.lower()

    sec_rank = pd.read_csv(sec_rankpath, encoding = 'ISO-8859-1')\
              .loc[:, ['School', 'ranking', 'median VCE Score']]\
              .rename(columns={"School": "School_Name"})\
              .dropna()\
              .replace(0, 15)
    sec_rank['School_Name'] = sec_rank['School_Name'].str.lower()

    # match the school ranking by names
    ranked = all_sec.merge(sec_rank, how='left', on='School_Name')\
                  .sort_values(by = 'ranking')\
                  .reset_index(drop=True)\
                  .replace(np.nan, 15)

    # find mean and sd of the median vce score
    vcescore = ranked.loc[:,['median VCE Score']]
    mean = vcescore.mean()
    sd = vcescore.std() 

    # assume normal distribution, find the probability that a school is less than or equal to its median vce score
    # assign the probability as a weighted score to each school

    weigh_score = []

    for i in range(len(ranked)):
        score = scipy.stats.norm(mean, sd).cdf(ranked.loc[i]['median VCE Score'])
        weigh_score.append(float(score))

    ranked['weighted_score'] = weigh_score
    df = ranked.loc[:, ['School_Name', 'LGA_Name', 'weighted_score']]
    df.to_csv("high_school_weighscore.csv", index=False)


# =======================================================


def primary_weight():
    schoolpath = 'data/manual_revised_school.csv'
    pri_rankpath = 'data/pri_rank.csv'

    all_school = pd.read_csv(schoolpath, encoding = 'ISO-8859-1')

    # change all school names to lower case
    all_school['School_Name'] = all_school['School_Name'].str.lower()
    all_pri = all_school.loc[all_school['School_Type'].isin(['Primary'])]


    pri_rank = pd.read_csv(pri_rankpath, encoding = 'ISO-8859-1')\
              .loc[:, ['School', 'ranking', 'State Overall Score']]\
              .rename(columns={"School": "School_Name"})

    # remove suburbs and postcode after school names
    pri_rank['School_Name'] = pri_rank['School_Name'].apply(lambda x: ''.join(re.findall('^[\w\d ]+', x))).str.lower()

    # match the school ranking by names
    # assign median value 80 to missing data
    pri_ranked = all_pri.merge(pri_rank, how='left', on='School_Name')\
                  .sort_values(by = 'ranking')\
                  .reset_index(drop=True)\
                  .replace(np.nan, 80)

    # calculate the weighted score for primary schools
    weigh_score_pri = []
    for i in range(len(pri_ranked)):
        score = 1-(100-pri_ranked.loc[i]['State Overall Score'])/(100-60)
        weigh_score_pri.append(float(score))

    pri_ranked['weighted_score'] = weigh_score_pri
    df = pri_ranked.loc[:, ['School_Name', 'LGA_Name', 'weighted_score']]
    df.to_csv("primary_school_weighscore.csv", index=False)

# =========================================================================

def sum_score():
    high_score = pd.read_csv('data/high_school_weighscore.csv', encoding = 'ISO-8859-1')
    pri_score = pd.read_csv('data/primary_school_weighscore.csv', encoding = 'ISO-8859-1')

    #change lga names to lower case and remove any brackets in them
    high_score['LGA_Name'] = high_score['LGA_Name']\
                            .apply(lambda x: re.sub('\(.*\)', '', x)\
                            .strip()\
                            .lower())
    pri_score['LGA_Name'] = pri_score['LGA_Name']\
                            .apply(lambda x: re.sub('\(.*\)', '', x)\
                            .strip()\
                            .lower())

    # group weighted scores by lga names and find the sum
    high_score = high_score.groupby(['LGA_Name'])\
                          .sum()\
                          .reset_index()
    pri_score = pri_score.groupby(['LGA_Name'])\
                        .sum()\
                        .reset_index()

    # merge two dataframes together
    df = high_score.merge(pri_score, how='outer', on='LGA_Name')
    df['total_score'] = df.sum(axis=1, numeric_only= True)
    df = df.rename(columns={'LGA_Name': 'LGA'})
    df.loc[:, ['LGA','total_score']]\
      .to_csv("schools in vic.csv", index=False)

    
# =======================================================

def final_merge():
    hp = pd.read_csv('data/lga-hp-2019.csv', encoding = 'ISO-8859-1')
    sh = pd.read_csv('data/schools in vic.csv', encoding = 'ISO-8859-1')
    cm = pd.read_csv('data/crime_data.csv')
    os = pd.read_csv("data/open_space.csv")
    ht = pd.read_csv('data/health.csv')
    
    hp['LGA'] = hp['LGA'].str.lower()
    sh['LGA'] = sh['LGA'].str.lower()
    cm['LGA'] = cm['LGA'].str.lower()
    os['LGA'] = os['LGA'].str.lower()
    ht['LGA'] = ht['LGA'].str.lower()

    sh_hp = hp.merge(sh, how='outer', on='LGA')\
            .reset_index(drop=True)
    sh_hp_cm = sh_hp.merge(cm, how='outer', on='LGA')\
            .reset_index(drop=True)
    sh_hp_cm_os = sh_hp_cm.merge(os, how='outer', on='LGA')\
            .reset_index(drop=True)
    sh_hp_cm_os_ht = sh_hp_cm_os.merge(ht, how='outer', on='LGA')\
            .reset_index(drop=True)

    final = sh_hp_cm_os_ht.loc[:, ['LGA', 'Mean House Price', 'total_score'\
                                   , 'Rate per 100,000 population', 'Total',\
                                   'total_medical_person_per_100000']]
    final.to_csv('data/final.csv')
