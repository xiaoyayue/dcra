#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:19:28 2019

@author: andreaquevedo and lilyyue
"""
import pandas as pd
import numpy as np
import requests
import time
from sklearn.preprocessing import LabelEncoder


#Importing public data:
chart_school = pd.read_csv('https://opendata.arcgis.com/datasets/a3832935b1d644e48c887e6ec5a65fcd_1.csv', encoding='iso-8859-1')
public_school=pd.read_csv('https://opendata.arcgis.com/datasets/4ac321b2d409438ebd76a6569ad94034_5.csv', encoding='iso-8859-1')
metro_station= pd.read_csv('https://opendata.arcgis.com/datasets/556208361a1d42c68727401386edf707_111.csv', encoding='iso-8859-1')
metro_bus= pd.read_csv('https://opendata.arcgis.com/datasets/e85b5321a5a84ff9af56fd614dab81b3_53.csv', encoding='iso-8859-1')
grocery=pd.read_csv('https://opendata.arcgis.com/datasets/1d7c9d0e3aac49c1aa88d377a3bae430_4.csv',encoding='iso-8859-1')
community_garden=pd.read_csv('https://opendata.arcgis.com/datasets/a82537b01c2141558ba5e9e13224d395_4.csv',encoding='iso-8859-1')
certified_business_ent=pd.read_csv('https://opendata.arcgis.com/datasets/3be48fcbb3b04afc9a0ee33177bfbcf0_33.csv',encoding='iso-8859-1')

#Removing unnecessary variables for analysis and renaming variables
chart_school = chart_school[['SCHOOL_NAM', 'ADDRESS', 'GRADES', 'SCHOOL_YEA', 'LATITUDE', 'LONGITUDE', 'WARD', 'ZIPCODE']]
public_school= public_school[['NAME','ADDRESS','LONGITUDE','LATITUDE','FACUSE','STATUS','MAR_ID','ZIPCODE']]
metro_station = metro_station[['NAME','ADDRESS', 'X', 'Y', 'LINE']]
metro_station.rename(columns={'X': 'LONGITUDE', 'Y': 'LATITUDE'}, inplace=True)
metro_bus = metro_bus[['BSTP_LAT','BSTP_LON', 'AT_STR', 'ON_STR', 'BSTP_HDG', 'BSTP_MSG_TEXT']]
metro_bus.rename(columns={'BSTP_LAT': 'LATITUDE', 'BSTP_LON': 'LONGITUDE'}, inplace=True)
grocery=grocery[['STORENAME','ADDRESS', 'X','Y','WARD','SSL','ZIPCODE','PRESENT16','PRESENT17','PRESENT18']]
grocery.rename(columns={'X': 'LONGITUDE', 'Y': 'LATITUDE'}, inplace=True)
community_garden= community_garden[['NAME','ADDRESS','ACRES','MAR_ID']]
certified_business_ent= certified_business_ent[['BUSINESSNAME', 'WARD', 'ADDRESS','MAR_ID','LATITUDE','LONGITUDE','CATEGORIES']]


#Checking for NAs
chart_school.isna().sum()
public_school.isna().sum()
metro_station.isna().sum()
metro_bus.isna().sum()
grocery.isna().sum()
community_garden.isna().sum()
certified_business_ent.isna().sum()

####################################################################
#########Retrieving geo data for school, metro_station #############
#########         gardens, and businesses              #############
####################################################################

#Citizenatlas API
#reviewing the citizenatlas page
host = 'http://citizenatlas.dc.gov'
url = '/newwebservices/locationverifier.asmx/findLocation2'
headers = {'Content-Type': 'application/x-www-form-urlencoded',
           'Content-Length': 'length'}

# access, parse, and transform
def get_address_info(addr_str):
    '''sends a post request to MAR API to retrieve address information
       accepts str, returns dict'''
    data = {'str': addr_str, 'f': 'json'}
    r = requests.post(url=host+url, data=data, headers=headers)
    parsed = r.json()
    info = parsed['returnDataset']['Table1'][0]
    return info


####Getting geo data for charter school dataset#####

#converting dataframe to list of dictionaries
chart_school_dict = chart_school.to_dict(orient = 'records')
chart_school_dict[:2]

#adding info to dictionaries
chart_school_info = []
for row in chart_school_dict:
    try:
        info = get_address_info(row['ADDRESS'])
        chart_school_info.append({**row, **info})
        time.sleep(0.25)
    except (KeyError, TypeError) as e:
        continue
chart_school_info[:1]

####Getting geo data for public school dataset#####

#converting dataframe to list of dictionaries
public_school_dict = public_school.to_dict(orient = 'records')
public_school_dict[:2]

#adding info to dictionaries
public_school_info = []
for row in public_school_dict:
    try:
        info = get_address_info(row['ADDRESS'])
        public_school_info.append({**row, **info})
        time.sleep(0.25)
    except (KeyError, TypeError) as e:
        continue
public_school_info[:1]

####Getting geo data for community garden dataset#####

#converting dataframe to list of dictionaries
community_garden_dict = community_garden.to_dict(orient = 'records')
community_garden_dict[:2]

#adding info to dictionaries
community_garden_info = []
for row in community_garden_dict:
    try:
        info = get_address_info(row['ADDRESS'])
        community_garden_info.append({**row, **info})
        time.sleep(0.25)
    except (KeyError, TypeError) as e:
        continue
community_garden_info[:1]

#####Getting geo data for metro station dataset#####

#converting dataframe to list of dictionaries
metro_station_dict = metro_station.to_dict(orient = 'records')
metro_station_dict[:2]

#adding info to dictionaries
metro_station_info = []
for row in metro_station_dict:
    try:
        info = get_address_info(row['ADDRESS'])
        metro_station_info.append({**row, **info})
        time.sleep(0.25)
    except (KeyError, TypeError) as e:
        continue
metro_station_info[:1]

#####Getting geo data for business dataset#####

#converting dataframe to list of dictionaries
certified_business_ent_dict = certified_business_ent.to_dict(orient = 'records')
certified_business_ent_dict[:2]

#adding info to dictionaries
certified_business_ent_info = []
for row in certified_business_ent_dict:
    try:
        info = get_address_info(row['ADDRESS'])
        certified_business_ent_info.append({**row, **info})
        time.sleep(0.25)
    except (KeyError, TypeError) as e:
        continue
certified_business_ent_info[:1]

#convert nested dicts to dataframes
chart_school_geo = pd.DataFrame(chart_school_info)
public_school_geo = pd.DataFrame(public_school_info)
metro_station_geo= pd.DataFrame(metro_station_info)
community_garden_geo=pd.DataFrame(community_garden_info)
certified_business_ent_geo=pd.DataFrame(certified_business_ent_info)

#keeping only variables of interest
chart_school_geo = chart_school_geo[['SCHOOL_NAM', 'ADDRESS', 'GRADES', 'SCHOOL_YEA', 'LATITUDE', 'LONGITUDE', 'MARID', 'SSL', 'WARD', 'ZIPCODE']]
public_school_geo= public_school_geo[['NAME','ADDRESS', 'FACUSE', 'STATUS', 'LATITUDE', 'LONGITUDE', 'MARID', 'SSL','WARD','ZIPCODE']]
metro_station_geo= metro_station_geo[['NAME','ADDRESS','LINE', 'LATITUDE', 'LONGITUDE', 'MARID', 'SSL', 'WARD','ZIPCODE']]
community_garden_geo=community_garden_geo[['NAME','ADDRESS','ACRES','LATITUDE', 'LONGITUDE', 'SSL','MARID','WARD','ZIPCODE']]
certified_business_ent_geo=certified_business_ent_geo[['BUSINESSNAME','ADDRESS','CATEGORIES','STATUS','LATITUDE','LONGITUDE','MARID', 'SSL', 'WARD','ZIPCODE']]

#checking for missing values
chart_school_geo.isna().sum()
metro_station_geo.isna().sum()
public_school_geo.isna().sum()
community_garden_geo.isna().sum()
certified_business_ent_geo.isna().sum()

#removing NAs from community_garden dataset
community_garden_geo=community_garden_geo.dropna()

#save as csv
chart_school_geo.to_csv('../../Data/chart_school_geo.csv')
public_school_geo.to_csv('../../public_school_geo.csv')
metro_station_geo.to_csv('../../Data/metro_station_geo.csv')
community_garden_geo.to_csv('../../community_garden_geo.csv')
certified_business_ent_geo.to_csv('../../certified_business_ent_geo.csv')

####################################################################
###Retrieving geo data for metro_bus dataset #######################
####################################################################

#Different url from the previous query since the metro_bus data set did not have
#address information, only latitude and longitude of stops

#Citizenatlas API
host = 'http://citizenatlas.dc.gov'
url = '/newwebservices/locationverifier.asmx/reverseLatLngGeocoding2'
headers = {'Content-Type': 'application/x-www-form-urlencoded',
           'Content-Length': 'length'}

# access, parse, and transform
def get_address_info1(lati,longi):
    '''sends a post request to MAR API to retrieve address information
       accepts str, returns dict'''
    data = {'lat': lati,'lng': longi ,'f': 'json'}
    r = requests.post(url=host+url, data=data, headers=headers)
    parsed = r.json()
    info = parsed['Table1'][0]
    wanted_keys = ['FULLADDRESS','MARID', 'SSL', 'LATITUDE', 'LONGITUDE', 'WARD', 'ZIPCODE']
    info_sub = dict((k, info[k]) for k in wanted_keys if k in info)
    return info_sub

#testing one set of latitude and longitude
print(get_address_info1('38.912752','-77.017827'))

####Getting geo data for metro bus dataset#####

#converting dataframe to list of dictionaries
metro_bus_dict = metro_bus.to_dict(orient = 'records')
metro_bus_dict[:2]

#adding info to dictionaries
#note: the code takes around 2 hours to run.
#to test the code , run for a subset of rows, metro_bus_dict[:10], for example

metro_bus_info = []
for row in metro_bus_dict:
    try:
        info = get_address_info1(row['LATITUDE'],row['LONGITUDE'])
        metro_bus_info.append({**row, **info})
        time.sleep(0.25)
    except (KeyError, TypeError) as e:
        continue

metro_bus_info[:1]

#convert nested dicts
metro_bus_geo = pd.DataFrame(metro_bus_info)

#checking for missing values
metro_bus_geo.isna().sum()

#save as csv
metro_bus_geo.to_csv('../../Data/metro_bus_geo.csv')

####################################################################
################## Cleaning grocery data ###########################
####################################################################

le= LabelEncoder()
col_names= ['PRESENT16', 'PRESENT17','PRESENT18']

for var in col_names:
    grocery[var]= le.fit_transform(grocery[var])

#dropping supermarkets not present throughout our years of interest
grocery['PRESENT18'] = grocery['PRESENT18'].replace({0:np.nan})
grocery.isna().sum()
grocery=grocery.dropna()

#keeping only variables of interest
grocery=grocery[['STORENAME','ADDRESS', 'LONGITUDE','LATITUDE','WARD','SSL','ZIPCODE']]

#save as csv
grocery.to_csv('../../Data/grocery_clean.csv')
