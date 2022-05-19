# IMPORTS
import streamlit as st
import pydeck as pdk
import sqlalchemy as db
import pandas as pd
from wordcloud import (WordCloud, get_single_color_func)
from sqlalchemy.sql import text
import math
import geopy.distance
import requests
import altair as alt
#import geocoder




def filter_by_distance(data):
    st.subheader('Filter by distance')
    # print('types:')
    # print(type(data.loc[0, 'distance_to_user']))
    # data['distance_to_user'] = data['distance_to_user'].astype(float)
    # print(type(data.loc[0, 'distance_to_user']))
    distanceRange = st.slider('Select a distance rangee', 0.0, data["distance_to_user"].max(), (0.0, data["distance_to_user"].max()))
    return data[(data["distance_to_user"] >= distanceRange[0]) & (data["distance_to_user"] <= distanceRange[1])]



def printMap(data):
    st.subheader('Map of listings')
    st.map(data[["lon", "lat"]])


def geo_coder(address):
    """
    geo code location
    input: address string
    output longitude and latitude
    !!max number of requests is limited
    """
    url = 'http://open.mapquestapi.com/geocoding/v1/address'
    params = {'key': 'XHd8D8B7BHYI9MPhOYmJ36YAlhNegD4J', 'location': address}
    r = requests.get(url, params=params)
    r2 = r.json()["results"][0]["locations"][0]['latLng']
    return r2['lng'], r2['lat']
    
    
def calculate_distance(lat1, lon1, lat2, lon2):
#    print(f"{lat1}, {lon1}")
    coords1 = (lat1, lon1)
    coords2 = (lat2, lon2)
    return geopy.distance.distance(coords1, coords2).km


def add_user_location():
    long = 0
    lat = 0
    location = ""
    location = st.text_input('My location (city, country):')
    st.write('The location is ', location)
    if location != "":
        long, lat = geo_coder(location)
    return location, long, lat
    


@st.cache #(allow_output_mutation=True)
def load_data_word_count():
    engine = db.create_engine('postgresql+psycopg2://ds21m031:surf1234@dsc-inf.postgres.database.azure.com/nyt_import')
 #   connection = engine.connect()
    metadata = db.MetaData(bind=engine)
    db.MetaData.reflect(metadata)
    WORDCOUNT = metadata.tables['ds21_b1_jobs_wordcount']
    query = db.select([
        WORDCOUNT.c.word,
        WORDCOUNT.c.count,
        WORDCOUNT.c.development
    ])
    return engine.execute(query).fetchall()


@st.cache #(allow_output_mutation=True)
def load_data_coeficient():
    engine = db.create_engine('postgresql+psycopg2://ds21m031:surf1234@dsc-inf.postgres.database.azure.com/nyt_import')
    metadata = db.MetaData(bind=engine)
    db.MetaData.reflect(metadata)
    RESULT = metadata.tables['ds21_b1_jobs_result']
    query = db.select([
        RESULT.c.lm_coefficint
    ])
    return engine.execute(query).fetchall()


@st.cache
def load_data_location():
    engine = db.create_engine('postgresql+psycopg2://ds21m031:surf1234@dsc-inf.postgres.database.azure.com/nyt_import')
    connection = engine.connect()

    sql_query = text("""SELECT longitude, latitude, lm_coefficint from ds21_b1_jobs_result;""")
    return connection.execute(sql_query).fetchall()

def do_wordCloud(df_coefficint, df_wordcount):
    # engine = db.create_engine('postgresql+psycopg2://ds21m031:surf1234@dsc-inf.postgres.database.azure.com/nyt_import')
    # metadata = db.MetaData(bind=engine)
    # db.MetaData.reflect(metadata)
    # WORDCOUNT = metadata.tables['ds21_b1_jobs_wordcount']
    # RESULT = metadata.tables['ds21_b1_jobs_result']

    # query = db.select([
    #    WORDCOUNT.c.word,
    #    WORDCOUNT.c.count,
    #    WORDCOUNT.c.development
    # ])
    # result_wordcount = engine.execute(query).fetchall()
    
    # Coef query
    # query = db.select([
    #    RESULT.c.lm_coefficint
    # ])
    # result_coefficint = engine.execute(query).fetchall()
    
    # df_coefficint = pd.DataFrame(result_coefficint)
    df_coefficint.columns = ['coef']
    df_coefficint = df_coefficint.assign(
        coef=lambda dataframe: dataframe['coef'].map(lambda coef: 'increase' if coef > 0 else (
            'decrease' if not math.isnan(coef) else 'noMatch'))
    )
    dev_occurrences = df_coefficint['coef'].value_counts()
    
    # df_wordcount = pd.DataFrame(result_wordcount)
    df_wordcount.columns = ['word', 'count', 'dev']
    
    df_wordcount = df_wordcount.assign(
        factor=df_wordcount.apply(lambda row: row['count'] / dev_occurrences[row['dev']], axis=1)
    )

    df_wordcount_less = df_wordcount.loc[df_wordcount['dev'] != 'noMatch'].sort_values('factor', ascending=False)
    df_wordcount_factor = df_wordcount.loc[df_wordcount['dev'] != 'noMatch'].sort_values('factor', ascending=False)

    df_wordcount_sum = df_wordcount.drop(['dev', 'factor'], axis=1)
    df_wordcount_sum = df_wordcount_sum.groupby('word').sum('count')
    df_wordcount_sum = df_wordcount_sum.sort_values('count', ascending=False)

    df_wordcount_less = df_wordcount_less.assign(
        count=df_wordcount_less.apply(lambda row: df_wordcount_sum.loc[row['word'],], axis=1)
    )

    df_wordcount_size = df_wordcount_less.drop(['dev', 'factor'], axis=1)
    df_wordcount_size = df_wordcount_size.drop_duplicates()

    df_wordcount_size.set_index('word', inplace=True)
    dt_wordcount_size = df_wordcount_size['count'].to_dict()

    wc = WordCloud(background_color='white', width=1000, height=1000, margin=2)
    wc.fit_words(dt_wordcount_size)

    df_wordcount_factor = df_wordcount_factor.loc[
        df_wordcount_factor.groupby('word')['factor'].nlargest(1).index.get_level_values(1)]

    color_to_words = {
        'green': df_wordcount_factor.loc[df_wordcount_factor['dev'] == 'increase', 'word'].tolist(),
        'red': df_wordcount_factor.loc[df_wordcount_factor['dev'] == 'decrease', 'word'].tolist()
    }

    default_color = 'grey'

    grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

    wc.recolor(color_func=grouped_color_func)

    st.image(wc.to_array())


def printWordCount(data):
    max_words = 50
    figsize = (14, 8)
    width = 2400
    height = 1300
    wc = WordCloud(background_color='white', max_words=max_words, width=width, height=height).fit_words(data)
    st.image(wc.to_array())


def filter_by_development(data):
    st.subheader('Filter by development')
    devSelect = st.multiselect("Development", data["development"].unique(), data["development"].unique() )
    data = data[data["development"].isin(devSelect)]
    st.write('Development types:', devSelect)
    return data


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


# GEOLOCATION
def filtered_map(df_location):
    st.subheader('Map of locations')

    # select
    select_dev = st.selectbox(
        'Select development:',
        ('all', 'increase', 'decrease'))

    # engine = db.create_engine('postgresql+psycopg2://ds21m031:surf1234@dsc-inf.postgres.database.azure.com/nyt_import')
    # connection = engine.connect()

    # sql_query = text("""SELECT longitude, latitude, lm_coefficint from ds21_b1_jobs_result;""")
    # result_location = connection.execute(sql_query).fetchall()

    # df_location = pd.DataFrame(result_location)
    df_location.columns = ['longitude', 'latitude', 'coef']
    df_location['coef'] = df_location['coef'].fillna(0)

    if select_dev == 'increase':
        df_location = df_location.loc[df_location['coef'] > 0]
    elif select_dev == 'decrease':
        df_location = df_location.loc[df_location['coef'] < 0]

    df_location = df_location.assign(
        color=lambda dataframe: dataframe['coef'].map(
            lambda coef: [0, 153, 0] if coef > 0 else ([153, 0, 0] if coef < 0 else [0, 128, 255]))
    )

    layer = pdk.Layer(
        type='ScatterplotLayer',
        data=df_location,
        get_position=['longitude', 'latitude']
    )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        layers=[pdk.Layer(
            type='ScatterplotLayer',
            data=df_location,
            pickable=True,
            opacity=1,
            stroked=True,
            filled=True,
            radius_scale=100,
            line_width_min_pixels=1,
            get_position=['longitude', 'latitude'],
            get_radius=40,
            get_fill_color='color',
            get_line_color=[0, 0, 0])
        ]
    ))


@st.cache(allow_output_mutation=True)
def load_data_jobs():
    engine = db.create_engine('postgresql+psycopg2://ds21m031:surf1234@dsc-inf.postgres.database.azure.com/nyt_import')
    connection = engine.connect()
    sql_query = text("""SELECT title, city, country, longitude, latitude, lm_coefficint, company_name_ad from ds21_b1_jobs_result;""")
    result_location = connection.execute(sql_query).fetchall()
    return pd.DataFrame(result_location)


def preprocess_data(data):
    data.columns = ['title', 'city', 'country',  'longitude', 'latitude', 'coef', 'company']
    data['coef'] = data['coef'].fillna(0)

    data = data.assign(
        color=lambda dataframe: dataframe['coef'].map(lambda coef: 'g' if coef > 0 else ('r' if coef < 0 else 'y'))
        )
    return data


def printMap(data, title):
    st.map(data)
    
    
def filter_by_country(data):
    st.subheader('Filter by country')
    countrySelect = st.multiselect("Country", data["country"].unique() )
    data = data[data["country"].isin(countrySelect)]
    st.write('Property types:', countrySelect)
    return data



def fancy_map(data):
    st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=48,
         longitude=16,
         zoom=5,
         pitch=50,
     ),
     layers=[
         pdk.Layer(
            'HexagonLayer',
            data=data,
            get_position='[longitude, latitude]',
            radius=20000,
            elevation_scale=400,
            elevation_range=[100, 1000],
            pickable=True,
            extruded=True,
         ),
         pdk.Layer(
             'ScatterplotLayer',
             data=data,
             get_position='[longitude, latitude]',
             get_color='[200, 30, 0, 160]',
             get_radius=2000,
         ),
     ],
 ))
    
    
def fancy_map2(data):
    st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=48,
        longitude=16,
        zoom=5,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=data[data['coef'] == 0],
            get_position='[longitude, latitude]',
            get_color='[246, 208, 64]',
            get_radius=20000,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=data[data['coef'] < 0],
            get_position='[longitude, latitude]',
            get_color='[230, 34, 59]',
            get_radius=20000,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=data[data['coef'] > 0],
            get_position='[longitude, latitude]',
            get_color='[0, 102, 72]',
            get_radius=20000,
        )
    ],
))

#################################################
#                 VISUALISATION                 #
#################################################

print("start")

data_jobs = load_data_jobs()    
data_word_count = load_data_word_count()
df_location_2 = preprocess_data(data_jobs)
df_location = pd.DataFrame(load_data_location())
df_coefficint = pd.DataFrame(load_data_coeficient())
df_wordcount = pd.DataFrame(data_word_count, columns=["word", "count", "development"])
# df_coefficint.columns = ['coef']


#print(df_location_2.head())


st.title('Jobs & Stocks - Group Project')


st.header('Popular keywords')
do_wordCloud(df_coefficint, df_wordcount)
# printWordCount(df_wc_temp[["word", "count"]].set_index("word").to_dict()["count"])
df_wordcount = pd.DataFrame(data_word_count, columns=["word", "count", "development"])
df_wc_temp = filter_by_development(df_wordcount)
df_wc_temp = df_wc_temp[["word", "count"]].groupby('word').sum().nlargest(20, 'count')
df_wc_temp.reset_index(inplace=True)
st.write(alt.Chart(df_wc_temp).mark_bar().encode(x=alt.X('word', sort=None), y='count', ))


st.header('Location of jobs')

filtered_map(df_location)

# printMap(df_location[["longitude", "latitude"]], "Map of locations")


st.header('Filter jobs by distance')


location, long, lat = add_user_location()

# location = "Eisenstadt, Austria"
# long = 16.572504
# lat = 48.008354

if location != "":
    print(df_location_2)
    df_location_2["distance_to_user"] = df_location_2.apply(lambda x: calculate_distance(x[4], x[3], lat, long), axis=1)
    df_location_2.at[df_location_2["country"] == "Anywhere", "distance_to_user"] = 0
    df_location_2 = df_location_2[df_location_2["distance_to_user"] < 1000]
    data_temp = filter_by_distance(df_location_2)
    r = st.write('Data sets selected:', len(data_temp))
    printMap(data_temp[["longitude", "latitude"]], "Filtered map of locations")
    
    st.header('Fancy map 1')
    fancy_map(data_temp[["longitude", "latitude", "coef"]])
    st.header('Fancy map 2')
    fancy_map2(data_temp[["longitude", "latitude", "coef"]])

    
    st.write(data_temp[["title", "company", "city", "country", "distance_to_user", "coef"]])



print("finished")
