# Databricks notebook source
#Reading Philadelphia dataset
# File location and type
file_location = "/FileStore/tables/Philadelphia_dataset.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

file = df.toPandas()

# COMMAND ----------

file.head()

# COMMAND ----------

pip install folium

# COMMAND ----------

pip install geopy

# COMMAND ----------

import folium
from branca.element import Figure
from geopy.geocoders import Nominatim

fig_size = Figure(width = 500, height = 500)
geolocator = Nominatim(user_agent="geoapiExercises")

latitude = [39.88388114, 39.98199552, 40.03120371, 39.92328314, 40.0021239, 39.98254529, 40.0287841, 39.97876149, 40.02518448, 39.98453229, 39.87883267, 39.9540352999999, 40.08244258, 39.99379565]

longitude  = [-75.23070645, -75.12379868, -75.15409118, -75.16265119, -75.14409686, -75.12473953, -75.05731383, -75.16960489, -75.10319215, -75.15700603, -75.24313797, -75.16244935, -74.96471021, -75.11168277]       

draw_map = folium.Map(width=500,height=500, location  = [39.96233372, -75.16144594], zoom_start = 11, min_zoom = 7, max_zoom = 18)

for i in latitude:
  for j in longitude:
    location = geolocator.reverse(str(i)+","+str(j))
    address = location.raw['address']
    city = address.get('city','')
    folium.Marker(location  = [i,j], popup = city, tooltip = 'Click here to know the city').add_to(draw_map)

draw_map

# COMMAND ----------

import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

plt.rc('font', size = 12)
figure, axis = plt.subplots(figsize = (10,6))

axis.plot(file._c8, color = 'magenta', label = 'Crimes')
axis.set_xlabel('Count')
axis.set_title("Time Series Plot of Crimes in Philadelphia")
axis.grid(True)
axis.legend(loc = 'upper left')

# COMMAND ----------

crime_list = file['_c16'].tolist()
crime_list = [x for x in crime_list[1:] if str(x)!='nan']
crime_list

# COMMAND ----------

crime_count = len(crime_list)
temp = set(crime_list)
unique_crime_list = list(temp)
unique_crime_count = len(unique_crime_list)
print("Before : "+str(crime_count))
print("After : "+str(unique_crime_count))
print("CRIMES IN PHILADELPHIA :")
unique_crime_list

# COMMAND ----------

pie_chart_dict = {}
for i in unique_crime_list:
  pie_chart_dict[i] = 0

pie_chart_dict

# COMMAND ----------

for x in crime_list:
  pie_chart_dict[x]+=1 
pie_chart_dict

# COMMAND ----------

from operator import itemgetter
for x in crime_list:
  pie_chart_dict[x]+=1 

N = 10
sorted_pie_chart_dict = dict(sorted(pie_chart_dict.items(), key =                                          itemgetter(1), reverse = True)[:N])
print("The top ten crimes are : ")
sorted_pie_chart_dict

# COMMAND ----------

import matplotlib.pyplot as plt

pie_labels = 'All Other Offenses','Thefts', 'Other Assaults', 'Vandalism/Criminal Mischief', 'Theft from Vehicle', 'Fraud', 'Narcotic / Drug Law Violations', 'Burglary Residential', 'Aggravated Assault No Firearm', 'Robbery No Firearm'
values = [756380, 641260, 595584, 370680, 368636, 284884, 222740, 154416, 150456, 95288]

figure, axes = plt.subplots()
axes.pie(values, labels = pie_labels, autopct = '%.2f', startangle = 90)
axes.axis('equal')
plt.title('Top 10 crimes in Philadelphia')

plt.show()

# COMMAND ----------

pip install wordcloud

# COMMAND ----------

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

words = ''
stop = set(STOPWORDS)
for i in unique_crime_list:
  i = str(i)
  tokens = i.split()
  for j in range(len(tokens)):
    tokens[j] = tokens[j].lower()
  words+=" ".join(tokens)+" "

wordcloud = WordCloud(width = 500, height = 500, background_color = 'grey',
                      stopwords = stop, min_font_size = 10).generate(words)

plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()

# COMMAND ----------

new_header = file.iloc[0]
file = file[1:]
file.columns = new_header
file

# COMMAND ----------

import pandas as pd

new_header = file.iloc[0]
file = file[1:]
file.columns = new_header
file['Date'] = pd.to_datetime(file['Date'])
file['DayOfWeek'] = file['Date'].dt.day_name()
file

# COMMAND ----------

!pip install plotly

# COMMAND ----------

values = []
labels = file.DayOfWeek.unique()

for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='1':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values.append(count)

values1=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='2':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values1.append(count)

values2=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='3':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values2.append(count)

values3=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='4':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values3.append(count)

values4=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='5':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values4.append(count)

values5=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='6':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values5.append(count)

values6=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='7':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values6.append(count)

values7=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='8':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values7.append(count)

values8=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='9':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values8.append(count)

values9=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='10':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values9.append(count)

values10=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='11':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values10.append(count)

values11=[]
for each in labels:
  count=0
  for i in range(0,21609):
    list1=file.iloc[i]
    if list1['Month']=='12':#(1-12 values)
      if list1['DayOfWeek']==each:
        count=count+1
  values11.append(count)

# COMMAND ----------

import plotly.graph_objects as go

fig=go.Figure()
fig.add_trace(go.Box(y=values,name="Jan"))
fig.add_trace(go.Box(y=values1,name="Feb"))
fig.add_trace(go.Box(y=values2,name="March"))
fig.add_trace(go.Box(y=values3,name="April"))
fig.add_trace(go.Box(y=values4,name="May"))
fig.add_trace(go.Box(y=values5,name="June"))
fig.add_trace(go.Box(y=values6,name="July"))
fig.add_trace(go.Box(y=values7,name="Aug"))
fig.add_trace(go.Box(y=values8,name="Sept"))
fig.add_trace(go.Box(y=values9,name="Oct"))
fig.add_trace(go.Box(y=values10,name="Nov"))
fig.add_trace(go.Box(y=values11,name="Dec"))
fig.update_traces(quartilemethod="inclusive")
fig.show()

# COMMAND ----------

import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

decom_freq = 24*60/15*365
result = sm.tsa.seasonal_decompose(file.Year.interpolate(),
                                  period = int(decom_freq),
                                  model = 'additive')

decom_plot = result.plot()
plt.show()

# COMMAND ----------

date_count = len(date_list)
temp = set(date_list)
unique_date_list = list(temp)
unique_date_count = len(unique_date_list)
print("Before : "+str(date_count))
print("After : "+str(unique_date_count))

# COMMAND ----------

pie_chart_dict2 = {}
for i in unique_date_list:
  pie_chart_dict2[i] = 0
pie_chart_dict2

# COMMAND ----------

from operator import itemgetter

for x in date_list:
  pie_chart_dict2[x]+=1 

N = 10
sorted_pie_chart_dict2 = dict(sorted(pie_chart_dict2.items(), key =                                          itemgetter(1), reverse = False)[:N])
print("Dates with the least number of crimes: ")
sorted_pie_chart_dict2

# COMMAND ----------

from operator import itemgetter

N = 10
reverse_sorted_pie_chart_dict2 = dict(sorted(pie_chart_dict2.items(), key                                      = itemgetter(1), reverse = True)                                          [:N])
print("Dates with the most number of crimes: ")
reverse_sorted_pie_chart_dict2

# COMMAND ----------

def Merge(dict1, dict2):
  result = {**dict1, **dict2}
  return result

final_dict = Merge(sorted_pie_chart_dict2, reverse_sorted_pie_chart_dict2)
print(final_dict)

# COMMAND ----------

def Merge(dict1, dict2):
  result = {**dict1, **dict2}
  return result

final_dict = Merge(sorted_pie_chart_dict2, reverse_sorted_pie_chart_dict2)
print(final_dict)for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(pie_labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Top-10 Dates with Most and Least Incidents")

plt.show()

# COMMAND ----------

new_header = file.iloc[0]
file = file[1:]
file.columns = new_header
file

# COMMAND ----------

import pandas as pd

cate2 = pd.DataFrame({"Hour":file.Hour.unique()})
labels2 = file.Hour.unique()
values2=[]
for each in labels2:
    values2.append(len(file[file.Hour==each]))
val2=pd.DataFrame({"Count":values2})
result2=pd.concat([cate2,val2],axis=1)

from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px

indices = find_peaks(result2['Count'], threshold=50000)[0]
fig = px.line(result2,x='Hour',y='Count')
fig.add_trace(go.Scatter(
    x = result2['Hour'][indices],
    y = [result2['Count'][j] for j in indices],
    mode = 'markers',
    marker = dict(
        size = 8,
        color = 'red',
        symbol = 'cross'
    ),
    name = 'Detected Peaks'
))
fig.show()
