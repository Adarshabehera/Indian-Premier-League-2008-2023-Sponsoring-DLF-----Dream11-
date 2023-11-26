#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import os
for dirname, _, filenames in os.walk("C:/Users/ADARSHA KUMAR BEHERA/Downloads/archive (12)"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as exp
from plotly.subplots import make_subplots
from IPython.display import Image
import squarify
import warnings
warnings.filterwarnings("ignore")
from glob import glob
import os


# In[4]:


path_dir = 'C:/Users/ADARSHA KUMAR BEHERA/Downloads/archive (12)/IPL - Player Performance Dataset/Most Runs'
for files in os.listdir(path_dir):
    print(files)


# In[5]:


df1= pd.read_csv("C:/Users/ADARSHA KUMAR BEHERA/Downloads/archive (12)/IPL - Player Performance Dataset/Most Runs/Most Runs - 2021.csv",encoding = 'latin-1')
df1.head(5).style.background_gradient(cmap='Blues_r')


# In[6]:


df2= pd.read_csv("C:/Users/ADARSHA KUMAR BEHERA/Downloads/archive (12)/IPL - Player Performance Dataset/Most Wickets/Most Wickets - 2021.csv",encoding = 'latin-1')
df2.head(5).style.background_gradient(cmap='Blues_r')


# In[7]:


df3= pd.read_csv("C:/Users/ADARSHA KUMAR BEHERA/Downloads/archive (12)/IPL - Player Performance Dataset/Best Bowling Strike Rate Innings/Best Bowling Strike Rate Innings - 2021.csv",encoding = 'latin-1')
df3.head(5).style.background_gradient(cmap='Blues_r')


# ## Data Cleaning :-

# In[8]:


df1.shape


# In[9]:


df2.shape


# In[10]:


df3.shape


# In[11]:


df1.columns


# In[12]:


df2.columns


# In[13]:


df2.isna().sum()


# In[14]:


df1.isna().sum()


# In[15]:


df1.describe()


# In[16]:


df1[["SR", "NO"]]


# In[17]:


df1_most_runs = df1.drop('POS',1)


# In[18]:


df1_most_runs.head(2).style.background_gradient(cmap = 'Blues_r')


# In[19]:


df1.head(3)


# In[20]:


df2.head(3)


# In[21]:


df2_most_wickets = df2.drop("POS",1)


# In[22]:


df2_most_wickets.head(3).style.background_gradient(cmap = 'Blues_r')


# In[23]:


df2_most_wickets.columns


# In[24]:


df1_most_runs.columns


# In[25]:


df1_most_runs['HS']


# In[26]:


df1_most_runs_HS = df1_most_runs["HS"].str.replace('*','',regex = True)
df1_most_runs.HS = df1_most_runs_HS
df1_most_runs.HS = pd.to_numeric(df1_most_runs.HS, errors = 'coerce')


# In[27]:


df1_most_runs.head(4).style.background_gradient(cmap = 'Blues_r')


# In[28]:


df2_most_wickets.describe().style.background_gradient(cmap = 'Blues_r')


# In[29]:


df1_most_runs.describe().style.background_gradient(cmap = 'Blues_r')


# ### Maximim no of matches played by the Players in a season :-

# In[30]:


df1_most_runs["Mat"]


# In[31]:


df2_most_wickets["Mat"]


# In[32]:


if df1_most_runs["Mat"].max() > df2_most_wickets["Mat"].max():
    matches = df1_most_runs["Mat"].max()
else:
    matches = df2_most_wickets['Mat'].max()


# In[33]:


matches


# In[34]:


print("Max matches played by a player in a single season :", matches)


# ### Correlation between the Players:-

# In[35]:


df1_most_runs.corr().style.background_gradient(cmap = "Blues_r")


# In[36]:


df2_most_wickets.corr().style.background_gradient(cmap = "Blues_r")


# In[37]:


import plotly.express as px
px.data.medals_wide(indexed = True)
fig = px.imshow(df1_most_runs.corr(),color_continuous_scale = "Ice_r")
fig.show()


# In[38]:


import plotly.express as px
px.data.medals_wide(indexed = True)
fig = px.imshow(df2_most_wickets.corr(),color_continuous_scale = "Ice_r")
fig.show()


# ## Basic data preprocessing for the batsmen Arena:-

# In[39]:


df1_most_runs.columns


# In[40]:


df1_most_runs[(df1_most_runs["Avg"] >= 40) & (df1_most_runs["BF"] > 120)].style.background_gradient(cmap = 'Blues_r')


# In[41]:


df1_most_runs[(df1_most_runs["Avg"] >= 40) & (df1_most_runs["BF"] > 120)]["Player"]


# In[42]:


df1_most_runs[(df1_most_runs["Runs"] > 250) & (df1_most_runs["SR"] >= 100) & (df1_most_runs["Avg"] > 45)].style.background_gradient(cmap = "Blues_r")


# In[43]:


df1_most_runs[(df1_most_runs)["50"]>= 4].style.background_gradient(cmap ="Blues_r")


# In[44]:


df1_most_runs[(df1_most_runs["4s"] > 50) & (df1_most_runs["6s"] > 5)]


# ### No of Matches played by the player in each season :-

# In[45]:


import plotly.express as px
fig = px.histogram(df1_most_runs, x="Mat",color="Mat")
fig.show()


# ### Some Bowling Visuals Part :-

# In[46]:


df2_most_wickets.head(3)


# In[47]:


df2_most_wickets.columns


# In[48]:


df2_most_wickets[(df2_most_wickets["Wkts"] > 15)].style.background_gradient(cmap = "Blues_r")


# In[49]:


df2_most_wickets[(df2_most_wickets["5w"] > 0)].style.background_gradient(cmap = "Blues_r")


# In[50]:


import plotly.express as px
fig = px.histogram(df1_most_runs, x = "Avg", color = "Avg")
fig.show()


# In[51]:


df1_most_runs.head(4)


# In[52]:


import plotly.express as px
fig = px.histogram(df1_most_runs, x = "Inns", color = "Inns")
fig.show()


# In[53]:


fig = px.histogram(df1_most_runs, x = "6s", color = "6s")
fig.show()


# ### From above we concluded that most players have never sixes which, counted as 52 and finally there are very least players who 
# ### really hitted sixes as 30
# ### Last but not the least , maximally the batsman used to hit sixes in the range between 10-15 where 13 counted as the highest.

# In[54]:


df1_most_runs.columns


# In[55]:


df1_most_runs_scored_players = df1_most_runs.sort_values(by = ["Runs"], ascending = False)
plt.figure(figsize = (10,5))
sns.barplot(x = df1_most_runs["Runs"],y = df1_most_runs["Player"][:10], palette = "Blues_r")


# In[56]:


df2_most_wickets.columns


# In[57]:


df2_most_wickets.head(3)


# In[58]:


df2_most_wickettakers_recent = df2_most_wickets.sort_values(by = ["Wkts"], ascending =False)
plt.figure(figsize = (6,6))
sns.barplot(x = df2_most_wickets["Wkts"], y = df2_most_wickets["Player"][:5], palette = "Blues_r")


# ## Creating the Players Avg and the matches played by Them :-

# In[59]:


x = df1_most_runs["Avg"]
y = df1_most_runs["Mat"]
fig = px.scatter(df1_most_runs, x, y, color = df1_most_runs["Avg"],
                 size = df1_most_runs["Avg"], title = "Player_Avgwise_Matches_played")
fig.show()


# In[60]:


df1_most_runs.columns


# In[61]:


x = df1_most_runs["Inns"]
y = df1_most_runs["Mat"]
fig = px.scatter(df1_most_runs, x, y, color = df1_most_runs["Inns"],
                 size = df1_most_runs["Inns"], title = "Player_Inningswise_Matches_played")
fig.show()


# In[62]:


x = df1_most_runs["Inns"]
y = df1_most_runs["6s"]
fig = px.scatter(df1_most_runs, x, y, color = df1_most_runs["6s"],
                 size = df1_most_runs["6s"], title = "Player_played_with_maximum_no_of_Sixes")
fig.show()


# ### Playing maximum no of matches doesn't give us guarentee that he will hit max sixes but other least matches played player do
# ### as we have seen in the scatter plot, as 13 matches leading the 30 sixes followed by 17 inns with 12 maximums.

# ## Players Avg, Balls Faced & Strike Rates reports in the fields :- 

# In[63]:


plt.figure(figsize = (20,8))

plt.plot(df1_most_runs["Player"][:10],
        df1_most_runs["Avg"][:10],
        color = "#19388A")
    
plt.plot(df1_most_runs["Player"][:10],
        df1_most_runs["SR"][:10],
        color = "b")

plt.plot(df1_most_runs["Player"][:10],
        df1_most_runs["BF"][:10],
        color = "#4F91CD")

plt.legend(["Average","Strike rate","Balls Faced"], loc = 'upper right')
plt.grid()
plt.xticks(rotation = 90)
plt.show()


# ###  From above we have analysed that Avg is better for KL Rahul, Strike rate is best for the batsman Prithvi Shaw
# ## and finally maximum balls faced by the indian Batsman Shikhar Dhawan etc. 

# # Best Strike Rates :-

# In[66]:


import plotly.graph_objects as go

df1_most_runs_SR = df1_most_runs.loc[:, ["Player", "SR"]]
df1_most_runs_SR = df1_most_runs.sort_values(by = ['SR'], ascending = False)

df1_most_runs_sort_by_matches = df1_most_runs.sort_values(by = ['SR'], ascending = False)

fig = go.Figure(go.Bar(
            x=df1_most_runs_SR["SR"],
            y=df1_most_runs_SR["Player"][:10],
            orientation='h'))

fig.show()


# In[68]:


df1_most_runs1_SR.head(2)


# In[69]:


df1_most_runs.columns


# In[70]:


df1_most_runs.head(3)


# # Best Batting Average :-

# In[71]:


import plotly.graph_objects as go

df1_most_runs_SR = df1_most_runs.loc[:, ["Player", "Avg"]]
df1_most_runs_SR = df1_most_runs.sort_values(by = ['Avg'], ascending = False)

df1_most_runs_sort_by_matches = df1_most_runs.sort_values(by = ['Avg'], ascending = False)

fig = go.Figure(go.Bar(
            x=df1_most_runs_SR["Avg"],
            y=df1_most_runs_SR["Player"][:10],
            orientation='h'))

fig.show()


# # Highest Sixes by any Players :-

# In[72]:


import plotly.express as px

df1_most_runs_6s = df1_most_runs.loc[:,["Player","6s"]]
df1_most_runs_6s = df1_most_runs_6s.sort_values(by = ["6s"], ascending = False)

sns.barplot(x = df1_most_runs_6s["Player"][:10], y = df1_most_runs_6s["6s"][:10])
plt.xticks(rotation = 90)
plt.show()

fig = px.bar(x = df1_most_runs_6s["Player"][:10], y = df1_most_runs_6s["6s"][:10])
plt.xlabel("Players")
plt.ylabel("No of Sixes")
fig.show()


# In[73]:


df1_most_runs_6s


# ## Highest 4s by players:-

# In[74]:


import plotly.express as px

df1_most_runs_4s = df1_most_runs.loc[:,["Player","4s"]]
df1_most_runs_4s = df1_most_runs_4s.sort_values(by = ["4s"], ascending = False)

fig = px.bar(x = df1_most_runs_4s["Player"][:10], y = df1_most_runs_4s["4s"][:10])
fig.show()


# In[76]:


fig = px.scatter(df1_most_runs, x = "Mat", y = "Avg", color = "Runs")
fig.show()


# In[80]:


fig = px.scatter(df1_most_runs, x = "Mat", y = "50", color = "Runs")
fig.show()


# In[77]:


df1_most_runs.columns


# In[78]:


df1_most_runs.head(3)


# In[81]:


fig = px.scatter(df1_most_runs, x = "Player", y = "SR", color = "Runs")
fig.show()


# In[82]:


fig = px.scatter(df1_most_runs, x = "Player", y = "SR", color = "Avg")
fig.show()


# ###### Total no of 50's in this season :-

# In[90]:


import plotly.graph_objects as go

fig = go.Figure(data = [go.Pie(values = df1_most_runs['50'].value_counts(),pull = [0,0,0,0,0])])
fig.show()


# In[87]:


import plotly.graph_objects as go

fig = go.Figure(data = [go.Pie(values = df1_most_runs['Mat'].value_counts(),pull = [0,0,0,0,0,0.25])])
fig.show()


# ## from above we analysed, that very rare and experienced players are being given most of the matches as 16,15,14,13 as given
# ## 2-5 % each and most of the players represented least no of Chances as 1,5 matches presented by 8-10 %

# In[91]:


player = []
for i in df1_most_runs['Player'][:5]:
    player.append(i)
Color1="#72FFFF"
Color2='#00D7FF'
Color3="#0096FF"
Color4="#5800FF"
runs = []
for i in df1_most_runs['Runs'][:5]:
    runs.append(i)

fig = make_subplots(rows = 2, cols = 3, specs = [[{"type": "Indicator"},
                                                  {"type": "Indicator"},
                                                  {"type": "Indicator"}],
                                                 [{"type": "Indicator"},
                                                  {"type": "Indicator"},
                                                  {"type": "Indicator"}]])

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = runs[0],
                             title = {'text': player[0],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 1000],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 250], 'color': Color1},
                                       {'range': [250, 500], 'color': Color2},
                                       {'range': [500, 750], 'color': Color3},
                                       {'range': [750, 1000], 'color': Color4}]})), row = 1, col = 1)

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = runs[1],
                             title = {'text': player[1],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 1000],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 250], 'color': Color1},
                                       {'range': [250, 500], 'color': Color2},
                                       {'range': [500, 750], 'color': Color3}, 
                                       {'range': [750, 1000], 'color': Color4}]})), row = 1, col = 3)

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = runs[2],
                             title = {'text': player[2],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 1000],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 250], 'color': Color1},
                                       {'range': [250, 500], 'color': Color2},
                                       {'range': [500, 750], 'color': Color3}, 
                                       {'range': [750, 1000], 'color': Color4}]})), row = 2, col = 1)

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = runs[3],
                             title = {'text': player[3],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 1000],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 250], 'color': Color1},
                                       {'range': [250, 500], 'color': Color2},
                                       {'range': [500, 750], 'color': Color3},
                                       {'range': [750, 1000], 'color': Color4}]})), row = 2, col = 2)

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = runs[4],
                             title = {'text': player[4],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 1000],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 250], 'color': Color1},
                                       {'range': [250, 500], 'color': Color2},
                                       {'range': [500, 750], 'color': Color3}, 
                                       {'range': [750, 1000], 'color': Color4}]})), row = 2, col = 3)

fig.update_layout(title = {'y':0.73,
                           'x':0.5,
                           'xanchor': 'center',
                           'yanchor': 'top',
                           'font': {'size': 30}})

fig.update_layout(font = {'color': "darkblue",
                          'family': "Times New Roman"},
                  title_text = "Top 5 Run Scorer")

fig.show()


# In[99]:


df2_most_wickets_sort_by_matches = df2_most_wickets.sort_values(by = ['Wkts'],
                                                        ascending = False)
plt.figure(figsize = (20, 8))
sns.barplot(x = df2_most_wickets["Wkts"],
            y = df2_most_wickets["Player"][:15],
            palette = "Blues");


# In[104]:


fig = px.histogram(df2_most_wickets, x = "Avg", color = "Avg",y = "Mat")
fig.show()


# In[110]:


import plotly.express as px

x = df2_most_wickets["Avg"]
y = df2_most_wickets["Mat"]

fig= px.scatter(df2_most_wickets, x, y, color = df2_most_wickets["Avg"], size = df2_most_wickets["Avg"],
                title = "Player_matches_wise_Averege_Distributions")
fig.show()


# ### After going thorugh the visual, we concluded that the player who played maximum matches doesn't mean that they have better 
# ### Avg, furthermore very selected players have best average rate as compared to all, such as - 13,14,18,20 in 10- 13 matches
# #### which is considerable and descent.

# In[117]:


import plotly.express as px
df2_most_wickets_5w = df2_most_wickets_5w.loc[:, ["Player", "5w"]]
df2_most_wickets_5w = df2_most_wickets_5w.sort_values(by = ['5w'], ascending = False)

fig = px.bar(df2_most_wickets_5w, x=df2_most_wickets_5w["Player"][:3], y=df2_most_wickets_5w["5w"][:3])
fig.show()


# In[112]:


df2_most_wickets_5w


# In[132]:


import plotly.express as px
df2_most_wickets_4w = df2_most_wickets_4w.loc[:, ["Player", "4w"]]
df2_most_wickets_4w = df2_most_wickets_4w.sort_values(by = ["4w"], ascending = True)
fig = px.bar(df, x=df2_most_wickets_4w["Player"][:10], y=df2_most_wickets_4w["4w"][:10])
fig.show()


# In[126]:


df2_most_wickets.head(3)


# In[133]:


fig = px.scatter(df2_most_wickets, x = "Player", y = "Avg", color = "Wkts")
fig.show()


# In[134]:


fig = px.scatter(df2_most_wickets, x = "Player", y = "Econ", color = "Wkts")
fig.show()


# In[135]:


player = []
for i in df2_most_wickets['Player'][:5]:
    player.append(i)
Color1="#72FFFF"
Color2='#00D7FF'
Color3="#0096FF"
Color4="#5800FF"
wkts = []
for i in df2_most_wickets['Wkts'][:5]:
    wkts.append(i)

fig = make_subplots(rows = 2, cols = 3, specs = [[{"type": "Indicator"},
                                                  {"type": "Indicator"},
                                                  {"type": "Indicator"}],
                                                 [{"type": "Indicator"},
                                                  {"type": "Indicator"},
                                                  {"type": "Indicator"}]])

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = wkts[0],
                             title = {'text': player[0],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 35],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 15], 'color': Color1},
                                       {'range': [15, 25], 'color': Color2},
                                       {'range': [25, 30], 'color': Color3},
                                       {'range': [30, 35], 'color': Color4}]})), row = 1, col = 1)



fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = wkts[1],
                             title = {'text': player[1],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 35],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 15], 'color': Color1},
                                       {'range': [15, 25], 'color': Color2},
                                       {'range': [25, 30], 'color': Color3}, 
                                       {'range': [30, 35], 'color': Color4}]})), row = 1, col = 3)

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = wkts[2],
                             title = {'text': player[2],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 35],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 15], 'color': Color1},
                                       {'range': [15, 25], 'color': Color2},
                                       {'range': [25, 30], 'color': Color3}, 
                                       {'range': [30, 35], 'color': Color4}]})), row = 2, col = 1)

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = wkts[3],
                             title = {'text': player[3],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 35],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 15], 'color': Color1},
                                       {'range': [15, 25], 'color': Color2},
                                       {'range': [25, 30], 'color': Color3},
                                       {'range': [30, 35], 'color': Color4}]})), row = 2, col = 2)

fig.add_trace((go.Indicator(mode = "gauge+number",
                             value = wkts[4],
                             title = {'text': player[4],
                                      'font': {'size': 24}},
                             gauge = {'axis': {'range': [None, 35],
                                               'tickwidth': 1,
                                               'tickcolor': "darkblue"},
                                      'bar': {'color': "black"},
                                      'bgcolor': "yellow",
                                      'borderwidth': 2,
                                      'bordercolor': "black",
                             'steps': [{'range': [0, 15], 'color': Color1},
                                       {'range': [15, 25], 'color': Color2},
                                       {'range': [25, 30], 'color': Color3}, 
                                       {'range': [30, 35], 'color': Color4}]})), row = 2, col = 3)

fig.update_layout(title = {'y':0.73,
                           'x':0.5,
                           'xanchor': 'center',
                           'yanchor': 'top',
                           'font': {'size': 30}})

fig.update_layout(font = {'color': "darkblue",
                          'family': "Times New Roman"},
                  title_text = "Top 5 Wicket Takers")

fig.show()


# ## This is the Final Visual of most wicket Taker of IPL 2022

# In[ ]:




