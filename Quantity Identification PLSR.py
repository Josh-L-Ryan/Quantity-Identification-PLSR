from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.ticker as ticker

# Params which define the appearance of graphs
params = {
    'axes.labelsize': 12,  # Set size for both x and y axis labels
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.minor.size': 3,   # minor tick length
    'ytick.minor.size': 3,  
    'xtick.major.size': 5,   # major tick length
    'ytick.major.size': 5,
    'xtick.major.width': 1.2,   # major tick width
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,   # minor tick width
    'ytick.minor.width': 0.8
}

plt.rcParams.update(params)


# Read in raw UV-vis data files
df = pd.read_csv(r"E:\Glass project\UV-Vis data\Chalc Bif binary calibration solutions.csv")
dfchalc = pd.read_csv(r"E:\Glass project\UV-Vis data\Chalc EtOH calibrations.csv")
dfbif = pd.read_csv(r"E:\Glass project\UV-Vis data\Bif EtOH calibrations.csv")
dfextra_abs = pd.read_csv(r"E:\Glass project\UV-Vis data\Chalc Bif PLSR extra data.csv", skiprows=1)
dfextra_concs = pd.read_csv(r"E:\Glass project\UV-Vis data\PLSR extra data concs.csv", skiprows=0)

# Note: y = concentration values, X = absorbance values, wl = wavelengths

# Concentration of each sample in mg/mL [Bif, Chalc]
y3 = [[0.0015251, 0.0059957], [0.0030747, 0.0035872], [0.0044853, 0.001905], [0.0061518, 0.0049984], [0.0074750, 0.0024836], [0.00050117, 0.0029340], 
[0.0010598, 0.00062342], [0.0052686, 0.0072707], [0.0079505, 0.0017509], [0.0089605, 0.0041647], [0.0, 0.001936], [0.0, 0.0024964], [0, 0.0035986], [0, 0.0047418], [0, 0.0059862], 
[0.0014485, 0], [0.0029085, 0], [0.0044633, 0], [0.0058180, 0], [0.0078753, 0]] #[Bif, Chalc]

# Handling of additional data
extradata_chalc = dfextra_actualconcs.iloc[:, 0]
extradata_bif = dfextra_actualconcs.iloc[:, 1]

for i in range(30):
    y3.append([extradata_bif.iloc[i], extradata_chalc.iloc[i]])

y = pd.DataFrame(y3)


# Wavelengths for each sample
wl = df.iloc[:, 0] 
wl = np.array([wl]*10)

# Data for binary chalc/bif in EtOH data
X1 = np.array(df.iloc[lambda x: x.index % 2 != 0, 1])
X2 = np.array(df.iloc[lambda x: x.index % 2 != 0, 3])
X3 = np.array(df.iloc[lambda x: x.index % 2 != 0, 5])
X4 = np.array(df.iloc[lambda x: x.index % 2 != 0, 7])
X5 = np.array(df.iloc[lambda x: x.index % 2 != 0, 9])
X6 = np.array(df.iloc[lambda x: x.index % 2 != 0, 11])
X7 = np.array(df.iloc[lambda x: x.index % 2 != 0, 13])
X8 = np.array(df.iloc[lambda x: x.index % 2 != 0, 15])
X9 = np.array(df.iloc[lambda x: x.index % 2 != 0, 17])
X10 = np.array(df.iloc[lambda x: x.index % 2 != 0, 19])

# Data for pure 4-hydroxychalcone in EtOH
Xchalc1 = np.array(dfchalc.iloc[:, 1])
Xchalc2 = np.array(dfchalc.iloc[:, 3])
Xchalc3 = np.array(dfchalc.iloc[:, 5])
Xchalc4 = np.array(dfchalc.iloc[:, 7])
Xchalc5 = np.array(dfchalc.iloc[:, 9])

# Data for pure bifonazole in EtOH
Xbif1 = np.array(dfbif.iloc[:, 1])
Xbif2 = np.array(dfbif.iloc[:, 3])
Xbif3 = np.array(dfbif.iloc[:, 5])
Xbif4 = np.array(dfbif.iloc[:, 7])
Xbif5 = np.array(dfbif.iloc[:, 9])

# Actual concs from extra data acquired 15-06-22
extradata = dfextra_abs.iloc[lambda x: x.index % 2 != 0, 1::2]

# Combine all the UV data into one array
X = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Xchalc1, Xchalc2, Xchalc3, Xchalc4, Xchalc5, Xbif1, Xbif2, Xbif3, Xbif4, Xbif5]
X = np.array(X)

X = [np.array(X), np.array(extradata.T)]
X = np.concatenate(X)


# Split the UV-vis data into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(X_train[0])

plt.plot(X_train[0])
plt.show()


# Instantiate a PLSR model and determine the optimum number of LVs required. This code can be commented out after the values are known.
#RMSECV_vs_LVs = []

#for i in range(1, 21):
    # Producing a PLSR model of the UV data using i latent variables
#    pls = PLSRegression(n_components=i)
#    pls.fit(X_train, y_train)

#    y_pred = pls.predict(X_test)
    #print(pls.predict(X_test))
    #print(pls.score(X_test, y_test))
    #print(y_test)

    # Paper for PLSR performance stuff: https://doi.org/10.1155/2017/6274178

    # 5-fold Cross-validation and r2 Performance
#    r2_score = cross_val_score(pls, X_train, y_train, cv=5, scoring="r2") #The score metric is r^2 (crossvalscore uses the default for the model u put in, for PLSR = r^2)
#    print("r2 score for each fold =", r2_score)
#    print("%0.2f accuracy with a standard deviation of %0.2f" % (r2_score.mean(), r2_score.std()))

    # 5-fold Cross-validation and RMSECV Performance - computes a RMSE score for each fold of cross validation.
#    RMSE_score = cross_val_score(pls, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
    # Take the mean of the RMSE to get an overall RMSECV score for this number of latent variables.
#    print("RMSECV for {} LVs".format(i), float(abs(np.mean(RMSE_score))))

#    RMSECV_vs_LVs.append(float(abs(np.mean(RMSE_score))))

#print(RMSECV_vs_LVs)

# RMSECV values vs number of LVs as determined in the code block above.
RMSECV_vs_LVs = [0.001744504426094757, 0.0009667625056333114, 0.0003813629761260281, 0.00031246901158107606, 0.0003245358067377145, 0.00033063108016803254, 0.0003309786731921879, 0.000320533431077412, 0.00032721138871296215, 0.000320374784299395, 0.000309302957644587, 0.0003082880210351866, 0.0003120849052568486, 0.0003152379776260702, 0.0003157931668027575, 0.0003142530902222053, 0.00031390732484352446, 0.00031392939948264034, 0.00031372929576540593, 0.00031376741960200235]

# Plot RMSECV vs #LVs
fig, ax = plt.subplots()

ax.plot(range(1,21), [x*1000 for x in RMSECV_vs_LVs],  color='#000000')
ax.scatter(range(1,21), [x*1000 for x in RMSECV_vs_LVs],  color='#000000')

# Adjust axes properties and appearance.
ax.set_xlabel("LVs")
ax.set_ylabel("RMSECV × 10$^{-3}$")
ax.set_xticks(range(0, 23, 2))
#ax.tick_params(axis="both", which="major", labelsize=26, length=14, width=2)
#ax.tick_params(axis="both", which="minor", labelsize=26, length=10, width=2)

ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins='5'))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins='5'))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

ax.set_xlim(left=0, right=20.5)
ax.set_ylim(top=2.0, bottom=0)

#[x.set_linewidth(2) for x in ax.spines.values()]

# Show and save the figure
plt.subplots_adjust(left=0.105, bottom=0.102, right=0.979, top=0.97, wspace=0.2, hspace=0.2)
plt.savefig(fname=r"E:\Glass project\PLSR Figs\RMSECV vs LVs.png", dpi=600)
plt.show()

# Additional validation data
test_data = pd.read_csv(r"E:\Glass project\PLSR Figs\Chalc Bif glass.csv", skiprows=1)
test_wavelength = test_data.iloc[:, 8]
test_abs = np.array(test_data.iloc[:600, 7:10:2])
test_abs = test_abs.T.reshape(2, 600)

