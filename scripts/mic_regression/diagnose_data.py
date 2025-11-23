
df = pd.read_csv("data/dramp_training_dataset.csv")
low = df.var().sort_values().head(20)
print(low)
print(pca.explained_variance_ratio_.sum())
print(df.corr()["log_mean_MIC"].abs().sort_values(ascending=False).head(30))
