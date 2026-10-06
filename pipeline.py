import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# ---------- 1. Load ----------
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
raw = df.copy()
print("SHAPE", df.shape)
print("DTYPES\n", df.dtypes)
print("HEAD\n", df.head().to_string())

# ---------- 2. Explore ----------
print("DESCRIBE\n", df.describe().round(2).to_string())
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
mt = pd.DataFrame({"missing": miss, "pct": miss_pct})
print("MISSING\n", mt[mt.missing > 0].to_string())
print("DUPLICATE ROWS", df.duplicated().sum(), "DUP IDS", df.PassengerId.duplicated().sum())

plt.figure(figsize=(6,3.5))
m = mt[mt.missing>0].sort_values("pct", ascending=False)
sns.barplot(x=m.index, y=m.pct, color="#2f6db5")
for i,v in enumerate(m.pct): plt.text(i, v+1, f"{v}%", ha="center")
plt.ylabel("% missing"); plt.title("Missing values by column"); plt.tight_layout()
plt.savefig("fig/missing.png", dpi=160); plt.close()

# ---------- 3. Inconsistencies ----------
print("SEX", df.Sex.unique(), "EMB", df.Embarked.unique(), "PCLASS", sorted(df.Pclass.unique()), "SURV", sorted(df.Survived.unique()))
print("WHITESPACE names", (df.Name != df.Name.str.strip()).sum())
df["Name"] = df.Name.str.strip()
print("WHITESPACE after", (df.Name != df.Name.str.strip()).sum())
print("FARE ZERO", (df.Fare == 0).sum())
print("AGE<1", (df.Age < 1).sum(), df.loc[df.Age<1, "Age"].tolist())
print("AGE FRACTIONAL (>=1)", ((df.Age % 1 != 0) & (df.Age >= 1)).sum())
print("Fare negative", (df.Fare<0).sum(), "Age negative", (df.Age<0).sum())
print("MAX SibSp", df.SibSp.max(), "MAX Parch", df.Parch.max())
dupT = df.Ticket.duplicated(keep=False).sum()
print("rows sharing a ticket", dupT)
print("SibSp=8 rows\n", df[df.SibSp==8][["Name","Ticket","SibSp","Parch"]].head(3).to_string())

# ---------- 4. Missing values ----------
# 4a Embarked
print("EMBARKED missing rows\n", df[df.Embarked.isnull()][["Name","Pclass","Fare","Embarked"]].to_string())
emb_mode = df.Embarked.mode()[0]
df["Embarked"] = df["Embarked"].fillna(emb_mode)
print("EMB MODE", emb_mode)

# 4b Cabin
df["HasCabin"] = df.Cabin.notnull().astype(int)
df["Deck"] = df.Cabin.str[0].fillna("Unknown")
print("DECK counts\n", df.Deck.value_counts().to_string())
print("SURV by HasCabin\n", df.groupby("HasCabin").Survived.mean().round(3).to_string())

# 4c Age via Title + Pclass median
df["Title"] = df.Name.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
print("TITLES raw\n", df.Title.value_counts().to_string())
title_map = {"Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Rare","the Countess":"Rare","Capt":"Rare","Col":"Rare",
             "Don":"Rare","Dr":"Rare","Jonkheer":"Rare","Major":"Rare","Rev":"Rare","Sir":"Rare"}
df["Title"] = df.Title.replace(title_map)
print("TITLES grouped\n", df.Title.value_counts().to_string())
age_before = df.Age.copy()
grp_med = df.groupby(["Title","Pclass"]).Age.transform("median")
print("GROUP MEDIAN table\n", df.groupby(["Title","Pclass"]).Age.median().round(1).unstack().to_string())
df["Age_Missing"] = df.Age.isnull().astype(int)
df["Age"] = df.Age.fillna(grp_med)
df["Age"] = df.Age.fillna(df.Age.median())
print("AGE NA after", df.Age.isnull().sum())
print("AGE stats before/after", age_before.mean().round(2), age_before.std().round(2), age_before.median(), "|", df.Age.mean().round(2), df.Age.std().round(2), df.Age.median())

fig, ax = plt.subplots(1,2, figsize=(9,3.5), sharey=True)
sns.histplot(age_before.dropna(), bins=30, color="#2f6db5", ax=ax[0]); ax[0].set_title("Age - before imputation")
sns.histplot(df.Age, bins=30, color="#2a9d8f", ax=ax[1]); ax[1].set_title("Age - after imputation")
plt.tight_layout(); plt.savefig("fig/age_before_after.png", dpi=160); plt.close()

# ---------- 5. Outliers ----------
def iqr_bounds(s):
    q1,q3 = s.quantile([.25,.75]); i = q3-q1
    return q1-1.5*i, q3+1.5*i
for c in ["Age","Fare","SibSp","Parch"]:
    lo,hi = iqr_bounds(df[c]); n = ((df[c]<lo)|(df[c]>hi)).sum()
    print(f"IQR {c}: lo={lo:.2f} hi={hi:.2f} outliers={n}")
print("FARE top\n", df.nlargest(5,"Fare")[["Name","Pclass","Fare","Ticket"]].to_string())
print("FARE skew before", round(df.Fare.skew(),2))
lo,hi = iqr_bounds(df.Fare)
cap = df.Fare.quantile(.99)
print("FARE 99th pct", round(cap,2))
df["Fare_Raw"] = df.Fare
df["Fare_Zero"] = (df.Fare == 0).astype(int)
df["Fare"] = df.Fare.clip(upper=cap)
df["Fare_Log"] = np.log1p(df.Fare_Raw)
print("FARE skew after cap", round(df.Fare.skew(),2), "log", round(df.Fare_Log.skew(),2))
print("FARE ZERO details\n", df[df.Fare_Raw==0][["Name","Pclass","Embarked"]].head(4).to_string())

fig, ax = plt.subplots(1,3, figsize=(11,3.4))
sns.boxplot(y=raw.Fare, ax=ax[0], color="#2f6db5"); ax[0].set_title("Fare - raw")
sns.boxplot(y=df.Fare, ax=ax[1], color="#2a9d8f"); ax[1].set_title("Fare - capped @ 99th pct")
sns.boxplot(y=df.Fare_Log, ax=ax[2], color="#e9c46a"); ax[2].set_title("Fare - log1p")
plt.tight_layout(); plt.savefig("fig/fare_box.png", dpi=160); plt.close()

# ---------- 6. Feature engineering / encoding / scaling ----------
df["FamilySize"] = df.SibSp + df.Parch + 1
df["IsAlone"] = (df.FamilySize == 1).astype(int)
df["Sex_Code"] = df.Sex.map({"male":0,"female":1})
df = pd.get_dummies(df, columns=["Embarked","Title"], prefix=["Emb","Title"], dtype=int)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[["Age_Scaled","Fare_Scaled"]] = sc.fit_transform(df[["Age","Fare_Log"]])
print("SCALED means", df[["Age_Scaled","Fare_Scaled"]].mean().round(3).tolist(), df[["Age_Scaled","Fare_Scaled"]].std().round(3).tolist())
print("FINAL SHAPE", df.shape)
print("FINAL COLS", list(df.columns))
print("FINAL NA", df.drop(columns=["Cabin"]).isnull().sum().sum())

# ---------- 7. Impact ----------
r = raw.copy(); r["Sex_Code"] = r.Sex.map({"male":0,"female":1})
c_raw = r[["Survived","Age","Fare","Pclass","Sex_Code"]].corr()["Survived"].round(3)
c_new = df[["Survived","Age","Fare_Raw","Fare_Log","Pclass","Sex_Code","HasCabin","FamilySize"]].corr()["Survived"].round(3)
print("CORR raw\n", c_raw.to_string()); print("CORR clean\n", c_new.to_string())
print("ROWS dropped by listwise deletion on Age", raw.dropna(subset=["Age"]).shape[0], "vs", len(raw))

# quick model impact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
feats_clean = ["Pclass","Sex_Code","Age","Fare_Log","FamilySize","IsAlone","HasCabin"] + [c for c in df.columns if c.startswith("Emb_") or c.startswith("Title_")]
Xc, yc = df[feats_clean], df.Survived
sc_clean = cross_val_score(LogisticRegression(max_iter=2000), Xc, yc, cv=5).mean()
rr = raw.copy(); rr["Sex_Code"] = rr.Sex.map({"male":0,"female":1}); rr = rr.dropna(subset=["Age"])
sc_naive = cross_val_score(LogisticRegression(max_iter=2000), rr[["Pclass","Sex_Code","Age","Fare","SibSp","Parch"]], rr.Survived, cv=5).mean()
print("CV ACC naive (drop NA rows)", round(sc_naive,4), "n=", len(rr), "| cleaned", round(sc_clean,4), "n=", len(df))

fig, ax = plt.subplots(figsize=(6.5,5))
cols = ["Survived","Pclass","Sex_Code","Age","Fare_Log","FamilySize","HasCabin"]
sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix (cleaned data)"); plt.tight_layout(); plt.savefig("fig/corr.png", dpi=160); plt.close()

df.drop(columns=["Cabin"]).to_csv("titanic_cleaned.csv", index=False)
