#!/usr/bin/env python
# coding: utf-8

# <div style="display: flex; background-color: RGB(255,114,0);" >
# <h1 style="margin: auto; padding: 30px; ">MISSION 1 — AMÉLIORATION DU LIVRABLE P6 GRÂCE À L'IA</h1>
# </div>
# 
# # OBJECTIF DE CETTE PARTIE
# 
# Ce notebook reprend le livrable du **Projet 6 "Bottleneck"** (analyse du stock et des ventes) et l'améliore grâce à l'IA, de manière **critique et documentée**, conformément à la Mission 1 du projet de fin de parcours OpenClassrooms BIA.
# 
# Deux axes d'amélioration ont été retenus à l'issue de la veille technologique (voir document de documentation associé) :
# 
# 1. **Détection d'anomalies avancée** : Isolation Forest (multivarié) vs Z-score/IQR (méthode P6, univariée)
# 2. **Segmentation du catalogue produits** : K-Means (ML) vs segmentation à règles fixes (méthode P6, seuils manuels)
# 
# Pour chaque axe : hypothèse testée → test des options → résultats chiffrés → décision justifiée → limites.
# 
# Le point de départ est le fichier `Bottleneck_donnees_consolidees_octobre.xlsx`, exporté à la fin du notebook P6 (825 produits ERP, 712 après filtrage sur le périmètre commercial web).

# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Étape 0 - Chargement du livrable P6</h2>
# </div>

# In[1]:


#Importation des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
RANDOM_STATE = 42  #seed fixée pour la reproductibilité (cahier des charges, critère KPI opérationnel)

# In[2]:


#Chargement du fichier consolidé, sortie du notebook P6
df = pd.read_excel("Bottleneck_donnees_consolidees_octobre.xlsx")
print("Dataset chargé : {} produits, {} colonnes".format(df.shape[0], df.shape[1]))
df.head(3)

# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Étape 6 - Détection d'anomalies avancée : Isolation Forest vs Z-score/IQR</h2>
# </div>
# 
# <div style="border: 1px solid RGB(255,114,0);" >
# <h3 style="margin: auto; padding: 15px; color: RGB(255,114,0); ">Hypothèse testée</h3>
# </div>
# 
# Le Z-score et l'IQR du notebook P6 ne détectent des anomalies que **sur le prix, variable par variable**. Hypothèse : une méthode multivariée (Isolation Forest) est capable de repérer des anomalies invisibles au prix seul — par exemple une combinaison stock/rotation/marge incohérente — et donc d'apporter une valeur ajoutée réelle par rapport à la méthode P6.

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 20px; color: RGB(51,165,182); ">6.1 - Rappel : méthode P6 (Z-score sur le prix, déjà en place)</h3>
# </div>

# In[3]:


#Rappel des anomalies déjà détectées en P6 (Z-score sur le prix uniquement)
n_anomalies_zscore = (df['z_score_price'].abs() > 3).sum()
print("Méthode P6 (Z-score prix) : {} produits détectés comme anomalies sur {} ({:.1f}% du catalogue)".format(
    n_anomalies_zscore, len(df), n_anomalies_zscore/len(df)*100))

#Rappel IQR (également présent dans le P6)
Q1, Q3 = df['price'].quantile([0.25, 0.75])
IQR = Q3 - Q1
borne_basse, borne_haute = Q1 - 1.5*IQR, Q3 + 1.5*IQR
n_anomalies_iqr = ((df['price'] < borne_basse) | (df['price'] > borne_haute)).sum()
print("Méthode P6 (IQR prix)    : {} produits détectés".format(n_anomalies_iqr))

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 20px; color: RGB(51,165,182); ">6.2 - Test : Isolation Forest (multivarié)</h3>
# </div>

# In[4]:


#Isolation Forest sur 5 variables conjointes : price, purchase_price, stock_quantity, total_sales, taux_marge
#Standardisation préalable : nécessaire pour que les variables à forte échelle (ca) ne dominent pas les autres
cols_if = ['price', 'purchase_price', 'stock_quantity', 'total_sales', 'taux_marge']
X_if = df[cols_if].copy()
print("Valeurs manquantes sur les variables utilisées :")
print(X_if.isna().sum())

scaler_if = StandardScaler()
X_if_scaled = scaler_if.fit_transform(X_if)

#contamination=0.05 : taux d'anomalies attendu, cohérent avec l'ordre de grandeur des ~4-5% d'outliers
#déjà identifiés par l'IQR sur le prix seul (critère KPI du cahier des charges)
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=RANDOM_STATE)
df['anomaly_if'] = iso_forest.fit_predict(X_if_scaled)          # -1 = anomalie, 1 = normal
df['score_anomalie_if'] = iso_forest.decision_function(X_if_scaled)  # score continu (plus bas = plus anormal)

n_anomalies_if = (df['anomaly_if'] == -1).sum()
print("\nIsolation Forest : {} produits détectés comme anomalies sur {} ({:.1f}% du catalogue)".format(
    n_anomalies_if, len(df), n_anomalies_if/len(df)*100))

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 20px; color: RGB(51,165,182); ">6.3 - Comparaison et cas concret de divergence</h3>
# </div>

# In[5]:


#Recoupement : combien d'anomalies Isolation Forest sont AUSSI détectées par le Z-score prix (méthode P6) ?
overlap = df[(df['anomaly_if'] == -1) & (df['z_score_price'].abs() > 3)]
print("Anomalies détectées par les DEUX méthodes : {}".format(len(overlap)))

#Cas intéressant : anomalies détectées UNIQUEMENT par Isolation Forest (invisibles au Z-score prix)
uniquement_if = df[(df['anomaly_if'] == -1) & (df['z_score_price'].abs() <= 3)].copy()
print("Anomalies détectées UNIQUEMENT par Isolation Forest (prix pourtant normal) : {}".format(len(uniquement_if)))
print("\nRépartition par type de produit de ces anomalies 'invisibles' au prix seul :")
print(uniquement_if['product_type'].value_counts())

# In[6]:


#Zoom sur ces produits : qu'ont-ils de commun ?
print("Stock moyen (catalogue entier)         : {:.1f} bouteilles".format(df['stock_quantity'].mean()))
print("Stock moyen (anomalies IF uniquement)  : {:.1f} bouteilles".format(uniquement_if['stock_quantity'].mean()))
print()
display(uniquement_if.sort_values('score_anomalie_if')[
    ['product_id','post_title','product_type','price','stock_quantity','total_sales','taux_marge','score_anomalie_if']
].head(10))

#=> RÉSULTAT : les 24 produits détectés uniquement par Isolation Forest sont à 71% des Champagnes
#(17/24), avec un stock moyen de 97 bouteilles contre 23 pour le catalogue entier (x4).
#Leur PRIX est parfaitement normal (sinon le Z-score P6 les aurait repérés) : c'est la COMBINAISON
#prix élevé + stock disproportionné + rotation lente qui est anormale, pas une variable isolée.
#=> C'est exactement le type de signal que la méthode P6 (univariée) ne peut pas voir par construction.

# In[7]:


#Visualisation : stock vs marge, anomalies mises en évidence
fig = px.scatter(df, x='stock_quantity', y='taux_marge', color=df['anomaly_if'].map({1:'Normal', -1:'Anomalie (Isolation Forest)'}),
                  hover_data=['post_title','price','total_sales'],
                  title="Anomalies multivariées (Isolation Forest) — stock vs taux de marge",
                  labels={'stock_quantity':'Stock (bouteilles)', 'taux_marge':'Taux de marge (%)', 'color':'Statut'},
                  color_discrete_map={'Normal':'#33A5B6','Anomalie (Isolation Forest)':'#FF7200'})
fig.show()

# <div style="border: 1px solid RGB(255,114,0);" >
# <h3 style="margin: auto; padding: 15px; color: RGB(255,114,0); ">Décision</h3>
# </div>
# 
# **Isolation Forest est retenu en complément du Z-score/IQR, pas en remplacement.**
# 
# Justification :
# - Isolation Forest détecte **24 anomalies supplémentaires** invisibles à la méthode P6 (prix normal, mais stock disproportionné) — majoritairement des Champagnes de prestige avec ~4x le stock moyen du catalogue. C'est une vraie valeur métier : ce sont des candidats à une **immobilisation de trésorerie non détectée** par la seule analyse du prix.
# - Le Z-score/IQR reste conservé car son interprétabilité "une variable, un seuil clair" est précieuse pour une explication rapide au CODIR (Nicolas), alors que le score Isolation Forest nécessite une lecture plus experte.
# - **Limite assumée** : le paramètre `contamination=0.05` est un choix arbitraire (pas de vérité terrain sur le vrai taux d'anomalies) ; il devra être ajusté avec Nicolas selon le nombre de vérifications manuelles que l'équipe peut absorber chaque mois.

# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Étape 7 - Segmentation du catalogue : K-Means vs règles fixes</h2>
# </div>
# 
# <div style="border: 1px solid RGB(255,114,0);" >
# <h3 style="margin: auto; padding: 15px; color: RGB(255,114,0); ">Hypothèse testée</h3>
# </div>
# 
# Le P6 classe déjà les produits un critère à la fois (top CA, top quantité, règle 20/80, mois de stock). Hypothèse : une segmentation combinant **simultanément marge, rotation et CA** (K-Means) fait émerger des familles de produits plus actionnables qu'une lecture croisée manuelle de plusieurs classements séparés.

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 20px; color: RGB(51,165,182); ">7.1 - Rappel : méthode P6 (règles fixes, ex. 20/80 en CA)</h3>
# </div>

# In[8]:


#Rappel de la règle 20/80 déjà calculée en P6
df_tri_ca = df.sort_values('ca_par_article', ascending=False).reset_index(drop=True)
df_tri_ca['part_ca_cumule_pct'] = (df_tri_ca['ca_par_article'] / df_tri_ca['ca_par_article'].sum() * 100).cumsum()
nb_articles_80 = (df_tri_ca['part_ca_cumule_pct'] <= 80).sum() + 1
print("Méthode P6 (règle 20/80) : {} articles génèrent 80% du CA, soit {:.1f}% du catalogue".format(
    nb_articles_80, nb_articles_80/len(df_tri_ca)*100))
print("=> Un seul critère (le CA) est utilisé : deux produits avec le même CA mais des marges opposées")
print("   se retrouvent traités de façon identique par cette règle.")

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 20px; color: RGB(51,165,182); ">7.2 - Test 1 : K-Means sur variables brutes</h3>
# </div>

# In[9]:


#Variables de segmentation : taux de marge, mois de stock (rotation), CA par article
seg_cols = ['taux_marge', 'mois_de_stock', 'ca_par_article']
X_seg = df[seg_cols].copy()
scaler_seg = StandardScaler()
X_seg_scaled = scaler_seg.fit_transform(X_seg)

#Recherche du k optimal : méthode du coude (inertie) + score de silhouette, k de 2 à 6
resultats_k = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_seg_scaled)
    tailles = pd.Series(labels).value_counts().to_dict()
    resultats_k.append({'k': k, 'inertie': km.inertia_, 'silhouette': silhouette_score(X_seg_scaled, labels), 'tailles_clusters': tailles})

df_resultats_k = pd.DataFrame(resultats_k)
print(df_resultats_k.to_string(index=False))
#=> Le k=2 a le meilleur score de silhouette (0.77) mais isole un unique gros segment (683 produits)
#contre 29 : trop grossier pour être actionnable en pratique (une seule frontière, peu de granularité).
#=> Avec k=4, un cluster ne contient qu'1 SEUL produit (l'outlier CA=2475€, un best-seller ponctuel) :
#K-Means est ici sensible à cette valeur extrême malgré la standardisation.

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 20px; color: RGB(51,165,182); ">7.3 - Test 2 : K-Means sur variables transformées (log) pour stabiliser le clustering</h3>
# </div>
# 
# `mois_de_stock` et `ca_par_article` sont fortement asymétriques (quelques produits extrêmes). Une transformation logarithmique (`log1p`) est testée pour réduire l'effet des valeurs extrêmes sur les centroïdes — une pratique recommandée pour le clustering sur données de vente asymétriques.

# In[10]:


#Transformation log1p pour réduire l'asymétrie (mois_de_stock et ca_par_article ont des queues lourdes)
X_seg_log = pd.DataFrame({
    'taux_marge': df['taux_marge'],
    'log_mois_stock': np.log1p(df['mois_de_stock']),
    'log_ca': np.log1p(df['ca_par_article']),
})
X_seg_log_scaled = scaler_seg.fit_transform(X_seg_log)

resultats_k_log = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_seg_log_scaled)
    tailles = pd.Series(labels).value_counts().to_dict()
    resultats_k_log.append({'k': k, 'silhouette': silhouette_score(X_seg_log_scaled, labels), 'tailles_clusters': tailles})

df_resultats_k_log = pd.DataFrame(resultats_k_log)
print(df_resultats_k_log.to_string(index=False))
#=> Avec la transformation log, k=4 conserve un score de silhouette correct (0.35, > seuil de 0.3 fixé
#au cahier des charges) SANS créer de cluster singleton : 4 segments équilibrés et interprétables.
#=> Décision : on retient k=4 sur variables log-transformées.

# <div style="border: 1px solid RGB(51,165,182);" >
# <h3 style="margin: auto; padding: 20px; color: RGB(51,165,182); ">7.4 - Segmentation finale retenue : K-Means k=4 (log)</h3>
# </div>

# In[11]:


#Application du K-Means final (k=4, variables log-transformées, seed fixée)
km_final = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
df['segment'] = km_final.fit_predict(X_seg_log_scaled)

profil_segments = df.groupby('segment').agg(
    nb_produits=('product_id', 'count'),
    marge_moyenne_pct=('taux_marge', 'mean'),
    mois_stock_moyen=('mois_de_stock', 'mean'),
    ca_moyen_article=('ca_par_article', 'mean'),
    ca_total_segment=('ca_par_article', 'sum'),
).round(1).sort_values('mois_stock_moyen')

#Attribution d'un nom métier à chaque segment sur la base de son profil
noms_segments = {}
for seg_id, row in profil_segments.iterrows():
    if row['mois_stock_moyen'] > 10:
        noms_segments[seg_id] = "Stock à risque (immobilisé)"
    elif row['mois_stock_moyen'] < 0.5:
        noms_segments[seg_id] = "Stock dormant (aucune vente en octobre)"
    elif row['marge_moyenne_pct'] > 38:
        noms_segments[seg_id] = "Coeur de gamme rentable"
    else:
        noms_segments[seg_id] = "Volume / meilleures ventes"

profil_segments['nom_segment'] = profil_segments.index.map(noms_segments)
df['nom_segment'] = df['segment'].map(noms_segments)
display(profil_segments)

# In[12]:


#Visualisation de la segmentation finale
fig = px.scatter(df, x='mois_de_stock', y='taux_marge', size='ca_par_article', color='nom_segment',
                  hover_data=['post_title','product_type'],
                  title="Segmentation du catalogue (K-Means, k=4) — rotation vs marge, taille = CA",
                  labels={'mois_de_stock':'Mois de stock (rotation)', 'taux_marge':'Taux de marge (%)', 'nom_segment':'Segment'},
                  color_discrete_sequence=['#33A5B6','#FF7200','#2E4053','#B8CCE8'])
fig.show()

# In[13]:


#Répartition du CA par segment (vision CODIR)
ca_par_segment = df.groupby('nom_segment')['ca_par_article'].sum().sort_values(ascending=False)
fig2 = px.pie(values=ca_par_segment.values, names=ca_par_segment.index,
              title="Répartition du chiffre d'affaires par segment produit",
              color_discrete_sequence=['#33A5B6','#FF7200','#2E4053','#B8CCE8'])
fig2.show()

# <div style="border: 1px solid RGB(255,114,0);" >
# <h3 style="margin: auto; padding: 15px; color: RGB(255,114,0); ">Décision</h3>
# </div>
# 
# **K-Means (k=4, variables log-transformées) est retenu, en complément de la règle 20/80** conservée pour sa lisibilité immédiate.
# 
# Justification :
# - Le score de silhouette (0.35) dépasse le seuil de 0,3 fixé au cahier des charges, et les 4 segments sont équilibrés (aucun singleton), contrairement au test sur variables brutes.
# - Contrairement à la règle 20/80 (un seul critère : le CA), le K-Means fait émerger un segment **"Stock à risque"** de 28 produits — marge la plus faible du catalogue (28,5%) ET rotation la plus lente (17,6 mois de stock en moyenne) — invisible dans un classement CA seul, puisque leur CA individuel n'est pas anormalement bas.
# - **Limite assumée** : le nombre de clusters (k=4) reste un choix parmi plusieurs arbitrages possibles (k=3 et k=5 donnent des scores de silhouette proches) ; les noms de segments sont une interprétation métier, à valider avec Nicolas avant diffusion au CODIR.

# <div style="background-color: RGB(51,165,182);" >
# <h2 style="margin: auto; padding: 20px; color:#fff; ">Étape 8 - Synthèse recruteur / CODIR</h2>
# </div>
# 
# ### Ce qui a été fait
# Le livrable P6 (fiabilisation des données + analyses univariées) a été enrichi par deux méthodes IA testées de façon comparative et documentée :
# - **Isolation Forest** (anomalies multivariées) vs Z-score/IQR (P6, univarié)
# - **K-Means** (segmentation multivariée) vs règle 20/80 (P6, univariée)
# 
# ### Impact / recommandations pour Nicolas et le CODIR
# 1. **24 produits à vérifier en priorité** (Champagnes majoritairement) : prix normal mais stock ~4x supérieur à la moyenne du catalogue → risque d'immobilisation de trésorerie non visible dans l'analyse P6 initiale.
# 2. **28 produits en "Stock à risque"** (segment K-Means) : marge la plus faible ET rotation la plus lente du catalogue → candidats à une opération commerciale ciblée (déstockage, mise en avant) plutôt qu'un réapprovisionnement.
# 3. Les deux méthodes P6 (Z-score/IQR, règle 20/80) restent utiles et sont conservées : rapides à expliquer, elles servent de première lecture ; les méthodes IA apportent la profondeur multivariée que l'analyse manuelle ne peut pas couvrir à elle seule.
# 
# ### Limites
# - Isolation Forest : taux de contamination fixé arbitrairement (5%), à recalibrer avec le retour terrain de Nicolas.
# - K-Means : nombre de segments (k=4) est un compromis, pas une vérité unique ; sensible aux valeurs extrêmes malgré la transformation log appliquée.
# - Aucune donnée client disponible dans Bottleneck : la segmentation reste au niveau produit, pas au niveau comportement d'achat individuel.
# 
# ### Prochaines pistes
# - Recalibrer `contamination` avec Nicolas sur 2-3 mois de données pour fiabiliser le seuil.
# - Étendre la segmentation à un historique multi-mois (au-delà d'octobre seul) pour distinguer saisonnalité et véritable stock dormant.
# - Si des données clients deviennent disponibles, appliquer une segmentation RFM complémentaire à la segmentation produit actuelle.
