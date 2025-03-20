import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Load the CSV file

embeddings = pd.read_csv('df_with_validation_embeddings_MPnet.csv')[['Prompt ID', 'Prompt', 'Malicious (0/1)', 'Department', 'Source', 'Confidence Score', 'Embeddings']]
similarity = pd.read_csv('FINAL_validated_prompts_with_similarity_MPnet.csv')
#df = pd.read_csv('df_with_validation_embeddings_MPnet.csv')

df = embeddings.merge(similarity[['Prompt ID', 'Similarity Score']], on='Prompt ID', how='left')

print("Columns after merge:", df.columns)
print("Rows after merge:", len(df))

df = df[df['Similarity Score'] > 0.55]

df['Embeddings'] = df['Embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

print("Rows that match:", len(df[df['Source'] == 'Generated']))

# Apply preprocessing to the 'Embeddings' column
embeddings = np.stack(df[df['Source'] == 'Generated']['Embeddings'].values)

# Apply UMAP
reducer = umap.UMAP()
umap_embeddings = reducer.fit_transform(embeddings)

# Add the UMAP embeddings to the DataFrame
df.loc[df['Source'] == 'Generated', 'UMAP_1'] = umap_embeddings[:, 0]
df.loc[df['Source'] == 'Generated', 'UMAP_2'] = umap_embeddings[:, 1]

sns.set(style="whitegrid")

# Create the scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='UMAP_1', 
    y='UMAP_2', 
    hue='Malicious (0/1)',  # Color by the 'Malicious' column
    palette='Set1',      # Use a color palette
    data=df, 
    s=20,                  # Size of the points
    alpha = 0.5
)

# Add titles and labels
plt.title('UMAP Visualization of MPnet Embeddings', fontsize=16)
plt.xlabel('UMAP Dimension 1', fontsize=14)
plt.ylabel('UMAP Dimension 2', fontsize=14)
plt.legend(title='Malicious (0/1)', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position

plt.tight_layout()
plt.show()
